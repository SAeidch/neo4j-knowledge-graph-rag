"""
rag_kg.py - Problem 2: using the knowledge graph (Problem 1) as RAG context

Pipeline (matches the assignment):
  1. Get a user question (command line or interactive prompt).
  2. Retrieve relevant data from the Neo4j graph built by kg_builder.py:
       a. turn the question into search terms (keywords + relationship hints)
       b. Cypher query 1: find the entities whose names match the terms. Default
          ranking ("hits"): most matched terms first, then most connections, done
          inside Cypher. Optional ranking ("specific", --ranking specific): exact
          name matches and rare terms first, done in Python (see rank_entities)
       c. Cypher query 2: fetch the relationships around those entities, with
          their evidence sentences and source documents
       d. Cypher query 3 (fallback only): search the stored chunk text when the
          graph has no matching entities/relationships
  3. Convert the retrieved data into plain text and add it as context to the
     user's question.
  4. Call a local Ollama model (default: tinyllama) to generate the answer.

Everything that happens (terms, Cypher queries and parameters, results, context,
the exact prompt, the response) is printed and appended to rag_log.txt so it can
be copied into the report.

Setup:
  - Neo4j instance running, containing the graph built by kg_builder.py
  - Ollama running, and the model pulled:  ollama pull tinyllama
  - Password via environment variable (PowerShell):
        $env:NEO4J_PASSWORD="your-password"

Usage:
  python rag_kg.py "How do plants remove CO2 from indoor air?"
  python rag_kg.py                      (interactive mode)
  python rag_kg.py --compare "question" (also show the answer WITHOUT graph context)
"""

import argparse
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import ollama
from neo4j import GraphDatabase

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "YOUR_PASSWORD_HERE")

LLM_MODEL = "tinyllama"       # any model you have pulled with ollama
TOP_K_ENTITIES = 5            # how many matching entities to expand
MAX_CANDIDATES = 500          # cap on candidate entities fetched by Query 1
MAX_RELATIONSHIPS = 12        # how many relationships to put in the context
MAX_CHUNKS = 2                # chunks used by the fallback query
MAX_EVIDENCE_CHARS = 220      # evidence sentences are cut to this length
MAX_CONTEXT_CHARS = 3000      # TinyLlama only has a 2048-token window, so keep the context short

LOG_FILE = Path(__file__).with_name("rag_log.txt")

# --------------------------------------------------------------------------
# Cypher queries. Parameters ($terms, $keys, ...) are always passed separately,
# never pasted into the query text.
# --------------------------------------------------------------------------

# Query 1 (default "hits" ranking): entities whose name contains at least one search term,
# ranked by (a) how many different terms they match and (b) how connected they are.
Q_FIND_ENTITIES_HITS = """
MATCH (e:Entity)
WITH e, [t IN $terms WHERE toLower(e.name) CONTAINS t] AS matched
WHERE size(matched) > 0
OPTIONAL MATCH (e)-[r]-(:Entity)
WITH e, matched, count(r) AS degree
RETURN e.key AS key, e.name AS name, e.type AS type,
       size(matched) AS hits, degree
ORDER BY hits DESC, degree DESC
LIMIT $top_k
"""

# Query 1 (optional "specific" ranking): every entity whose name contains at least one search term (the candidates),
# with the list of terms it matched and how many relationships it has. The candidates are
# ranked afterwards in Python (see rank_entities) so that SPECIFIC matches win over
# entities that only match generic words such as "indoor" or "air".
Q_FIND_ENTITIES = """
MATCH (e:Entity)
WITH e, [t IN $terms WHERE toLower(e.name) CONTAINS t] AS matched
WHERE size(matched) > 0
OPTIONAL MATCH (e)-[r]-(:Entity)
WITH e, matched, count(r) AS degree
RETURN e.key AS key, e.name AS name, e.type AS type, matched, degree
LIMIT $max_candidates
"""

# Query 2: relationships touching the matched entities (either direction).
# Ordering: relationships whose BOTH ends matched come first, then those whose
# relationship type matches a verb in the question (e.g. "remove" -> REMOVES),
# then those backed by the most evidence.
Q_FIND_RELATIONSHIPS = """
MATCH (a:Entity)-[r]->(b:Entity)
WHERE a.key IN $keys OR b.key IN $keys
RETURN a.name AS source, type(r) AS relation, b.name AS target,
       r.evidence AS evidence, r.sources AS sources,
       (a.key IN $keys AND b.key IN $keys) AS both_matched,
       any(h IN $hints WHERE toLower(type(r)) CONTAINS h) AS hint_match
ORDER BY both_matched DESC, hint_match DESC, size(coalesce(r.evidence, [])) DESC
LIMIT $max_rels
"""

# Query 3 (fallback): chunks of the original documents that contain the terms.
Q_FALLBACK_CHUNKS = """
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WITH d, c, [t IN $terms WHERE toLower(c.text) CONTAINS t] AS matched
WHERE size(matched) > 0
RETURN d.name AS document, c.index AS chunk, c.text AS text, size(matched) AS hits
ORDER BY hits DESC
LIMIT $max_chunks
"""

# --------------------------------------------------------------------------
# Step 2a: question -> search terms
# --------------------------------------------------------------------------
STOPWORDS = set("""a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have
having he her here hers him his how i if in into is it its just me more most my no nor not of off on once
only or other our out over own same she should so some such than that the their theirs them then there
these they this those through to too under until up very was we were what when where which while who whom
why will with would you your""".split())

# Generic question words that say nothing about WHICH entity is meant
GENERIC_WORDS = {"effect", "impact", "important", "different", "type", "kind", "tell", "explain",
                 "describe", "give", "show", "list", "many", "much", "study", "paper", "research",
                 "find", "found", "result", "work", "help", "role", "use", "used", "using", "main"}

# Verb stems that correspond to relationship types in the graph. They are NOT used to
# find entities; they are used to rank relationships (Query 2).
RELATION_STEMS = ["remov", "affect", "reduc", "contain", "caus", "improv", "influenc", "locat",
                  "defin", "author", "publish", "measur", "test", "draw", "absorb", "increas",
                  "decreas", "associat", "part"]


def stem_plural(word):
    """Very small stemmer: plants -> plant, studies -> study (keeps 'glass', 'analysis')."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def extract_terms(question):
    """
    Split a question into (entity_terms, relation_hints).
      entity_terms:   content words, used to find entities by name (Query 1)
      relation_hints: verb stems matching relationship types, used for ranking (Query 2)
    """
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", question.lower())
    terms, hints = [], []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        rel = next((s for s in RELATION_STEMS if tok.startswith(s)), None)
        if tok in ("wrote", "written", "write", "writes"):
            rel = "author"                        # "who wrote ..." -> AUTHORED / AUTHORED_WITH relations
        if rel:                                   # a verb like "remove" / "affects"
            if rel not in hints:
                hints.append(rel)
            continue
        if tok in GENERIC_WORDS or (len(tok) < 3 and not any(c.isdigit() for c in tok)):
            continue
        tok = stem_plural(tok)
        if tok not in terms:
            terms.append(tok)
    return terms, hints


# --------------------------------------------------------------------------
# Logging helper
# --------------------------------------------------------------------------
def log(text=""):
    """Print a line and append it to rag_log.txt."""
    print(text)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def run_query(session, title, query, **params):
    """Run a Cypher query, log it (text + parameters + number of rows) and return the rows."""
    log(f"\n--- {title} ---")
    log(query.strip())
    log(f"parameters: {params}")
    rows = [record.data() for record in session.run(query, **params)]
    log(f"-> {len(rows)} row(s)")
    return rows


def rank_entities(candidates, top_k):
    """
    Rank candidate entities (rows from Query 1) and keep the best `top_k`.
      1. exact matches first: the entity name IS a search term (e.g. "CO2", "plants")
      2. then by specificity score: each matched term is worth 1/ln(1+n), where n is how
         many candidates contain that term, so rare terms (co2) count for more than
         common ones (indoor, air)
      3. then by number of connections
    """
    n = Counter(t for c in candidates for t in c["matched"])
    weight = {t: 1.0 / math.log(1 + n[t]) for t in n}
    for c in candidates:
        name = c["name"].lower()
        c["exact"] = any(name in (t, t + "s", t + "es") for t in c["matched"])
        c["score"] = round(sum(weight[t] for t in c["matched"]), 2)
    candidates.sort(key=lambda c: (c["exact"], c["score"], c["degree"]), reverse=True)
    return candidates[:top_k]


# --------------------------------------------------------------------------
# Step 2: retrieval from Neo4j
# --------------------------------------------------------------------------
def retrieve(session, question, top_k=TOP_K_ENTITIES, max_rels=MAX_RELATIONSHIPS, ranking="hits"):
    """Run the Cypher queries and return everything found, as plain Python data."""
    terms, hints = extract_terms(question)
    log(f"\nSEARCH TERMS (entities): {terms}")
    log(f"RELATION HINTS (ranking): {hints}")
    result = {"terms": terms, "entities": [], "relationships": [], "chunks": []}
    if not terms:
        log("No usable search terms in the question.")
        return result

    if ranking == "specific":
        candidates = run_query(session, "CYPHER QUERY 1: find entities matching the terms",
                               Q_FIND_ENTITIES, terms=terms, max_candidates=MAX_CANDIDATES)
        result["entities"] = rank_entities(candidates, top_k)
        log(f"Ranked {len(candidates)} candidate(s) in Python (exact match, specificity score, connections); keeping the top {top_k}:")
        for e in result["entities"]:
            kind = "exact match" if e["exact"] else "partial match"
            log(f"   {e['name']} ({e['type']}): {kind}, matched {e['matched']}, score {e['score']}, {e['degree']} connection(s)")
    else:                                               # default: ranking done inside Cypher
        result["entities"] = run_query(session, "CYPHER QUERY 1: find matching entities",
                                       Q_FIND_ENTITIES_HITS, terms=terms, top_k=top_k)
        for e in result["entities"]:
            log(f"   {e['name']} ({e['type']}): matched {e['hits']} term(s), {e['degree']} connection(s)")

    if result["entities"]:
        keys = [e["key"] for e in result["entities"]]
        result["relationships"] = run_query(session, "CYPHER QUERY 2: relationships around those entities",
                                            Q_FIND_RELATIONSHIPS, keys=keys, hints=hints, max_rels=max_rels)
        for r in result["relationships"]:
            log(f"   {r['source']} --{r['relation']}--> {r['target']}")

    # Fallback: nothing usable in the graph, so search the stored chunk text instead
    if not result["relationships"]:
        log("\nNo graph relationships found; falling back to a text search of the stored chunks.")
        result["chunks"] = run_query(session, "CYPHER QUERY 3 (fallback): chunks containing the terms",
                                     Q_FALLBACK_CHUNKS, terms=terms, max_chunks=MAX_CHUNKS)
    return result


# --------------------------------------------------------------------------
# Step 3: retrieved data -> text context
# --------------------------------------------------------------------------
EVIDENCE_PATTERN = re.compile(r"^\[(.+?) \| chunk (\d+)\]\s*(.*)$", re.S)


def clip(text, limit):
    """Collapse whitespace and cut to `limit` characters."""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def build_context(retrieved, max_chars=MAX_CONTEXT_CHARS):
    """
    Turn the retrieved graph data into text. Each relationship becomes one line:
        - Plants --removes--> CO2. Evidence: "<sentence>" [S1]
    Documents are numbered [S1], [S2], ... and listed once at the end, so the long
    file names do not eat the small model's context window.
    """
    sources = {}                      # document name -> "S1", "S2", ...
    lines, used = [], 0

    def source_id(doc):
        return sources.setdefault(doc, f"S{len(sources) + 1}")

    for r in retrieved["relationships"]:
        evidence_list = r.get("evidence") or []
        sentence, doc = "", None
        if evidence_list:
            m = EVIDENCE_PATTERN.match(evidence_list[0])      # "[document | chunk N] sentence"
            if m:
                doc, sentence = m.group(1), clip(m.group(3), MAX_EVIDENCE_CHARS)
            else:
                sentence = clip(evidence_list[0], MAX_EVIDENCE_CHARS)
        relation = r["relation"].replace("_", " ").lower()
        line = f"- {r['source']} --{relation}--> {r['target']}."
        if sentence:
            line += f' Evidence: "{sentence}"'
        if doc:
            line += f" [{source_id(doc)}]"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)

    parts = []
    if lines:
        parts.append("Facts from the knowledge graph:\n" + "\n".join(lines))
        if sources:
            parts.append("Sources:\n" + "\n".join(f"[{sid}] {doc}" for doc, sid in sources.items()))
    elif retrieved["chunks"]:                       # fallback: raw passages
        passages = [f"[{c['document']}, chunk {c['chunk']}] {clip(c['text'], 600)}" for c in retrieved["chunks"]]
        parts.append("Passages from the documents:\n" + "\n".join(passages))
    return "\n\n".join(parts)


SYSTEM_PROMPT = ("You are a helpful assistant. Answer the question using only the context provided. "
                 "If the context does not contain the answer, say that you do not know.")


def build_messages(question, context):
    """Add the retrieved context to the user's question (the augmented prompt)."""
    user_content = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}]


# --------------------------------------------------------------------------
# Step 4: generation with the local LLM
# --------------------------------------------------------------------------
def generate(messages, model):
    """Call the Ollama model. temperature 0 for repeatable answers; num_ctx matches TinyLlama's window."""
    response = ollama.chat(model=model, messages=messages,
                           options={"temperature": 0, "num_ctx": 2048, "num_predict": 300})
    return response["message"]["content"].strip()


def answer_question(session, question, model, compare=False, ranking="hits"):
    """Run the whole RAG pipeline for one question and log every step."""
    log("=" * 72)
    log(f"USER QUESTION: {question}")

    retrieved = retrieve(session, question, ranking=ranking)
    context = build_context(retrieved)

    log("\n--- RAG CONTEXT (retrieved data converted to text) ---")
    log(context or "(nothing was retrieved from the graph)")

    if context:
        messages = build_messages(question, context)
    else:                      # nothing retrieved: still give the model the same instructions
        messages = build_messages(question, "(no relevant information was found in the knowledge graph)")

    log("\n--- PROMPT SENT TO THE LLM ---")
    for m in messages:
        log(f"[{m['role']}]\n{m['content']}\n")

    response = generate(messages, model)
    log(f"--- RESPONSE ({model}, with graph context) ---")
    log(response)

    if compare:
        plain = generate([{"role": "user", "content": question}], model)
        log(f"\n--- RESPONSE ({model}, WITHOUT graph context, for comparison) ---")
        log(plain)
    return response


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Answer questions using the Neo4j knowledge graph as RAG context.")
    parser.add_argument("question", nargs="*", help="question (omit for interactive mode)")
    parser.add_argument("--model", default=LLM_MODEL, help="Ollama model name (default: tinyllama)")
    parser.add_argument("--compare", action="store_true", help="also answer without graph context")
    parser.add_argument("--ranking", choices=["hits", "specific"], default="hits",
                        help="entity ranking: 'hits' = most matched terms first (default), "
                             "'specific' = exact matches and rare terms first")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as err:
        sys.exit(f"Could not connect to Neo4j at {NEO4J_URI}: {err}\n"
                 "Check that the instance is RUNNING and that NEO4J_PASSWORD is set.")
    log(f"Connected to Neo4j at {NEO4J_URI}; model = {args.model}")

    try:
        with driver.session() as session:
            if args.question:
                answer_question(session, " ".join(args.question), args.model, args.compare, args.ranking)
            else:
                print("Interactive mode. Type a question, or 'quit' to exit.")
                while True:
                    q = input("\nQuestion: ").strip()
                    if q.lower() in ("", "quit", "exit"):
                        break
                    answer_question(session, q, args.model, args.compare, args.ranking)
    except ollama.ResponseError as err:
        sys.exit(f"Ollama error: {err}\nIs the model pulled? Try: ollama pull {args.model}")
    finally:
        driver.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # avoid Windows console encoding errors
    main()
