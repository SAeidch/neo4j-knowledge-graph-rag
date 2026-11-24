"""
kg_builder.py - Knowledge Graph formation pipeline

Pipeline (matches the 7 assignment steps):
  1. Read documents from a local corpus directory
  2. Extract text from TXT, PDF and DOCX files
  3. Split documents into overlapping chunks
  4. Use a local LLM (Ollama, llama3.1:8b) to extract entities + relationships
  5. Validate and normalize the extracted information
  6. Insert entities and relationships into Neo4j
  7. Keep source-document and evidence information (Document/Chunk nodes,
     MENTIONS edges, and an `evidence` list on every relationship)

Graph model written to Neo4j:
  (:Document {id, name, path})
      -[:HAS_CHUNK]->  (:Chunk {id, index, text})
      -[:MENTIONS]->   (:Entity {key, name, type})
  (:Entity)-[:SOME_RELATION {evidence: [...], sources: [...]}]->(:Entity)

Usage examples:
  python kg_builder.py                                  # defaults
  python kg_builder.py --clear                          # wipe graph first
  python kg_builder.py --clear --chunk-size 500 --overlap 50
  python kg_builder.py --clear --max-docs 3 --max-chars 15000   # quick, small run
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pymupdf  # PyMuPDF (pip install pymupdf)
import ollama  # Python client for the local Ollama server
from docx import Document as DocxDocument  # python-docx
from neo4j import GraphDatabase

# --------------------------------------------------------------------------
# CONFIGURATION - edit these (or set the environment variables)
# --------------------------------------------------------------------------
# Folder with your TXT/PDF/DOCX files (default: ./corpus next to this script; override with
# --corpus or the KG_CORPUS_DIR environment variable)
CORPUS_DIR = os.getenv("KG_CORPUS_DIR", str(Path(__file__).with_name("corpus")))
OLLAMA_MODEL = "llama3.1:8b"                # any model you have pulled with ollama
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# Never hard-code the password: set it in the shell, e.g. PowerShell  $env:NEO4J_PASSWORD="..."
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "YOUR_PASSWORD_HERE")

CHUNK_SIZE = 1000   # target characters per chunk
CHUNK_OVERLAP = 150  # characters shared between consecutive chunks

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

# Safety limits for the LLM call so one bad chunk cannot hang the whole run
OLLAMA_TIMEOUT = 180      # seconds to wait for one chunk before giving up
MAX_OUTPUT_TOKENS = 1200  # cap on generated tokens (stops endless repetition loops)

# Entity types we accept; anything else is mapped to "Other"
ALLOWED_ENTITY_TYPES = {
    "Person", "Organization", "Location", "Event", "Concept",
    "Technology", "Product", "Date", "Other",
}


# --------------------------------------------------------------------------
# STEP 1 + 2: Read the corpus and extract text
# --------------------------------------------------------------------------
def read_corpus(corpus_dir):
    """Return a sorted list of supported files found (recursively) in corpus_dir."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root.resolve()}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def extract_text(path):
    """Extract plain text from a .txt, .pdf or .docx file."""
    ext = path.suffix.lower()

    if ext == ".txt":
        # errors="replace" so odd encodings don't crash the run
        return path.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        with pymupdf.open(path) as pdf:
            return "\n\n".join(page.get_text() for page in pdf)

    if ext == ".docx":
        doc = DocxDocument(path)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        # Tables are not part of doc.paragraphs, so read them separately
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts)

    raise ValueError(f"Unsupported file type: {ext}")


def strip_references(text):
    """
    Cut off the reference list of a paper (if found in the second half of the
    text). References produce many junk author/title entities and slow the run.
    """
    matches = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", text, re.IGNORECASE))
    if matches and matches[-1].start() > len(text) * 0.5:
        return text[:matches[-1].start()]
    return text


# --------------------------------------------------------------------------
# STEP 3: Chunking
# --------------------------------------------------------------------------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into chunks of ~`size` characters with `overlap` characters
    shared between neighbours. Each chunk tries to end at a paragraph or
    sentence boundary (searching the last 40% of the window) so facts are
    less likely to be cut in half.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text).strip()  # collapse blank lines

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window_start = start + int(size * 0.6)
            for sep in ("\n\n", ". ", "\n", " "):  # best boundary first
                idx = text.rfind(sep, window_start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always move forward
    return chunks


# --------------------------------------------------------------------------
# STEP 4: LLM extraction of entities and relationships
# --------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You extract knowledge graphs from text.
Return ONLY a JSON object with exactly this shape:
{{
  "entities": [{{"name": "...", "type": "..."}}],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "evidence": "..."}}
  ]
}}
Rules:
- Entity "type" must be one of: {", ".join(sorted(ALLOWED_ENTITY_TYPES))}.
- Use the full, canonical name of an entity (e.g. "Marie Curie", not "she").
- Relationship "type" is a short verb phrase in UPPER_SNAKE_CASE
  (e.g. WORKS_FOR, FOUNDED, LOCATED_IN, PART_OF).
- "source" and "target" must be names that also appear in "entities".
- "evidence" must be the exact sentence (copied verbatim) from the text that
  supports the relationship.
- Only extract facts stated in the text. Do not invent anything.
- If there is nothing to extract, return {{"entities": [], "relationships": []}}."""


def extract_graph_from_chunk(chunk, model=OLLAMA_MODEL, retries=2):
    """
    Ask the local LLM for entities/relationships. Returns a dict (maybe empty).
    Never hangs: each call has a timeout and an output-token cap, and any
    failure (timeout, bad JSON, Ollama error) is retried and then skipped.
    """
    client = ollama.Client(timeout=OLLAMA_TIMEOUT)
    for attempt in range(1, retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"TEXT:\n{chunk}"},
                ],
                format="json",                 # forces valid JSON output
                options={
                    "temperature": 0,          # deterministic, less hallucination
                    "num_predict": MAX_OUTPUT_TOKENS,  # hard cap on output length
                },
            )
            data = json.loads(response["message"]["content"])
            if isinstance(data, dict):
                return data
        except Exception as err:  # timeout, connection error, bad JSON, ...
            print(f"    ! LLM call failed (attempt {attempt}/{retries}): {type(err).__name__}: {err}")
    return {"entities": [], "relationships": []}


# --------------------------------------------------------------------------
# STEP 5: Validation and normalization
# --------------------------------------------------------------------------
def normalize_name(name):
    """Clean an entity name: trim, collapse spaces, strip quotes and a leading 'the'."""
    name = re.sub(r"\s+", " ", str(name)).strip(" \t\n\"'`.,;:")
    name = re.sub(r"^(the)\s+", "", name, flags=re.IGNORECASE)
    return name


def entity_key(name):
    """Case-insensitive identity so 'NASA' and 'nasa' become one node."""
    return name.lower()


def normalize_rel_type(rel_type):
    """UPPER_SNAKE_CASE, letters/digits/underscores only (safe to put in Cypher)."""
    rel = re.sub(r"[^A-Za-z0-9]+", "_", str(rel_type)).strip("_").upper()[:40]
    if not rel:
        return "RELATED_TO"
    if rel[0].isdigit():
        rel = "REL_" + rel
    return rel


def _squash(s):
    """Lowercase + collapse whitespace, used for evidence matching."""
    return re.sub(r"\s+", " ", s).strip().lower()


def find_evidence_sentence(chunk, src, dst):
    """Fallback: find a sentence in the chunk that mentions both entities."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk):
        s = sentence.lower()
        if src.lower() in s and dst.lower() in s:
            return sentence.strip()
    return None


def validate_and_normalize(raw, chunk):
    """
    Turn raw LLM output into clean entities/relationships.
      - drop malformed items, empty names and self-loops
      - drop entities whose name does not literally appear in the chunk
        (guards against hallucinated entities)
      - map unknown entity types to "Other"
      - normalize relationship types
      - require relationship endpoints to be valid entities
      - require evidence that is actually in the chunk (else try to recover a
        sentence mentioning both entities, else drop the relationship)
      - de-duplicate
    """
    chunk_lower = chunk.lower()
    squashed_chunk = _squash(chunk)

    # ---- entities ----
    entities = {}  # key -> {"key", "name", "type"}
    for e in raw.get("entities", []) or []:
        if not isinstance(e, dict) or not e.get("name"):
            continue
        name = normalize_name(e["name"])
        if len(name) < 2 or name.lower() not in chunk_lower:
            continue
        etype = str(e.get("type", "Other")).strip().title()
        if etype not in ALLOWED_ENTITY_TYPES:
            etype = "Other"
        entities.setdefault(entity_key(name), {"key": entity_key(name), "name": name, "type": etype})

    # ---- relationships ----
    relationships = {}  # (src, type, dst) -> dict
    for r in raw.get("relationships", []) or []:
        if not isinstance(r, dict):
            continue
        if not (r.get("source") and r.get("target")):
            continue
        src, dst = normalize_name(r["source"]), normalize_name(r["target"])
        src_key, dst_key = entity_key(src), entity_key(dst)
        if src_key == dst_key:                                 # no self-loops
            continue
        if src_key not in entities or dst_key not in entities:  # endpoints must be valid
            continue

        evidence = str(r.get("evidence", "")).strip()
        if not evidence or _squash(evidence) not in squashed_chunk:
            evidence = find_evidence_sentence(chunk, src, dst)
        if not evidence:
            continue  # cannot ground this relationship in the text -> discard

        rel_type = normalize_rel_type(r.get("type", ""))
        relationships.setdefault(
            (src_key, rel_type, dst_key),
            {"source": src_key, "target": dst_key, "type": rel_type, "evidence": evidence},
        )

    return list(entities.values()), list(relationships.values())


# --------------------------------------------------------------------------
# STEP 6 + 7: Neo4j insertion with source/evidence tracking
# --------------------------------------------------------------------------
def setup_schema(session):
    """Uniqueness constraints (also create indexes, and make MERGE safe)."""
    session.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE")
    session.run("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
    session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")


def write_document(tx, doc_id, name, path):
    tx.run(
        "MERGE (d:Document {id: $id}) SET d.name = $name, d.path = $path",
        id=doc_id, name=name, path=path,
    )


def write_chunk_graph(tx, doc_id, doc_name, chunk_id, index, text, entities, relationships):
    """Write one chunk, its entities and its relationships in a single transaction."""
    # Chunk node, linked to its source document
    tx.run(
        """
        MATCH (d:Document {id: $doc_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.index = $index, c.text = $text
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        doc_id=doc_id, chunk_id=chunk_id, index=index, text=text,
    )

    # Entity nodes (MERGE on key = same entity from different chunks/docs is one node)
    for e in entities:
        tx.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (e:Entity {key: $key})
            ON CREATE SET e.name = $name, e.type = $type
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id, key=e["key"], name=e["name"], type=e["type"],
        )

    # Relationships. Neo4j cannot parameterize relationship TYPES, so the type is
    # inserted into the query string. This is safe because normalize_rel_type()
    # guarantees it contains only A-Z, 0-9 and underscores.
    for r in relationships:
        tx.run(
            f"""
            MATCH (a:Entity {{key: $src}}), (b:Entity {{key: $dst}})
            MERGE (a)-[rel:{r["type"]}]->(b)
            SET rel.evidence = coalesce(rel.evidence, []) +
                    CASE WHEN $evidence IN coalesce(rel.evidence, []) THEN [] ELSE [$evidence] END,
                rel.sources = coalesce(rel.sources, []) +
                    CASE WHEN $doc_name IN coalesce(rel.sources, []) THEN [] ELSE [$doc_name] END
            """,
            src=r["source"], dst=r["target"],
            evidence=f"[{doc_name} | chunk {index}] {r['evidence']}",
            doc_name=doc_name,
        )


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build a knowledge graph in Neo4j from a document corpus.")
    parser.add_argument("--corpus", default=CORPUS_DIR)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--clear", action="store_true", help="delete the whole graph before loading")
    parser.add_argument("--resume", action="store_true", help="skip chunks already stored in Neo4j (use without --clear, same chunk settings)")
    parser.add_argument("--max-docs", type=int, default=None, help="only process the first N documents")
    parser.add_argument("--max-chars", type=int, default=None, help="only use the first N characters of each document")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()  # fails fast if Neo4j is off or the password is wrong
    print(f"Connected to Neo4j at {NEO4J_URI}")

    totals = {"files": 0, "chunks": 0, "entities": 0, "relationships": 0}
    start_time = time.time()

    with driver.session() as session:
        if args.clear:
            session.run("MATCH (n) DETACH DELETE n")
            print("Cleared existing graph.")
        setup_schema(session)

        files = read_corpus(args.corpus)
        print(f"Found {len(files)} document(s) in '{args.corpus}'")
        if args.max_docs:
            files = files[:args.max_docs]
            print(f"Processing only the first {len(files)} document(s) (--max-docs)")

        for path in files:
            doc_name = path.name
            doc_id = path.relative_to(args.corpus).as_posix()
            print(f"\n== {doc_name}")

            text = extract_text(path)
            if not text.strip():
                print("   (no extractable text, skipping)")
                continue
            text = strip_references(text)       # drop the reference list
            if args.max_chars:
                text = text[:args.max_chars]    # only keep the first N characters

            chunks = chunk_text(text, args.chunk_size, args.overlap)
            print(f"   {len(text)} characters -> {len(chunks)} chunk(s)")
            session.execute_write(write_document, doc_id, doc_name, str(path))
            totals["files"] += 1

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}::{i}"
                # Resume support: skip chunks that were already written
                if args.resume and session.run(
                    "MATCH (c:Chunk {id: $id}) RETURN count(c) AS n", id=chunk_id
                ).single()["n"]:
                    print(f"   chunk {i + 1}/{len(chunks)}: already done, skipping")
                    continue

                t0 = time.time()
                try:
                    raw = extract_graph_from_chunk(chunk, args.model)
                    entities, relationships = validate_and_normalize(raw, chunk)
                    session.execute_write(
                        write_chunk_graph, doc_id, doc_name, chunk_id, i, chunk,
                        entities, relationships,
                    )
                except Exception as err:  # one bad chunk should not kill the run
                    print(f"   chunk {i + 1}/{len(chunks)}: FAILED ({type(err).__name__}: {err}), skipping")
                    continue
                totals["chunks"] += 1
                totals["entities"] += len(entities)
                totals["relationships"] += len(relationships)
                print(f"   chunk {i + 1}/{len(chunks)}: {len(entities)} entities, "
                      f"{len(relationships)} relationships ({time.time() - t0:.0f}s)")

        # Final counts straight from the database (after de-duplication by MERGE)
        n_ent = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        n_rel = session.run("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS n").single()["n"]

    driver.close()
    print("\n===== SUMMARY =====")
    print(f"chunk size={args.chunk_size}, overlap={args.overlap}, model={args.model}")
    print(f"time taken: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"documents: {totals['files']}   chunks: {totals['chunks']}")
    print(f"extracted (before dedupe): {totals['entities']} entities, {totals['relationships']} relationships")
    print(f"in Neo4j (after dedupe):   {n_ent} entities, {n_rel} entity-entity relationships")


if __name__ == "__main__":
    main()
