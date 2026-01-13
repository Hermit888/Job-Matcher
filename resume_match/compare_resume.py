import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

#-----------------
#Utilities
#-----------------
def read_text_file(path:str) -> str:
    """Read an entire text file as a single string (UTF-8)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()

def read_non_empty_lines(path:str) -> List[str]:
    """Read a text file line-by-line and return non-empty stripped lines."""
    lines: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            s = line.strip()
            if s:
                lines.append(s)
    return lines

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

#input CV text, output sentences list.
def split_resume_into_sentences(text: str) -> List[str]:
    """
    First split by newlines into rough lines
    Then further split by sentence-ending punctuation .!?
    Keep order, remove empties
    """
    text = text.replace("\r\n", "\n").replace("\r", "n")
    text = text.replace("•", "\n").replace("·", "\n").replace("●", "\n")

    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sentences: List[str] = []

    # Split on punctuation that usually ends a sentence.
    # This is not perfect, but stable and good enough for Phase 1.
    end_splitter = re.compile(r"(?<=[.!?])\s+")

    for ln in raw_lines:
        ln = normalize_whitespace(ln)
        parts = end_splitter.split(ln)
        for p in parts:
            p = normalize_whitespace(p)
            # filter very short fragments (often headers like "Skills" or single characters)
            if len(p) >= 4:
                sentences.append(p)

    # De-duplicate consecutive identical sentences (rare but can happen)
    deduped: List[str] = []
    prev = None
    for s in sentences:
        if s != prev:
            deduped.append(s)
        prev = s
    return deduped

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between all rows in a and all rows in b.
    a: (m, d), b:(n, d) -> sim: (m, n)
    """
    #Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

@dataclass
class Match:
    score: float
    sentence: str
    sentence_id: int

def find_top_matches_for_requirement(
        requirement: str,
        resume_sentences: List[str],
        model: "SentenceTransformer",
        resume_embeddings: np.ndarray,
        top_k: int = 2,
    ) -> List[Match]:
    """
    For a single JD requirement sentence, find top_k most similar resume sentences.
    We compute requirement embedding once and compare to precomputed resume embeddings.
    """
    req_emb = model.encode([requirement], convert_to_numpy=True, normalize_embeddings=True)
    sims = (resume_embeddings @ req_emb.T).reshape(-1)

    if len(sims) == 0:
        return []

    top_k = min(top_k, len(sims))
    top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    matches: List[Match] = []
    for idx in top_idx:
        matches.append(Match(score=float(sims[idx]), sentence=resume_sentences[idx], sentence_id=int(idx)))
    return matches


def print_requirement_result(requirement: str, matches: List[Match]) -> None:
    print("\n" + "=" * 80)
    print("JD Requirement:")
    print(f"{requirement}")
    print("\nTop matches in CV:")
    if not matches:
        print("(no sentences found)")
        return
    for m in matches:
        print(f"[{m.score:.2f}] (s{m.sentence_id}) {m.sentence}")

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: JD requirements -> CV evidence sentences (MiniLM embeddings)")
    parser.add_argument("--resume", default=os.path.join("data", "sample_resume.txt"), help="Path to resume txt file")
    parser.add_argument(
        "--requirements",
        default=os.path.join("data", "jd_requirements.txt"),
        help="Path to JD requirements txt file (one requirement per line)",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--top_k", type=int, default=2, help="Top-K CV sentences to show per requirement")
    parser.add_argument(
        "--min_sent_len",
        type=int,
        default=4,
        help="Minimum sentence length (characters) to keep from resume splitting",
    )
    args = parser.parse_args()

    if SentenceTransformer is None:
        raise RuntimeError(
            "Missing dependency: sentence-transformers\n"
            "Install with: pip install sentence-transformers"
        )

    # 1) Load resume and split into sentences
    resume_text = read_text_file(args.resume)
    resume_sentences = split_resume_into_sentences(resume_text)
    # apply min length filter
    resume_sentences = [s for s in resume_sentences if len(s) >= args.min_sent_len]

    if not resume_sentences:
        raise RuntimeError(
            f"No usable sentences extracted from resume: {args.resume}\n"
            "If you used a PDF/DOC, please convert to TXT first for Phase 1."
        )

    # 2) Load JD requirements lines
    requirements = read_non_empty_lines(args.requirements)
    if not requirements:
        raise RuntimeError(f"No requirements found in: {args.requirements}")

    # 3) Load model
    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # 4) Precompute resume embeddings (normalize to allow dot-product as cosine similarity)
    print(f"Encoding {len(resume_sentences)} CV sentences...")
    resume_embeddings = model.encode(
        resume_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    # resume_embeddings shape: (num_sentences, dim)

    # 5) For each requirement, find top matches
    print(f"\nMatching {len(requirements)} JD requirements to CV sentences (top_k={args.top_k})...")
    for req in requirements:
        req = normalize_whitespace(req)
        matches = find_top_matches_for_requirement(
            requirement=req,
            resume_sentences=resume_sentences,
            model=model,
            resume_embeddings=resume_embeddings,
            top_k=args.top_k,
        )
        print_requirement_result(req, matches)

    print("\nDone.")


if __name__ == "__main__":
    main()