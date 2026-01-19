import argparse
import os
import re
from dataclasses import dataclass
from typing import List
from resume_match.text_extract import extract_text

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# -----------------
# Utilities
# -----------------
def read_non_empty_lines(path: str) -> List[str]:
    """Read a text file line-by-line and return non-empty stripped lines."""
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def split_resume_into_sentences(text: str) -> List[str]:
    """
    First split by newlines into rough lines
    Then further split by sentence-ending punctuation .!?
    Keep order, remove empties
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("•", "\n").replace("·", "\n").replace("◦", "\n")
    text = text.replace("", "\n")

    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sentences = []

    end_splitter = re.compile(r"(?<=[.!?])\s+")

    for ln in raw_lines:
        ln = normalize_whitespace(ln)
        parts = end_splitter.split(ln)
        for p in parts:
            p = normalize_whitespace(p)
            if len(p) >= 4:
                sentences.append(p)

    # De-duplicate consecutive identical sentences
    deduped = []
    prev = None
    for s in sentences:
        if s != prev:
            deduped.append(s)
        prev = s
    return deduped


def has_verb(text: str) -> bool:
    """
    Simple heuristic verb detection.
    Check for common action verbs in resumes.
    """
    text_lower = text.lower()

    common_action_verbs = [
        'developed', 'implemented', 'designed', 'built', 'created',
        'managed', 'led', 'coordinated', 'organized', 'improved',
        'engineered', 'optimized', 'deployed', 'maintained',
        'provided', 'conducted', 'analyzed', 'reduced', 'increased',
        'customized', 'enabling', 'ensuring', 'supporting',
        'boosting', 'enhancing', 'promoting', 'developing',
        'building', 'maintaining', 'designing', 'creating'
    ]

    if any(verb in text_lower for verb in common_action_verbs):
        return True

    return False


def is_likely_section_header(text: str) -> bool:
    """
    Generic section header detection without hardcoded keywords.

    Criteria:
    1. Short text (< 60 characters)
    2. Special format (ALL CAPS or Title Case)
    3. No verbs (noun phrase)
    4. No sentence-ending punctuation
    """
    text = text.strip()

    if not text or len(text) > 60:
        return False

    if text and text[-1] in '.!?;,':
        return False

    if ',' in text:
        return False

    # Check for ALL CAPS format
    letters = ''.join(c for c in text if c.isalpha())
    if letters and letters.isupper() and len(text) <= 50:
        return True

    # Check for Title Case
    words = text.replace(':', '').split()
    if len(words) <= 5 and len(words) >= 1:
        capitalized_words = sum(1 for w in words if w and w[0].isupper())
        if capitalized_words >= len(words) * 0.8:
            if not has_verb(text):
                return True

    # Check special format (letters, spaces, colons, hyphens only)
    if re.match(r'^[A-Za-z\s:\-&/]+$', text) and len(text) <= 40:
        if not has_verb(text):
            return True

    return False


def clean_header_prefix(text: str) -> str:
    """
    Remove header prefix from the beginning of a sentence.
    Example: "SKILLS Core CS: Data Structures..." -> "Core CS: Data Structures..."

    Only removes if the first word looks like a standalone header.
    """
    text = text.strip()

    # Split by whitespace to get first word
    parts = text.split(None, 1)
    if len(parts) < 2:
        return text  # Only one word, keep as is

    first_word = parts[0].rstrip(':')
    rest = parts[1]

    # Check if first word looks like a header
    if (first_word.isupper() or first_word.istitle()) and len(first_word) <= 20:
        if is_likely_section_header(first_word):
            return rest

    return text


def is_header_or_noise_line(s: str) -> bool:
    """
    Determine if a line is a header or noise.
    Uses generic algorithm without hardcoded keywords.
    """
    s = normalize_whitespace(s)
    if not s:
        return True

    # Use generic header detection
    if is_likely_section_header(s):
        return True

    # Too short => likely fragment
    if len(s) < 8:
        return True

    return False


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between all rows in a and all rows in b.
    """
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
        model,
        resume_embeddings: np.ndarray,
        top_k: int = 2,
) -> List[Match]:
    """
    For a single JD requirement, find top_k most similar resume sentences.
    """
    req_emb = model.encode([requirement], convert_to_numpy=True, normalize_embeddings=True)
    sims = (resume_embeddings @ req_emb.T).reshape(-1)

    if len(sims) == 0:
        return []

    top_k = min(top_k, len(sims))
    top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    matches = []
    for idx in top_idx:
        matches.append(Match(
            score=float(sims[idx]),
            sentence=resume_sentences[idx],
            sentence_id=int(idx)
        ))
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


def main():
    parser = argparse.ArgumentParser(
        description="JD requirements -> CV evidence sentences"
    )
    parser.add_argument(
        "--resume",
        default=os.path.join("data", "sample_resume.txt"),
        help="Path to resume file"
    )
    parser.add_argument(
        "--requirements",
        default=os.path.join("data", "jd_requirements.txt"),
        help="Path to JD requirements file",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Top-K CV sentences per requirement"
    )
    parser.add_argument(
        "--min_sent_len",
        type=int,
        default=4,
        help="Minimum sentence length",
    )
    args = parser.parse_args()

    if SentenceTransformer is None:
        raise RuntimeError(
            "Missing dependency: sentence-transformers\n"
            "Install with: pip install sentence-transformers"
        )

    # Load resume and split into sentences
    resume_text, extraction_method = extract_text(args.resume)
    print(f"Resume extracted via: {extraction_method}")
    resume_sentences = split_resume_into_sentences(resume_text)

    # Clean header prefixes from sentences
    resume_sentences = [clean_header_prefix(s) for s in resume_sentences]

    # Filter headers and noise
    resume_sentences = [s for s in resume_sentences if not is_header_or_noise_line(s)]

    # Apply minimum length filter
    resume_sentences = [s for s in resume_sentences if len(s) >= args.min_sent_len]

    if not resume_sentences:
        raise RuntimeError(f"No usable sentences extracted from resume: {args.resume}")

    # Load JD requirements
    requirements = read_non_empty_lines(args.requirements)
    if not requirements:
        raise RuntimeError(f"No requirements found in: {args.requirements}")

    # Load model
    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # Precompute resume embeddings
    print(f"Encoding {len(resume_sentences)} CV sentences...")
    resume_embeddings = model.encode(
        resume_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Match each requirement
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