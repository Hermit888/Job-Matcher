"""
Keyword highlighting and extraction module.
Complements semantic matching with exact keyword detection.
"""

import argparse
import os
import random
import re
from typing import List, Set, Dict
from dataclasses import dataclass

from resume_match.text_extract import extract_text
from resume_match.compare_resume import split_resume_into_sentences


@dataclass
class KeywordMatch:
    """Represents a keyword match in a sentence."""
    keyword: str
    sentence: str
    sentence_id: int
    exact_match: bool  # True if exact match, False if case-insensitive


def extract_keywords_from_jd(jd_text: str) -> Set[str]:
    common_tech_keywords = {
        'python', 'java', 'c++', 'javascript', 'typescript', 'go', 'rust',
        'ruby', 'php', 'swift', 'kotlin', 'scala', 'r',
        'react', 'vue', 'angular', 'django', 'flask', 'spring', 'express',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
        'dynamodb', 'cassandra', 'oracle',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
        'terraform', 'ansible', 'ci/cd',
        'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum',
        'git', 'ml', 'ai', 'nlp', 'deep learning', 'machine learning',
        'linux', 'bash', 'vim', 'vscode', 'jupyter', 'postman',
    }

    found = set()
    jd_lower = jd_text.lower()
    for kw in common_tech_keywords:
        pat = build_keyword_pattern(kw)
        if pat.search(jd_lower):
            found.add(kw)
    return found


def load_keywords_from_file(filepath: str) -> Set[str]:
    keywords = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                kw = line.strip().lower()
                if kw:
                    keywords.add(kw)
    except FileNotFoundError:
        print(f"Warning: Keywords file not found: {filepath}")
    return keywords


# ---------------------------
# 改进 2 的核心：更稳健的 pattern
# ---------------------------
def build_keyword_pattern(keyword: str) -> re.Pattern:
    """
    Build a regex pattern that is robust to:
    - Symbol keywords: C++, CI/CD
    - Multi-word keywords: "machine learning"
    - Common variants for REST API: "RESTful API", "REST APIs"
    """
    kw = keyword.strip().lower()

    # Special-case: REST API variants (still within keyword matching scope)
    if kw == "rest api":
        # rest api / restful api / rest apis
        return re.compile(r"(?<![a-z0-9])rest(?:ful)?\s+api(?:s)?(?![a-z0-9])", re.IGNORECASE)

    # Normalize whitespace in multi-word keywords
    if " " in kw:
        parts = [re.escape(p) for p in kw.split()]
        # allow flexible spaces between words
        inner = r"\s+".join(parts)
        return re.compile(r"(?<![a-z0-9])" + inner + r"(?![a-z0-9])", re.IGNORECASE)

    # If keyword contains non-alphanumeric symbols, avoid \b which fails on C++/CI/CD
    if re.search(r"[^a-z0-9]", kw):
        inner = re.escape(kw)
        return re.compile(r"(?<![a-z0-9])" + inner + r"(?![a-z0-9])", re.IGNORECASE)

    # Plain word keyword
    return re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)


def find_keywords_in_sentences(keywords: Set[str], sentences: List[str]) -> List[KeywordMatch]:
    matches: List[KeywordMatch] = []

    # Precompile patterns for speed and consistency
    patterns: Dict[str, re.Pattern] = {kw: build_keyword_pattern(kw) for kw in keywords}

    for i, sentence in enumerate(sentences):
        s_lower = sentence.lower()
        for kw, pat in patterns.items():
            if pat.search(s_lower):
                matches.append(KeywordMatch(
                    keyword=kw,
                    sentence=sentence,
                    sentence_id=i,
                    exact_match=True
                ))
    return matches


def highlight_keywords_in_text(text: str, keywords: Set[str]) -> str:
    highlighted = text

    # Longest-first to avoid partial overlaps
    sorted_keywords = sorted(keywords, key=len, reverse=True)

    for kw in sorted_keywords:
        pat = build_keyword_pattern(kw)

        # Replacement keeps the matched surface form
        highlighted = pat.sub(lambda m: f"**{m.group(0)}**", highlighted)

    return highlighted


def compute_keyword_coverage(keywords: Set[str], matches: List[KeywordMatch]) -> Dict[str, object]:
    matched = {m.keyword for m in matches}
    missing = keywords - matched
    rate = len(matched) / len(keywords) if keywords else 0.0
    return {
        "total_keywords": len(keywords),
        "matched_keywords": len(matched),
        "missing_keywords": len(missing),
        "coverage_rate": rate,
        "matched_list": sorted(matched),
        "missing_list": sorted(missing),
    }


def print_keyword_report(keywords: Set[str], matches: List[KeywordMatch], sentences: List[str]) -> None:
    print("\n" + "=" * 80)
    print("KEYWORD MATCHING REPORT")
    print("=" * 80)

    stats = compute_keyword_coverage(keywords, matches)
    print(f"\nCoverage: {stats['matched_keywords']}/{stats['total_keywords']} "
          f"({stats['coverage_rate'] * 100:.1f}%)")

    if stats["matched_list"]:
        print(f"\n✓ MATCHED KEYWORDS ({len(stats['matched_list'])}):")
        for kw in stats["matched_list"]:
            kw_matches = [m for m in matches if m.keyword == kw]
            sentence_ids = {m.sentence_id for m in kw_matches}
            print(f"  • {kw.upper()}: found in sentence(s) {sorted(sentence_ids)}")

    if stats["missing_list"]:
        print(f"\n✗ MISSING KEYWORDS ({len(stats['missing_list'])}):")
        for kw in stats["missing_list"]:
            print(f"  • {kw.upper()}")

    print("\n" + "-" * 80)
    print("MATCHED SENTENCES (with highlights):")
    print("-" * 80)

    # Group matched keywords by sentence
    sent2kws: Dict[int, Set[str]] = {}
    for m in matches:
        sent2kws.setdefault(m.sentence_id, set()).add(m.keyword)

    for sent_id in sorted(sent2kws.keys()):
        kws = sent2kws[sent_id]
        highlighted = highlight_keywords_in_text(sentences[sent_id], kws)
        print(f"\n[s{sent_id}] {highlighted}")


def main():
    parser = argparse.ArgumentParser(description="Keyword coverage + highlighting for a resume.")
    parser.add_argument("--resume", required=True, help="Path to resume file (.txt/.docx/.pdf)")
    parser.add_argument("--jd", default=None, help="Optional: JD text file to extract keywords from")
    parser.add_argument("--skills", default=None, help="Optional: skills keyword file (one per line)")
    parser.add_argument("--sample_k", type=int, default=0, help="Randomly sample K keywords (0 = no sampling)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    resume_text, method = extract_text(args.resume)
    print(f"Resume extracted via: {method}")

    sentences = split_resume_into_sentences(resume_text)

    keywords: Set[str] = set()

    if args.jd:
        with open(args.jd, "r", encoding="utf-8", errors="ignore") as f:
            jd_text = f.read()
        keywords |= extract_keywords_from_jd(jd_text)

    if args.skills:
        keywords |= load_keywords_from_file(args.skills)

    if not keywords:
        raise RuntimeError("No keywords loaded. Provide --jd and/or --skills.")

    # Optional sampling (your item c)
    if args.sample_k and args.sample_k > 0:
        random.seed(args.seed)
        k = min(args.sample_k, len(keywords))
        keywords = set(random.sample(sorted(keywords), k))
        print(f"Randomly sampled {k} keywords.")

    print(f"Searching for {len(keywords)} keywords...")

    matches = find_keywords_in_sentences(keywords, sentences)
    print_keyword_report(keywords, matches, sentences)


if __name__ == "__main__":
    main()
