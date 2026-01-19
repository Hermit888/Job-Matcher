"""
Debug script to inspect sentence extraction process
Run this to see what's happening to the missing sentences
"""

import sys

sys.path.insert(0, '.')

from resume_match.text_extract import extract_text
from resume_match.compare_resume import (
    split_resume_into_sentences,
    clean_header_prefix,
    is_header_or_noise_line,
    normalize_whitespace
)


def debug_extraction(resume_path):
    print(f"\n{'=' * 80}")
    print(f"Debugging: {resume_path}")
    print(f"{'=' * 80}\n")

    # Step 1: Extract raw text
    resume_text, method = extract_text(resume_path)
    print(f"[1] Extracted via: {method}")
    print(f"[1] Total characters: {len(resume_text)}")
    print(f"\n[1] First 500 chars of raw text:")
    print(resume_text[:500])
    print("\n" + "-" * 80 + "\n")

    # Step 2: Split into sentences
    sentences = split_resume_into_sentences(resume_text)
    print(f"[2] After splitting: {len(sentences)} sentences")
    print("\n[2] All sentences:")
    for i, s in enumerate(sentences):
        print(f"  s{i}: {s[:80]}{'...' if len(s) > 80 else ''}")
    print("\n" + "-" * 80 + "\n")

    # Step 3: Clean header prefixes
    cleaned = [clean_header_prefix(s) for s in sentences]
    print(f"[3] After cleaning prefixes:")
    changes = 0
    for i, (original, cleaned_s) in enumerate(zip(sentences, cleaned)):
        if original != cleaned_s:
            changes += 1
            print(f"  s{i} CHANGED:")
            print(f"    Before: {original[:80]}")
            print(f"    After:  {cleaned_s[:80]}")
    print(f"  Total changes: {changes}")
    print("\n" + "-" * 80 + "\n")

    # Step 4: Filter headers and noise
    filtered = []
    removed = []
    for i, s in enumerate(cleaned):
        if is_header_or_noise_line(s):
            removed.append((i, s))
        else:
            filtered.append(s)

    print(f"[4] After filtering: {len(filtered)} sentences (removed {len(removed)})")
    if removed:
        print("\n[4] REMOVED sentences:")
        for i, s in removed:
            print(f"  s{i}: {s[:80]}{'...' if len(s) > 80 else ''}")
    print("\n" + "-" * 80 + "\n")

    # Step 5: Length filter
    min_len = 4
    final = [s for s in filtered if len(s) >= min_len]
    print(f"[5] After length filter (>={min_len}): {len(final)} sentences")

    # Look for programming-related sentences
    print("\n" + "=" * 80)
    print("SEARCHING FOR PROGRAMMING-RELATED SENTENCES:")
    print("=" * 80)

    keywords = ["python", "java", "c++", "programming", "javascript"]

    print("\n[Raw text search]")
    for kw in keywords:
        if kw.lower() in resume_text.lower():
            # Find the line
            for line in resume_text.split('\n'):
                if kw.lower() in line.lower():
                    print(f"  Found '{kw}' in: {line.strip()[:80]}")
                    break

    print("\n[After splitting]")
    for kw in keywords:
        for i, s in enumerate(sentences):
            if kw.lower() in s.lower():
                print(f"  s{i} contains '{kw}': {s[:80]}")

    print("\n[After cleaning]")
    for kw in keywords:
        for i, s in enumerate(cleaned):
            if kw.lower() in s.lower():
                print(f"  s{i} contains '{kw}': {s[:80]}")

    print("\n[After filtering]")
    for kw in keywords:
        for s in filtered:
            if kw.lower() in s.lower():
                print(f"  Found '{kw}': {s[:80]}")

    print("\n[FINAL]")
    for kw in keywords:
        for s in final:
            if kw.lower() in s.lower():
                print(f"  ✓ '{kw}': {s[:80]}")


if __name__ == "__main__":
    import os

    files = [
        os.path.join("data", "sample_resume.txt"),
        os.path.join("data", "sample_resume.pdf"),
        os.path.join("data", "sample_resume.docx"),
    ]

    for f in files:
        if os.path.exists(f):
            debug_extraction(f)
            print("\n\n")