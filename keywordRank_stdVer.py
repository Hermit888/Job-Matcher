import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

################
# Golobal keyword weights
################
ALPHA = 0.4   # KeyBERT cosine similarity
BETA  = 0.2   # JD 区域权重
GAMMA = 0.2   # 关键词出现频率
DELTA = 0.2   # keyword ↔ job title cosine similarity

################
# Secontion weights for matching with job description. This change when target key word appears in different places of the JD.
################
SECTION_WEIGHTS = {
    "title": 1.0,
    "required": 0.8,
    "responsibilities": 0.6,
    "preferred": 0.4
}

################################
# keyword frequency
################################
def term_frequency(term, full_text):
    tokens = re.findall(r"\b\w+\b", full_text.lower())
    return tokens.count(term.lower())

################################
# Computing total freq
################################
def compute_frequency_stats(extracted_keywords, full_raw_text):
    freqs = []

    for kws in extracted_keywords.values():
        for keyword, _ in kws:
            freqs.append(term_frequency(keyword, full_raw_text))

    mean = np.mean(freqs)
    std = np.std(freqs)

    return mean, std
# Z-score for standard deviation
def frequency_z_score(freq, mean, std, eps=1e-6):
    z = (freq - mean) / (std + eps)

    normalized = 1 / (1 + np.exp(-z))  # sigmoid

    return normalized

################################
# keywords ranking
################################
def rank_keywords(
    extracted_keywords,    # dict: {section: [(keyword, keybert_sim), ...]}
    job_title,             # str
    full_jd_text,          # str
    full_raw_text,         # str
    embedding_model        # SentenceTransformer
):
    title_emb = embedding_model.encode(job_title)
    ranked = []

    freq_mean, freq_std = compute_frequency_stats(
        extracted_keywords, full_raw_text
    )

    for section, kws in extracted_keywords.items():
        section_weight = SECTION_WEIGHTS.get(section, 0.5)

        for keyword, keybert_sim in kws:
            # keyword frequency with standard deviation.
            freq = term_frequency(keyword, full_raw_text)
            freq_score = frequency_z_score(
                freq, freq_mean, freq_std
            )

            # keyword similarity with
            kw_emb = embedding_model.encode(keyword)
            title_sim = cosine_similarity(
                [kw_emb], [title_emb]
            )[0][0]

            # Summary score
            score = (
                ALPHA * keybert_sim +
                BETA  * section_weight +
                GAMMA * freq_score +
                DELTA * title_sim
            )

            ranked.append({
                "keyword": keyword,
                "section": section,
                "keybert_sim": round(keybert_sim, 4),
                "section_weight": section_weight,
                "freq": freq,
                "title_sim": round(title_sim, 4),
                "final_score": round(score, 4)
            })

    # Ranking
    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    # Return format in:
    # List[Dict[str, Any]]
    # {
    #     "keyword": str,          # keyword
    #     "section": str,          # in which jd section
    #     "freq": int,             # 在 JD 全文中的出现次数
    #     "freq_score": float,     # 标准差归一化后的频率得分 (0, 1)
    #     "final_score": float     # 综合排序分数（已加权）
    # }
    return ranked
