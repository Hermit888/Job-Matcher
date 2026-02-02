from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

"""
summary based on the test cases: 
1. all-mpnet-base-v2 looks perform better. It may be because it is more vector-dense.
2. contents only. don't include section headers like "What We Look For In You", "Nice To Haves", etc.
3. maybe only top_n=3. 5 is too easily get irrelevant ones.
4. oervall looks having less powerful keywords than other sections.
"""

# different sections in a job description
key_respon = """

"""

required = """

"""

preferred = """

"""


def loading_model():
    """"
    loading sentence transformer model and keybert
    """
    embedded_model = SentenceTransformer('all-mpnet-base-v2')
    kw_model = KeyBERT(model=embedded_model)

    return kw_model


def key_respon_keywords(key_respon):
    """
    extract key responsibilities keywords
    """
    # determine whether there is texts
    if key_respon.strip() == "":
        return False

    kw_model = loading_model()
    keywords = kw_model.extract_keywords(key_respon, top_n=3)

    return keywords


def required_keywords(required):
    """
    extract required skills & backgraound keywords
    """
    # determine whether there is texts
    if required.strip() == "":
        return False

    kw_model = loading_model()
    keywords = kw_model.extract_keywords(required, top_n=3)

    return keywords


def preferred_keywords(preferred):
    """
    extract preferred skills keywords
    """
    # determine whether there is texts
    if preferred.strip() == "":
        return False

    kw_model = loading_model()
    keywords = kw_model.extract_keywords(preferred, top_n=3)

    return keywords