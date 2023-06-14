from modules.config import *
# Used throughout


def num_tokens(
        text: str,
        token_model=GPT_ENCODING
) -> int:
    """
    Returns the number of tokens in a string.
    """
    if token_model == GPT_ENCODING:
        return len(token_model.encode(text))
    elif token_model == BERT_ENCODING:
        return len(token_model.tokenize(text))


def get_embedding(content: list or str, embedding_model: str = GPT_EMBEDDING_MODEL):
    """
        Returns the embedding of a string given an embedding model.
    """
    if embedding_model == GPT_EMBEDDING_MODEL:
        return openai.Embedding.create(input=content, model=embedding_model)
    else:
        similarity_model = SentenceTransformer(embedding_model)
        return similarity_model.encode(content)


def is_float(value):
    """
    Check if a value is a float.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False
