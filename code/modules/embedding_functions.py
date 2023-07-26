from config import GPT_TOKENISER, T5_TOKENISER, BART_TOKENISER, GPT_EMBEDDING_MODEL, ModelNotSupportedError, openai


# Used throughout


def num_tokens(
        text: str,
        token_model  # = GPT_TOKENISER
) -> int:
    """
    Returns the number of tokens in a string.
    """
    if token_model == GPT_TOKENISER:
        return len(token_model.encode(text))
    elif token_model == T5_TOKENISER:
        return len(token_model.tokenize(text))
    elif token_model == BART_TOKENISER:
        return len(token_model.tokenize(text))
    else:
        raise ModelNotSupportedError('The model type isn\'t currently supported. Please select from GPT, T5 and '
                                     'BART.')


def get_embedding(content: list or str, embedding_model: str = GPT_EMBEDDING_MODEL):
    """
        Returns the embedding of a string given an embedding model.
    """
    if embedding_model == GPT_EMBEDDING_MODEL:
        return openai.Embedding.create(input=content, model=embedding_model)
    else:
        return embedding_model.encode(content)


def is_float(value):
    """
    Check if a value is a float.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False
