import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict]:
    """
    Chunk a full document (list of page dicts from extract_pages) by token count
    across page boundaries, with overlap between consecutive chunks.

    Returns list of dicts:
    [{"text": str, "chunk_index": int, "page_number": int}, ...]
    page_number = the page where this chunk *starts*.
    """
    # Build a flat token list, tracking which page each token came from
    all_tokens = []
    token_page_map = []  # parallel list: token_page_map[i] = page_number of token i

    for page in pages:
        page_tokens = ENCODING.encode(page["text"])
        all_tokens.extend(page_tokens)
        token_page_map.extend([page["page_number"]] * len(page_tokens))

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(all_tokens):
        end = min(start + chunk_size, len(all_tokens))
        chunk_tokens = all_tokens[start:end]
        chunks.append({
            "text": ENCODING.decode(chunk_tokens),
            "chunk_index": chunk_index,
            "page_number": token_page_map[start],  # page where chunk starts
        })

        if end >= len(all_tokens):
            break

        start = end - overlap
        chunk_index += 1

    return chunks