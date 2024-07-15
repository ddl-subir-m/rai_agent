from typing import List


def chunk_code(code: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    lines = code.split('\n')
    chunks = []
    for i in range(0, len(lines), chunk_size - overlap):
        chunk = '\n'.join(lines[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
