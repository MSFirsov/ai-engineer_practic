import os
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

client = OpenAI()
CHAT_MODEL = os.environ["CHAT_MODEL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
HANDBOOK = Path(__file__).resolve().parent / "handbook.txt"

Chunker = Callable[[str], list[str]]
Found = list[tuple[float, str]]


def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_into_paragraphs(text: str) -> list[tuple[str, str]]:
    paragraphs = []
    section = ""
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.isupper():
            section = block
            continue
        paragraphs.append((section, block))
    return paragraphs


def split_into_chunks(text: str) -> list[str]:
    return [f"[{section}]\n{block}" for section, block in split_into_paragraphs(text)]


def split_into_sentences(paragraph: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", paragraph)


def chunk_by_sentences(text: str, n: int = 1, step: int | None = None) -> list[str]:
    chunks = []
    for section, paragraph in split_into_paragraphs(text):
        sentences = split_into_sentences(paragraph)
        for i in range(0, len(sentences), step if step is not None else n):
            group = " ".join(sentences[i:i + n])
            chunks.append(f"[{section}]\n{group}")
    return chunks


def chunk_by_sections(text: str) -> list[str]:
    groups: dict[str, list[str]] = {}
    for section, paragraph in split_into_paragraphs(text):
        if section not in groups:
            groups[section] = []
        groups[section].append(paragraph)
    chunks = []
    for section, paragraphs in groups.items():
        chunks.append(f"[{section}]\n" + "\n".join(paragraphs))
    return chunks


def build_index(path: Path = HANDBOOK, chunker: Chunker = partial(chunk_by_sentences, n=2)) -> tuple[list[str], np.ndarray]:
    chunks = chunker(path.read_text(encoding="utf-8"))
    return chunks, embed(chunks)


def search(query: str, chunks: list[str], chunk_vectors: np.ndarray, k: int = 3) -> Found:
    query_vec = embed([query])[0]
    scores = [cosine(query_vec, v) for v in chunk_vectors]
    ranked = sorted(zip(scores, chunks), reverse=True)
    return ranked[:k]


def build_prompt(question: str, found: Found) -> str:
    rule = "Ответь на вопрос, используя только информацию из контекста. Если в контексте нет ответа, скажи: «В документе этого нет»."
    context = "\n\n".join(chunk for score, chunk in found)
    return f"{rule}\n\nКонтекст:\n{context}\n\nВопрос: {question}"


def ask(question: str, chunks: list[str], chunk_vectors: np.ndarray, k: int = 3) -> tuple[str, Found]:
    found = search(question, chunks, chunk_vectors, k)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, found)}],
        temperature=0.2,
    )
    return response.choices[0].message.content, found


if __name__ == "__main__":
    chunks, chunk_vectors = build_index()
    print(f"Кусков: {len(chunks)}, векторы: {chunk_vectors.shape}")

    while True:
        question = input("\nВопрос: ")
        if question.lower() in ("выход", "exit"):
            break
        answer, found = ask(question, chunks, chunk_vectors)

        print("\n--- Найденные куски ---")
        for score, chunk in found:
            print(f"{score:.3f}  {chunk[:80]!r}")
        print("\n--- Ответ ---")
        print(answer)
