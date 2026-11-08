import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

client = OpenAI()
CHAT_MODEL = os.environ["CHAT_MODEL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
HANDBOOK = Path(__file__).resolve().parent / "handbook.txt"


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_into_chunks(text):
    chunks = []
    section = ""
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.isupper():
            section = block
            continue
        chunks.append(f"[{section}]\n{block}")
    return chunks


def build_index(path=HANDBOOK):
    chunks = split_into_chunks(path.read_text(encoding="utf-8"))
    return chunks, embed(chunks)


def search(query, chunks, chunk_vectors, k=3):
    query_vec = embed([query])[0]
    scores = [cosine(query_vec, v) for v in chunk_vectors]
    ranked = sorted(zip(scores, chunks), reverse=True)
    return ranked[:k]


def build_prompt(question, found):
    rule = "Ответь на вопрос, используя только информацию из контекста. Если в контексте нет ответа, скажи: «В документе этого нет»."
    context = "\n\n".join(chunk for score, chunk in found)
    return f"{rule}\n\nКонтекст:\n{context}\n\nВопрос: {question}"


def ask(question, chunks, chunk_vectors, k=3):
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
