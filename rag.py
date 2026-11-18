import hashlib
import math
import os
import re
import readline
import sys
from collections.abc import Callable
from functools import partial, lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import numpy as np

load_dotenv()

client = OpenAI()
CHAT_MODEL = os.environ["CHAT_MODEL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
HANDBOOK = Path(__file__).resolve().parent / "handbook.txt"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "embeddings"

Chunker = Callable[[str], list[str]]
Found = list[tuple[float, str]]
History = list[tuple[str, str]]
Searcher = Callable[[str, list[str], np.ndarray, int], Found]

MAX_HISTORY = 3
POOL = 10
RRF_K = 60
STOP_WORDS = set(
    "а в во и или ли не на с со к ко по о об от до за из у что кто как когда где куда сколько "
    "какой какая какое какие если я мы вы он она они это то же бы для при есть нет можно нужно "
    "такое такой такая чем".split()
)
REFUSAL = "В документе этого нет"
REWRITE_RULE = (
    "Ты переписываешь вопросы для поиска по документу. Перепиши последний вопрос пользователя так, "
    "чтобы он был понятен без истории диалога. Если вопрос и так понятен, верни его без изменений. "
    "Не отвечай на вопрос. Верни только один переписанный вопрос, без пояснений."
)


def embed(texts: list[str]) -> np.ndarray:
    """Считает эмбеддинги текстов одним запросом к модели.

    Args:
        texts: Тексты, для которых нужны векторы.

    Returns:
        Матрица размера (число текстов, размерность модели), строки идут в порядке texts.
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])


def cache_key(text: str) -> str:
    """Строит ключ кэша для эмбеддинга текста.

    В ключ входит имя модели, поэтому при смене EMBED_MODEL старые векторы не подхватятся.

    Args:
        text: Текст куска.

    Returns:
        SHA-256 от имени модели и текста в виде hex-строки.
    """
    data = f"{EMBED_MODEL}\n{text}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def embed_cached(texts: list[str]) -> np.ndarray:
    """Возвращает эмбеддинги текстов, считая только те, которых нет в кэше.

    Каждый вектор хранится в отдельном файле CACHE_DIR/<cache_key>.npy. Недостающие
    считаются одним запросом и сохраняются на диск. Печатает, сколько векторов взято
    из кэша и сколько посчитано.

    Args:
        texts: Тексты, для которых нужны векторы.

    Returns:
        Матрица размера (число текстов, размерность модели), строки идут в порядке texts.
    """
    paths = [CACHE_DIR / f"{cache_key(t)}.npy" for t in texts]
    missing = [i for i, p in enumerate(paths) if not p.exists()]
    if missing:
        vectors = embed([texts[i] for i in missing])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for i, vector in zip(missing, vectors):
            np.save(paths[i], vector)
    print(f"эмбеддинги: из кэша {len(texts) - len(missing)}, посчитано {len(missing)}")
    return np.array([np.load(p) for p in paths])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Считает косинусную близость двух векторов.

    Args:
        a: Первый вектор.
        b: Второй вектор той же размерности.

    Returns:
        Число от -1 до 1: чем ближе к 1, тем больше похожи направления векторов.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_into_paragraphs(text: str) -> list[tuple[str, str]]:
    """Делит документ на абзацы и запоминает раздел каждого.

    Абзацы разделены пустой строкой. Блок, написанный целиком заглавными буквами,
    считается заголовком раздела и в результат не попадает.

    Args:
        text: Полный текст документа.

    Returns:
        Список пар (название раздела, текст абзаца) в порядке документа.
    """
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
    """Режет документ на куски по абзацам.

    Args:
        text: Полный текст документа.

    Returns:
        Список кусков вида "[РАЗДЕЛ]\\nтекст абзаца".
    """
    return [f"[{section}]\n{block}" for section, block in split_into_paragraphs(text)]


def split_into_sentences(paragraph: str) -> list[str]:
    """Делит абзац на предложения по точке, восклицательному и вопросительному знаку.

    Args:
        paragraph: Текст одного абзаца.

    Returns:
        Список предложений, знак в конце каждого сохраняется.
    """
    return re.split(r"(?<=[.!?])\s+", paragraph)


def chunk_by_sentences(text: str, n: int = 1, step: int | None = None) -> list[str]:
    """Режет документ на куски по n предложений внутри каждого абзаца.

    Окно не выходит за границу абзаца, поэтому последний кусок абзаца может быть короче.

    Args:
        text: Полный текст документа.
        n: Сколько предложений в одном куске.
        step: На сколько предложений сдвигается окно. По умолчанию равен n,
            то есть куски не перекрываются. При step < n соседние куски перекрываются.

    Returns:
        Список кусков вида "[РАЗДЕЛ]\\nпредложения через пробел".
    """
    chunks = []
    for section, paragraph in split_into_paragraphs(text):
        sentences = split_into_sentences(paragraph)
        for i in range(0, len(sentences), step if step is not None else n):
            group = " ".join(sentences[i:i + n])
            chunks.append(f"[{section}]\n{group}")
    return chunks


def chunk_by_sections(text: str) -> list[str]:
    """Режет документ на куски по разделам: один раздел — один кусок.

    Args:
        text: Полный текст документа.

    Returns:
        Список кусков вида "[РАЗДЕЛ]\\nвсе абзацы раздела, каждый с новой строки".
    """
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
    """Читает документ, режет его на куски и считает их эмбеддинги.

    Args:
        path: Путь к текстовому файлу документа.
        chunker: Функция нарезки: принимает текст, возвращает список кусков.
            По умолчанию окно из двух предложений.

    Returns:
        Пара (куски, матрица векторов): строка i матрицы соответствует куску i.
    """
    chunks = chunker(path.read_text(encoding="utf-8"))
    return chunks, embed_cached(chunks)


@lru_cache(maxsize=1000)
def embed_query(query: str) -> np.ndarray:
    """Считает эмбеддинг поискового запроса и запоминает его в памяти.

    Повторный вызов с той же строкой не обращается к модели. Кэш живёт
    до конца работы программы и хранит до 1000 запросов.

    Args:
        query: Текст запроса.

    Returns:
        Вектор запроса.
    """
    return embed([query])[0]


def search(query: str, chunks: list[str], chunk_vectors: np.ndarray, k: int = 3) -> Found:
    """Находит k кусков, самых близких к запросу по косинусной близости.

    Args:
        query: Текст запроса.
        chunks: Куски документа.
        chunk_vectors: Матрица эмбеддингов кусков, строка i соответствует chunks[i].
        k: Сколько кусков вернуть.

    Returns:
        Список пар (близость, кусок), отсортированный от самого похожего к менее похожим.
    """
    query_vec = embed_query(query)
    scores = [cosine(query_vec, v) for v in chunk_vectors]
    ranked = sorted(zip(scores, chunks), reverse=True)
    return ranked[:k]


def tokenize(text: str) -> list[str]:
    """Разбивает текст на слова для поиска по словам.

    Слова приводятся к нижнему регистру, «ё» заменяется на «е». Знаки @ и # отбрасываются,
    дефис и двоеточие внутри слова сохраняются: "#it-help" даёт "it-help", "11:15" остаётся целым.

    Args:
        text: Текст куска или запроса.

    Returns:
        Список слов без стоп-слов из STOP_WORDS.
    """
    words = re.findall(r"\w+(?:[:-]\w+)*", text.lower().replace("ё", "е"))
    words = [word for word in words if word not in STOP_WORDS]
    return words


def bm25_scores(query: str, chunks: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Считает для каждого куска оценку BM25: насколько он подходит запросу по совпадению слов.

    Оценка куска — сумма по словам запроса. Слово даёт тем больше, чем реже оно
    встречается в документе и чем чаще в этом куске. Длинные куски получают штраф.

    Args:
        query: Текст запроса.
        chunks: Куски документа.
        k1: Насколько быстро перестают помогать повторы слова в куске.
        b: Сила штрафа за длину куска, от 0 (нет штрафа) до 1.

    Returns:
        Оценки в порядке chunks. Ноль означает, что общих слов с запросом нет.
    """
    chunk_words = [tokenize(chunk) for chunk in chunks]
    avg_len = sum(len(words) for words in chunk_words) / len(chunks)
    scores = [0.0] * len(chunks)
    for word in set(tokenize(query)):
        chunks_with_word = sum(1 for words in chunk_words if word in words)
        if chunks_with_word == 0:
            continue
        rarity = math.log((len(chunks) - chunks_with_word + 0.5) / (chunks_with_word + 0.5) + 1)
        for i, words in enumerate(chunk_words):
            count = words.count(word)
            scores[i] += rarity * count * (k1 + 1) / (count + k1 * (1 - b + b * len(words) / avg_len))
    return scores


def keyword_search(query: str, chunks: list[str], k: int = 3) -> Found:
    """Находит k кусков, лучших по совпадению слов с запросом (BM25).

    Args:
        query: Текст запроса.
        chunks: Куски документа.
        k: Сколько кусков вернуть.

    Returns:
        Список пар (оценка BM25, кусок) от лучшего к худшему. Куски без общих слов
        с запросом не возвращаются, поэтому список может быть короче k или пустым.
    """
    ranked = sorted(zip(bm25_scores(query, chunks), chunks), reverse=True)
    return [(score, chunk) for score, chunk in ranked[:k] if score > 0]


def hybrid_search(query: str, chunks: list[str], chunk_vectors: np.ndarray, k: int = 3) -> Found:
    """Объединяет поиск по смыслу и поиск по словам через Reciprocal Rank Fusion.

    Каждый поиск возвращает POOL лучших кусков. Кусок получает 1 / (RRF_K + место)
    за каждый список, в котором он есть, и эти числа складываются.

    Args:
        query: Текст запроса.
        chunks: Куски документа.
        chunk_vectors: Матрица эмбеддингов кусков.
        k: Сколько кусков вернуть.

    Returns:
        Список пар (суммарная оценка, кусок) от лучшего к худшему.
    """
    by_meaning = search(query, chunks, chunk_vectors, POOL)
    by_words = keyword_search(query, chunks, POOL)
    scores: dict[str, float] = {}
    for found in (by_meaning, by_words):
        for rank, (score, chunk) in enumerate(found, start=1):
            scores[chunk] = scores.get(chunk, 0.0) + 1 / (RRF_K + rank)
    ranked = sorted(((score, chunk) for chunk, score in scores.items()), reverse=True)
    return ranked[:k]


def build_prompt(question: str, found: Found) -> str:
    """Собирает промпт для генератора: правило, найденный контекст и вопрос.

    Args:
        question: Вопрос пользователя.
        found: Результат search: пары (близость, кусок).

    Returns:
        Текст промпта. Правило требует отвечать только по контексту
        и отказывать фразой «В документе этого нет», если ответа нет.
    """
    rule = f"Ответь на вопрос, используя только информацию из контекста. Если в контексте нет ответа, скажи: «{REFUSAL}»."
    context = "\n\n".join(chunk for score, chunk in found)
    return f"{rule}\n\nКонтекст:\n{context}\n\nВопрос: {question}"


def chunk_section(chunk: str) -> str:
    """Достаёт название раздела из куска.

    Args:
        chunk: Кусок вида "[РАЗДЕЛ]\\nтекст".

    Returns:
        Название раздела без квадратных скобок.
    """
    return chunk.split("\n", 1)[0].strip("[]")


def list_sources(found: Found) -> list[str]:
    """Собирает разделы найденных кусков без повторов.

    Args:
        found: Результат search: пары (близость, кусок).

    Returns:
        Названия разделов в порядке выдачи: первым идёт раздел самого похожего куска.
    """
    sections: list[str] = []
    for score, chunk in found:
        section = chunk_section(chunk)
        if section not in sections:
            sections.append(section)
    return sections


def rewrite_query(question: str, history: History) -> str:
    """Переписывает уточняющий вопрос так, чтобы он был понятен без истории диалога.

    Нужна для поиска: в вопросе вроде «А на спорт?» нет слов, по которым можно найти
    нужный кусок. Без истории модель не вызывается.

    Args:
        question: Последний вопрос пользователя.
        history: Предыдущие пары (вопрос, ответ) от старых к новым.

    Returns:
        Самостоятельный вопрос. Если истории нет или модель вернула пустой ответ,
        возвращается исходный вопрос.
    """
    if not history:
        return question
    dialog = "\n".join(f"Пользователь: {q}\nАссистент: {a}" for q, a in history)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_RULE},
            {"role": "user", "content": f"История:\n{dialog}\n\nПоследний вопрос: {question}"},
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or question).strip()


def ask(question: str, chunks: list[str], chunk_vectors: np.ndarray, k: int = 3,
        history: History | None = None, searcher: Searcher = search) -> tuple[str, Found]:
    """Отвечает на вопрос по документу с учётом истории диалога.

    Переписывает вопрос, ищет по переписанному варианту и отдаёт модели историю
    вместе с промптом, в котором стоит исходный вопрос. Если вопрос был переписан,
    печатает запрос, по которому шёл поиск.

    Args:
        question: Вопрос пользователя.
        chunks: Куски документа.
        chunk_vectors: Матрица эмбеддингов кусков.
        k: Сколько кусков положить в контекст.
        history: Предыдущие пары (вопрос, ответ). None означает пустую историю.
        searcher: Функция поиска. По умолчанию поиск по смыслу (search).

    Returns:
        Пара (ответ модели, найденные куски с их близостью).
    """
    history = history or []
    query = rewrite_query(question, history)
    if query != question:
        print(f"Запрос для поиска: {query}")
    found = searcher(query, chunks, chunk_vectors, k)
    messages: list[ChatCompletionMessageParam] = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": build_prompt(question, found)})
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or "", found


if __name__ == "__main__":
    chunks, chunk_vectors = build_index()
    print(f"Кусков: {len(chunks)}, векторы: {chunk_vectors.shape}")

    searcher: Searcher = search
    if "--hybrid" in sys.argv:
        searcher = hybrid_search
    print(f"Поиск: {searcher.__name__}")

    history: History = []
    while True:
        question = input("\nВопрос: ")
        if question.lower() in ("выход", "exit"):
            break
        answer, found = ask(question, chunks, chunk_vectors, history=history[-MAX_HISTORY:], searcher=searcher)
        history.append((question, answer))
        print("\n--- Найденные куски ---")
        for score, chunk in found:
            print(f"{score:.3f}  {chunk[:80]!r}")
        print("\n--- Ответ ---")
        print(answer)
        if REFUSAL not in answer:
            print("Источник: ", "; ".join(list_sources(found)))
