import hashlib
import math
import re
import readline
import sys
from collections.abc import Callable
from functools import partial, lru_cache
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api.types import Metadata
from docx import Document
from openai import OpenAI
from openai.types.chat import (ChatCompletionMessageFunctionToolCall,
                               ChatCompletionMessageParam,
                               ChatCompletionFunctionToolParam)
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from sentence_transformers import CrossEncoder

from rag_common import (
    HANDBOOK,
    REFUSAL,
    Found,
    build_prompt,
    chunk_by_sentences,
    chunk_paragraphs,
    chunk_section,
    list_sources,
    load_model_names,
    split_into_paragraphs,
    split_into_sentences as split_into_sentences,
)

CHAT_MODEL, EMBED_MODEL = load_model_names()
client = OpenAI()
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "embeddings"
CHROMA_DIR = Path(__file__).resolve().parent / ".cache" / "chroma"

Chunker = Callable[[str], list[str]]
History = list[tuple[str, str]]
Searcher = Callable[[str, list[str], np.ndarray, int], Found]

MAX_HISTORY = 3
POOL = 10
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RRF_K = 60
STOP_WORDS = set(
    "а в во и или ли не на с со к ко по о об от до за из у что кто как когда где куда сколько "
    "какой какая какое какие если я мы вы он она они это то же бы для при есть нет можно нужно "
    "такое такой такая чем".split()
)
REWRITE_RULE = (
    "Ты переписываешь вопросы для поиска по документу. Перепиши последний вопрос пользователя так, "
    "чтобы он был понятен без истории диалога. Если вопрос и так понятен, верни его без изменений. "
    "Не отвечай на вопрос. Верни только один переписанный вопрос, без пояснений."
)


class StructuredAnswer(BaseModel):
    """Ответ по найденному контексту с признаком наличия ответа."""

    model_config = ConfigDict(extra="forbid", strict=True)
    has_answer: bool
    answer: str


class HandbookSearchArgs(BaseModel):
    """Аргументы поиска по справочнику, которые предлагает модель."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    query: str = Field(min_length=1)


HANDBOOK_SEARCH_TOOL: ChatCompletionFunctionToolParam = {
    "type": "function",
    "function": {
        "name": "search_handbook",
        "description": "Ищет подходящие фрагменты справочника сотрудников по текстовому запросу.",
        "parameters": HandbookSearchArgs.model_json_schema(),
    },
}

HANDBOOK_TOOL_RULE = (
    "Для каждого вопроса сначала вызови search_handbook ровно один раз. "
    "Передай в query самостоятельный запрос на русском языке. "
    "После получения результата ответь только по найденным фрагментам. "
    "Укажи has_answer=true, если в них есть ответ. "
    f"Если ответа нет, укажи has_answer=false и запиши в answer «{REFUSAL}»."
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


def split_into_chunks(text: str) -> list[str]:
    """Режет документ на куски по абзацам.

    Args:
        text: Полный текст документа.

    Returns:
        Список кусков вида "[РАЗДЕЛ]\\nтекст абзаца".
    """
    return [f"[{section}]\n{block}" for section, block in split_into_paragraphs(text)]


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


def build_index(path: Path = HANDBOOK, chunker: Chunker = partial(chunk_by_sentences, n=2)) -> tuple[
    list[str], np.ndarray]:
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


def get_collection() -> "chromadb.Collection":
    """Открывает локальную коллекцию Chroma для текущей модели эмбеддингов.

    Returns:
        Коллекция с косинусным расстоянием и готовыми внешними эмбеддингами.
    """
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return db.get_or_create_collection(
        name=f"handbook-{cache_key('')[:12]}",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def save_to_chroma(chunks: list[str], chunk_vectors: np.ndarray) -> int:
    """Сохраняет куски и их векторы в Chroma по стабильным идентификаторам.

    Args:
        chunks: Непустой список уникальных кусков текущего справочника.
        chunk_vectors: Матрица эмбеддингов в том же порядке, что и chunks.

    Returns:
        Общее число записей в коллекции после сохранения.
    """
    collection = get_collection()
    collection.upsert(
        ids=[cache_key(chunk) for chunk in chunks],
        documents=chunks,
        embeddings=chunk_vectors.tolist(),
    )
    return collection.count()


def search_chroma(query: str, k: int = 3) -> Found:
    """Ищет куски в сохранённой базе без повторного чтения справочника.

    Args:
        query: Текст поискового запроса.
        k: Максимальное число результатов.

    Returns:
        Пары (косинусная близость, кусок) от лучшего к худшему.
        Для пустой коллекции или k <= 0 возвращается пустой список.
    """
    collection = get_collection()
    if collection.count() == 0 or k <= 0:
        return []
    result = collection.query(
        query_embeddings=[embed_query(query).tolist()],
        n_results=min(k, collection.count()),
        include=["documents", "distances"],
    )
    documents = result["documents"][0]
    distances = result["distances"][0]
    return [(1 - distance, chunk) for distance, chunk in zip(distances, documents)]


def get_documents_collection() -> "chromadb.Collection":
    """Открывает отдельную коллекцию документов с метаданными.

    Returns:
        Коллекция для текущей embedding-модели с косинусным расстоянием.
    """
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return db.get_or_create_collection(
        name=f"documents-{cache_key('')[:12]}",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Извлекает текст из PDF постранично.

    Args:
        path: Путь к PDF-файлу с текстовым слоем.

    Returns:
        Пары (номер страницы с единицы, текст) для страниц с текстом.

    Raises:
        ValueError: PDF зашифрован или в нём не найден извлекаемый текст.
    """
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError(f"PDF зашифрован: {path.name}")

    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text().strip()
        if text:
            pages.append((number, text))

    if not pages:
        raise ValueError(f"В PDF нет извлекаемого текста: {path.name}")
    return pages


def read_docx_paragraphs(path: Path) -> list[tuple[str, str]]:
    """Читает абзацы DOCX и сохраняет ближайший заголовок раздела.

    Args:
        path: Путь к документу DOCX.

    Returns:
        Пары (название раздела, текст абзаца). Для абзацев перед первым
        заголовком название раздела — пустая строка.

    Raises:
        ValueError: В документе нет абзацев основного текста.
    """
    document = Document(str(path))
    paragraphs = []
    section = ""

    for paragraph in document.paragraphs:
        text = " ".join(paragraph.text.split())
        if not text:
            continue
        style = paragraph.style
        style_name = style.name if style is not None else None
        if style_name and style_name.startswith("Heading "):
            section = text
        else:
            paragraphs.append((section, text))

    if not paragraphs:
        raise ValueError(f"В DOCX нет абзацев основного текста: {path.name}")
    return paragraphs


def read_document_chunks(path: Path) -> list[tuple[str, int | None]]:
    """Читает файл и готовит его куски с указанием страниц.

    Args:
        path: Путь к DOCX, PDF с текстовым слоем или текстовому файлу в UTF-8.

    Returns:
        Пары (кусок, номер страницы). Для DOCX и TXT номер равен None.
    """
    chunks: list[tuple[str, int | None]] = []
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        for number, text in read_pdf_pages(path):
            for chunk in chunk_by_sentences(text, n=2, detect_sections=False):
                chunks.append((chunk, number))
    elif suffix == ".docx":
        paragraphs = read_docx_paragraphs(path)
        for chunk in chunk_paragraphs(paragraphs, n=2):
            chunks.append((chunk, None))
    else:
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_by_sentences(text, n=2):
            chunks.append((chunk, None))

    return chunks


def sync_document(path: Path, source: str) -> int:
    """Синхронизирует куски одного документа с базой.

    Args:
        path: Путь к DOCX, PDF с текстовым слоем или текстовому файлу в UTF-8.
        source: Постоянное уникальное имя документа, например handbook.txt.

    Returns:
        Число уникальных кусков этого источника после синхронизации.
        Пустой текстовый файл удаляет все записи только своего источника.

    Raises:
        ValueError: Некорректное имя источника, зашифрованный PDF,
            PDF без извлекаемого текста или DOCX без абзацев основного текста.
        OSError: Файл не удалось прочитать; записи базы при этом не меняются.
    """
    if not source.strip() or "\n" in source:
        raise ValueError("Нужно непустое имя источника без перевода строки")
    entries = list(dict.fromkeys(read_document_chunks(path)))
    chunks = []
    ids = []
    metadatas: list[Metadata] = []
    for chunk, page in entries:
        metadata: Metadata = {"source": source, "section": chunk_section(chunk)}
        id_text = f"{source}\n{chunk}"
        if page is not None:
            metadata["page"] = page
            id_text = f"{source}\n{page}\n{chunk}"
        chunks.append(chunk)
        ids.append(cache_key(id_text))
        metadatas.append(metadata)
    collection = get_documents_collection()
    old_ids = set(collection.get(where={"source": source}, include=[])["ids"])
    if chunks:
        vectors = embed_cached(chunks)
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
        )
    stale_ids = old_ids - set(ids)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    return len(chunks)


def search_documents(query: str, k: int = 3, source: str | None = None) -> Found:
    """Ищет по всем документам или одному источнику.

    Args:
        query: Поисковый запрос.
        k: Максимальное число результатов.
        source: Точное имя источника; None означает поиск по всем документам.

    Returns:
        Пары (близость, текст фрагмента с заголовком).
        В заголовке указаны имя источника и раздел для TXT и DOCX или номер страницы для PDF.
        При k <= 0 или отсутствии подходящих записей возвращается пустой список.
    """
    if k <= 0:
        return []
    collection = get_documents_collection()
    where = {"source": source} if source is not None else None
    count = len(collection.get(where=where, include=[])["ids"])
    if count == 0:
        return []
    result = collection.query(
        query_embeddings=[embed_query(query).tolist()],
        n_results=min(k, count),
        where=where,
        include=["documents", "distances", "metadatas"],
    )
    found = []
    for chunk, distance, metadata in zip(
            result["documents"][0], result["distances"][0], result["metadatas"][0]
    ):
        body = chunk.split("\n", 1)[1]
        page = metadata.get("page")
        if page is None:
            location = metadata["section"]
        else:
            location = f"стр. {page}"
        label = f"[{metadata['source']} / {location}]"
        found.append((1 - distance, f"{label}\n{body}"))
    return found


def documents_searcher(query: str, chunks: list[str], chunk_vectors: np.ndarray,
                       k: int = 3, *, source: str | None = None) -> Found:
    """Подключает поиск документов к ask и evaluate.

    Args:
        query: Поисковый запрос.
        chunks: Не используется; тексты находятся в базе.
        chunk_vectors: Не используется; векторы находятся в базе.
        k: Максимальное число результатов.
        source: Необязательный фильтр; передаётся только по имени.

    Returns:
        Найденные куски с именами источников и разделами или номерами страниц.
    """
    return search_documents(query, k, source)


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


@lru_cache(maxsize=1)
def get_reranker() -> "CrossEncoder":
    """Загружает CrossEncoder один раз на процесс.

    Returns:
        Локальная модель оценки пары «запрос — фрагмент» на CPU.
    """
    return CrossEncoder(
        RERANK_MODEL,
        device="cpu",
        max_length=512,
        cache_folder=str(Path(__file__).resolve().parent / ".cache" / "reranker"),
    )


def rerank(query: str, candidates: Found, k: int = 3) -> Found:
    """Переоценивает найденные фрагменты по отношению к запросу.

    Args:
        query: Текст поискового запроса.
        candidates: Кандидаты первого поиска; исходные оценки не используются.
        k: Максимальное число результатов.

    Returns:
        Пары (оценка CrossEncoder, исходный текст) по убыванию оценки.
        При пустом списке или k <= 0 возвращается пустой список.
    """
    if not candidates or k <= 0:
        return []
    texts = [text for _, text in candidates]
    pairs = [(query, text) for text in texts]
    scores = get_reranker().predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(zip(scores, texts), key=lambda item: item[0], reverse=True)
    return [(float(score), text) for score, text in ranked[:k]]


def reranked_search(query: str, chunks: list[str], chunk_vectors: np.ndarray,
                    k: int = 3, *, base_searcher: Searcher = search,
                    pool_size: int = 20) -> Found:
    """Отбирает кандидатов обычным поиском и переоценивает CrossEncoder.

    Args:
        query: Поисковый запрос.
        chunks: Куски для исходного поиска; для Chroma могут быть пустыми.
        chunk_vectors: Векторы для исходного поиска.
        k: Сколько результатов вернуть после реранкинга.
        base_searcher: Поиск первого этапа, по умолчанию NumPy.
        pool_size: Сколько кандидатов запросить на первом этапе.

    Returns:
        Найденные фрагменты с оценками CrossEncoder; при k <= 0 — пустой список.

    Raises:
        ValueError: Число кандидатов меньше запрошенного k.
    """
    if k <= 0:
        return []
    if pool_size < k:
        raise ValueError("pool_size должен быть не менее k")
    candidates = base_searcher(query, chunks, chunk_vectors, pool_size)
    return rerank(query, candidates, k)


def generate_structured_answer(question: str, found: Found) -> StructuredAnswer:
    """Запрашивает у чат-модели ответ по найденным фрагментам.

    Args:
        question: Исходный вопрос пользователя.
        found: Список пар (оценка, текст фрагмента).

    Returns:
        Объект с признаком наличия ответа и текстом, проверенный по схеме.

    Raises:
        RuntimeError: API не вернул разобранный структурированный ответ.
    """
    instruction = (
        "Укажи has_answer=true, если в контексте есть ответ на вопрос, иначе has_answer=false. "
        f"В поле answer запиши ответ. При has_answer=false запиши «{REFUSAL}»."
    )
    response = client.chat.completions.parse(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": build_prompt(question, found)},
        ],
        response_format=StructuredAnswer,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise RuntimeError("Модель не вернула структурированный ответ")
    return result


def request_handbook_search(
        messages: list[ChatCompletionMessageParam],
) -> ChatCompletionMessageFunctionToolCall:
    """Запрашивает у чат-модели вызов инструмента поиска.

    Args:
        messages: Сообщения с инструкцией и вопросом.

    Returns:
        Вызов search_handbook с идентификатором и аргументами в JSON.

    Raises:
        RuntimeError: Модель вернула не один вызов или выбрала другой инструмент.
    """
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=[HANDBOOK_SEARCH_TOOL],
        tool_choice="required",
        temperature=0,
    )
    calls = response.choices[0].message.tool_calls or []
    if len(calls) != 1:
        raise RuntimeError("Модель должна запросить ровно один инструмент")

    call = calls[0]
    if call.type != "function" or call.function.name != "search_handbook":
        raise RuntimeError("Модель запросила неизвестный инструмент")
    return call


def ask_with_search_tool(
        question: str, chunks: list[str], chunk_vectors: np.ndarray, k: int = 3,
) -> tuple[StructuredAnswer, Found]:
    """Выполняет поиск по запросу модели и возвращает ответ на вопрос.

    Args:
        question: Исходный вопрос пользователя.
        chunks: Фрагменты индексированного справочника.
        chunk_vectors: Матрица эмбеддингов фрагментов.
        k: Максимальное число фрагментов для ответа.

    Returns:
        Структурированный ответ и найденные фрагменты с оценками.

    Raises:
        RuntimeError: Модель не вернула ожидаемый вызов или разобранный ответ.
        ValidationError: Аргументы предложенного поиска не прошли проверку.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": HANDBOOK_TOOL_RULE},
        {"role": "user", "content": question},
    ]
    call = request_handbook_search(messages)
    args = HandbookSearchArgs.model_validate_json(call.function.arguments)
    found = search(args.query, chunks, chunk_vectors, k)
    context = "\n\n".join(chunk for _, chunk in found) or "Фрагменты не найдены."

    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": context,
    })

    response = client.chat.completions.parse(
        model=CHAT_MODEL,
        messages=messages,
        response_format=StructuredAnswer,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise RuntimeError("Модель не вернула структурированный ответ")
    return result, found


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


def chroma_searcher(query: str, chunks: list[str], chunk_vectors: np.ndarray, k: int = 3) -> Found:
    """Подключает поиск Chroma к общему интерфейсу Searcher.

    Args:
        query: Поисковый запрос.
        chunks: Не используется; тексты уже находятся в Chroma.
        chunk_vectors: Не используется; векторы уже находятся в Chroma.
        k: Максимальное число результатов.

    Returns:
        Найденные пары (близость, текст).
    """
    return search_chroma(query, k)


def prepare_search(use_chroma: bool = False, use_hybrid: bool = False) -> tuple[list[str], np.ndarray, Searcher]:
    """Готовит данные и функцию для выбранного способа поиска.

    Args:
        use_chroma: Использовать сохранённую базу без build_index.
        use_hybrid: Использовать гибридный поиск по индексу в памяти.

    Returns:
        Куски, векторы и функция поиска. Для Chroma данные пустые.

    Raises:
        ValueError: Выбраны несовместимые режимы или база пуста.
    """
    if use_chroma and use_hybrid:
        raise ValueError("Выбери один режим: --chroma или --hybrid")
    if use_chroma:
        if get_collection().count() == 0:
            raise ValueError("База пуста. Сначала выполни: uv run rag.py --index")
        return [], np.empty((0, 0)), chroma_searcher
    chunks, chunk_vectors = build_index()
    return chunks, chunk_vectors, hybrid_search if use_hybrid else search


def main() -> None:
    """Запускает загрузку базы, поиск без генерации или диалог по справочнику."""
    use_chroma = "--chroma" in sys.argv
    use_hybrid = "--hybrid" in sys.argv
    search_only = "--search-only" in sys.argv
    if "--index" in sys.argv:
        if use_chroma or use_hybrid or search_only:
            raise SystemExit("Запускай --index отдельно от режимов поиска")
        chunks, chunk_vectors = build_index()
        print(f"Записей в Chroma: {save_to_chroma(chunks, chunk_vectors)}")
        return
    try:
        chunks, chunk_vectors, searcher = prepare_search(use_chroma, use_hybrid)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Поиск: {searcher.__name__}")
    if search_only:
        print("Режим поиска: без ответа модели и без истории диалога")
    history: History = []
    while True:
        question = input("\nВопрос: ")
        if question.lower() in ("выход", "exit"):
            break
        if search_only:
            found = searcher(question, chunks, chunk_vectors, 3)
            print("\n--- Найденные куски ---")
            for score, chunk in found:
                print(f"{score:.3f} {chunk}")
            continue
        answer, found = ask(question, chunks, chunk_vectors, history=history[-MAX_HISTORY:], searcher=searcher)
        history.append((question, answer))
        print("\n--- Найденные куски ---")
        for score, chunk in found:
            print(f"{score:.3f} {chunk[:80]!r}")
        print("\n--- Ответ ---")
        print(answer)
        if REFUSAL not in answer:
            print("Источник: ", "; ".join(list_sources(found)))


if __name__ == "__main__":
    main()
