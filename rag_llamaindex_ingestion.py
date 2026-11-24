from pathlib import Path

import chromadb
from llama_index.core import Document
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_llamaindex import create_llamaindex_embeddings


def read_llamaindex_document(path: Path, source: str) -> Document:
    """Читает текстовый файл целиком для индексации через LlamaIndex.

    Args:
        path: Путь к непустому текстовому файлу в UTF-8.
        source: Постоянный уникальный идентификатор файла в индексе.

    Returns:
        Документ LlamaIndex с полным текстом и метаданными источника.

    Raises:
        ValueError: Файл не содержит текста.
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Файл пуст: {path.name}")
    return Document(text=text, id_=source, metadata={"source": source})


def build_llamaindex_pipeline(store: ChromaVectorStore) -> IngestionPipeline:
    """Собирает нарезку, эмбеддинги и сохранение фрагментов в Chroma.

    Args:
        store: Обёртка LlamaIndex над выбранной коллекцией Chroma.

    Returns:
        Пайплайн с учётом изменений документов в рамках одного сеанса.
    """
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=128, chunk_overlap=16),
            create_llamaindex_embeddings(),
        ],
        docstore=SimpleDocumentStore(),
        vector_store=store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )


def sync_llamaindex_document(path: Path, source: str, pipeline: IngestionPipeline) -> int:
    """Обрабатывает один текстовый файл через ранее созданный пайплайн.

    Args:
        path: Путь к текстовому файлу в UTF-8.
        source: Постоянный уникальный идентификатор файла в индексе.
        pipeline: Пайплайн, который хранит историю синхронизации.

    Returns:
        Количество фрагментов, обработанных за этот вызов. Ноль, если файл не изменился.
    """
    document = read_llamaindex_document(path, source)
    nodes = pipeline.run(documents=[document])
    return len(nodes)


def open_llamaindex_store(chroma_dir: Path) -> ChromaVectorStore:
    """Открывает постоянную коллекцию Chroma для документов LlamaIndex.

    Args:
        chroma_dir: Каталог, в котором Chroma хранит тексты и векторы.

    Returns:
        Обёртка LlamaIndex над коллекцией, готовая к записи и поиску.
    """
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name="llamaindex-documents",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    return ChromaVectorStore(chroma_collection=collection)


def restore_llamaindex_pipeline(store: ChromaVectorStore, state_dir: Path) -> IngestionPipeline:
    """Собирает пайплайн и загружает состояние предыдущего запуска при наличии.

    Args:
        store: Постоянная коллекция, которую использует пайплайн.
        state_dir: Каталог сохранённого состояния пайплайна.

    Returns:
        Пайплайн с готовой нарезкой, клиентом эмбеддингов и историей обработки.
    """
    pipeline = build_llamaindex_pipeline(store)
    if state_dir.exists():
        pipeline.load(persist_dir=str(state_dir))
    return pipeline
