import os

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from rag_common import Found, load_model_names


def create_llamaindex_embeddings() -> OpenAILikeEmbedding:
    """Настраивает модель эмбеддингов LlamaIndex для локальной LM Studio.

    Returns:
        Клиент эмбеддингов с настройками из окружения.
    """
    _, embed_model = load_model_names()
    return OpenAILikeEmbedding(
        model_name=embed_model,
        api_base=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )


def build_llamaindex_index(chunks: list[str]) -> VectorStoreIndex:
    """Создаёт индекс LlamaIndex из готовых фрагментов справочника.

    Args:
        chunks: Фрагменты текста, которые надо проиндексировать.

    Returns:
        Индекс с текстами и их эмбеддингами в памяти.
    """
    nodes = [TextNode(text=chunk) for chunk in chunks]
    return VectorStoreIndex(nodes, embed_model=create_llamaindex_embeddings())


def search_llamaindex(query: str, index: VectorStoreIndex, k: int = 3) -> Found:
    """Находит подходящие фрагменты через LlamaIndex.

    Args:
        query: Текст поискового запроса.
        index: Готовый индекс фрагментов.
        k: Максимальное число результатов.

    Returns:
        Пары (оценка близости, текст). При k <= 0 возвращает пустой список.
    """
    if k <= 0:
        return []
    retriever = index.as_retriever(similarity_top_k=k)
    matches = retriever.retrieve(query)
    return [(match.get_score(), match.get_content()) for match in matches]
