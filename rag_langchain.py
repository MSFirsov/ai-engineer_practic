import os

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import rag


def create_langchain_embeddings() -> OpenAIEmbeddings:
    """Настраивает получение эмбеддингов через локальную LM Studio."""
    return OpenAIEmbeddings(
        model=rag.EMBED_MODEL,
        base_url=os.environ["OPENAI_BASE_URL"],
        check_embedding_ctx_length=False,
    )


def build_langchain_store(chunks: list[str]) -> InMemoryVectorStore:
    """Создаёт векторное хранилище LangChain в памяти.

    Args:
        chunks: Готовые фрагменты текста.

    Returns:
        Хранилище с текстами и векторами фрагментов.
    """
    store = InMemoryVectorStore(create_langchain_embeddings())
    documents = [Document(page_content=chunk) for chunk in chunks]
    store.add_documents(documents)
    return store


def search_langchain(query: str, store: InMemoryVectorStore, k: int = 3) -> rag.Found:
    """Ищет в LangChain и приводит результат к формату проекта.

    Args:
        query: Текст поискового запроса.
        store: Индекс фрагментов LangChain.
        k: Максимальное число результатов.

    Returns:
        Пары (оценка близости, текст). При k <= 0 возвращает пустой список.
    """
    if k <= 0:
        return []
    matches = store.similarity_search_with_score(query, k=k)
    return [(float(score), document.page_content) for document, score in matches]


def build_langchain_answer_chain() -> Runnable[dict[str, str], str]:
    """Собирает цепочку из шаблона, чат-модели и вывода текста.

    Returns:
        Цепочка, которая принимает словарь с prompt и возвращает строку.
    """
    prompt_template = ChatPromptTemplate.from_messages([("human", "{prompt}")])
    chat = ChatOpenAI(
        model=rag.CHAT_MODEL,
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=0.2,
        use_responses_api=False,
    )
    return prompt_template | chat | StrOutputParser()


def ask_langchain(
    question: str,
    store: InMemoryVectorStore,
    answer_chain: Runnable[dict[str, str], str],
    k: int = 3,
) -> tuple[str, rag.Found]:
    """Ищет фрагменты и отвечает через цепочку LangChain.

    Args:
        question: Вопрос пользователя.
        store: Индекс фрагментов LangChain.
        answer_chain: Готовая цепочка генерации ответа.
        k: Максимальное число фрагментов для контекста.

    Returns:
        Пара (ответ модели, найденные фрагменты с оценками).
    """
    found = search_langchain(question, store, k=k)
    prompt = rag.build_prompt(question, found)
    answer = answer_chain.invoke({"prompt": prompt})
    return answer, found
