import os
import re
from pathlib import Path

from dotenv import load_dotenv

HANDBOOK = Path(__file__).resolve().parent / "handbook.txt"
REFUSAL = "В документе этого нет"
Found = list[tuple[float, str]]


def load_model_names() -> tuple[str, str]:
    """Загружает настройки и возвращает имена чат-модели и модели эмбеддингов."""
    load_dotenv()
    return os.environ["CHAT_MODEL"], os.environ["EMBED_MODEL"]


def split_into_paragraphs(
        text: str, *, detect_sections: bool = True
) -> list[tuple[str, str]]:
    """Делит документ на абзацы и запоминает раздел каждого.

    Абзацы разделены пустой строкой. По умолчанию блок из заглавных букв
    считается заголовком раздела и в результат не попадает.

    Args:
        text: Полный текст документа.
        detect_sections: Считать ли блоки заглавными буквами заголовками.

    Returns:
        Список пар (название раздела, текст абзаца) в порядке документа.
    """
    paragraphs = []
    section = ""
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if detect_sections and block.isupper():
            section = block
            continue
        paragraphs.append((section, block))
    return paragraphs


def split_into_sentences(paragraph: str) -> list[str]:
    """Делит абзац на предложения по точке, восклицательному и вопросительному знаку.

    Args:
        paragraph: Текст одного абзаца.

    Returns:
        Список предложений, знак в конце каждого сохраняется.
    """
    return re.split(r"(?<=[.!?])\s+", paragraph)


def chunk_paragraphs(
        paragraphs: list[tuple[str, str]], n: int = 1, step: int | None = None
) -> list[str]:
    """Режет готовые абзацы на окна предложений.

    Args:
        paragraphs: Пары (название раздела, текст абзаца) по порядку.
        n: Сколько предложений входит в окно.
        step: Сдвиг окна. Если None, окно сдвигается на n предложений.

    Returns:
        Куски с заголовком раздела в первой строке.
    """
    chunks = []
    for section, paragraph in paragraphs:
        sentences = split_into_sentences(paragraph)
        for i in range(0, len(sentences), step if step is not None else n):
            group = " ".join(sentences[i:i + n])
            chunks.append(f"[{section}]\n{group}")
    return chunks


def chunk_by_sentences(
        text: str, n: int = 1, step: int | None = None, *, detect_sections: bool = True
) -> list[str]:
    """Режет документ на куски по n предложений внутри каждого абзаца.

    Окно не выходит за границу абзаца, поэтому последний кусок абзаца может быть короче.

    Args:
        text: Полный текст документа.
        n: Сколько предложений в одном куске.
        step: На сколько предложений сдвигается окно. По умолчанию равен n,
            то есть куски не перекрываются. При step < n соседние куски перекрываются.
        detect_sections: Считать ли блоки заглавными буквами заголовками.

    Returns:
        Список кусков вида "[РАЗДЕЛ]\\nпредложения через пробел".
    """
    paragraphs = split_into_paragraphs(text, detect_sections=detect_sections)
    return chunk_paragraphs(paragraphs, n=n, step=step)


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
