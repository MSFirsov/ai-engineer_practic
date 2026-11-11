import time

import numpy as np

from rag import Chunker, Found, build_index, search, ask, split_into_chunks, chunk_by_sentences, chunk_by_sections

REFUSAL = "В документе этого нет"

EVAL_SET = [
    {"question": "Сколько дней отпуска в году?", "expected": "28 календарных"},
    {"question": "Можно ли поработать месяц из Турции?", "expected": "из-за границы"},
    {"question": "Вернут ли деньги за такси, если я потерял чек?", "expected": "без чеков"},
    {"question": "Куда писать, если сломался ноутбук?", "expected": "#it-help"},
    {"question": "Сгорают ли неиспользованные дни отпуска?", "expected": "не более 10 дней"},
    {"question": "Что делать, если пришло странное письмо?", "expected": "#security"},
    {"question": "Можно ли загрузить код в ChatGPT?", "expected": "чат-боты"},
    {"question": "Можно ли не работать без справки, если плохо себя чувствую?", "expected": "дней здоровья"},
    {"question": "Компания оплатит починку разбитого экрана?", "expected": "Ремонт оборудования"},
    {"question": "До какого числа сдавать траты?", "expected": "до 5 числа"},
    {"question": "Кто поможет новичку освоиться?", "expected": "бадди"},
    {"question": "Я простудился, кого и когда предупредить?", "expected": "до 11:00"},
    {"question": "Когда дают медицинскую страховку?", "expected": "ДМС"},
    {"question": "Сколько людей в компании?", "expected": "47 человек"},
    {"question": "Сколько всего дней можно отдыхать в году?", "expected": "дней здоровья"},
    {"question": "А если больше трёх лет работаю?", "expected": "3 дополнительных"},
    {"question": "Есть ли сокращенные рабочие дни?", "expected": "Летом"},
    {"question": "Есть ли бесплатные обеды?", "expected": "По средам в офисе"},
    {"question": "Когда новогодний корпоратив?", "expected": "в последнюю пятницу декабря"},
    {"question": "В офисе живет кот?", "expected": "Байт"},
]

NO_ANSWER = [
    "Какая зарплата у джуниора?",
    "Есть ли парковка у офиса?",
    "Можно ли приводить собаку в офис?",
    "Оплачивает ли компания такси до дома после 22:00?",
    "Сколько дней отпуска можно взять на испытательном сроке?",
    "Кто уборщик в офисе?",
    "Компенсируют ли проезд от дома до офиса?",
]


def find_rank(found: Found, expected: str) -> int | None:
    for i, (score, text) in enumerate(found, start=1):
        if expected.lower() in text.lower():
            return i
    return None


def evaluate(eval_set: list[dict[str, str]], chunks: list[str], chunk_vectors: np.ndarray, k: int,
             show_misses: bool = False) -> tuple[float, float]:
    hits = 0
    rr_sum = 0.0
    for item in eval_set:
        found = search(item["question"], chunks, chunk_vectors, k)
        rank = find_rank(found, item["expected"])
        if rank is None:
            if show_misses:
                print(f" промах: {item["question"]}")
        else:
            hits += 1
            rr_sum += 1 / rank
    return hits / len(eval_set), rr_sum / len(eval_set)


def check_refusals(questions: list[str], chunks: list[str], chunk_vectors: np.ndarray) -> float:
    refusals = 0
    for question in questions:
        answer, found = ask(question, chunks, chunk_vectors)
        print(f"Вопрос: {question}\nОтвет: {answer}")
        if REFUSAL in answer:
            print("получили отказ")
            refusals += 1
    return refusals / len(questions)


STRATEGIES: dict[str, Chunker] = {
    "абзацы": split_into_chunks,
    "предложения": chunk_by_sentences,
    "окно 2": lambda text: chunk_by_sentences(text, n=2),
    "окно 2 перекрытие": lambda text: chunk_by_sentences(text, n=2, step=1),
    "окно 3": lambda text: chunk_by_sentences(text, n=3),
    "разделы": chunk_by_sections,
}


def compare_strategies(strategies: dict[str, Chunker]) -> None:
    for name, chunker in strategies.items():
        start = time.perf_counter()
        chunks, chunk_vectors = build_index(chunker=chunker)
        index_time = time.perf_counter() - start
        avg_len = sum(len(chunk) for chunk in chunks) / len(chunks)

        hit1, _ = evaluate(EVAL_SET, chunks, chunk_vectors, k=1, show_misses=True)
        hit3, mrr = evaluate(EVAL_SET, chunks, chunk_vectors, k=3, show_misses=True)
        print(f"{name:<18}  кусков: {len(chunks):>3}   ср. длина: {avg_len:>4.0f}   "
              f"hit@1 = {hit1:.2f}   hit@3 = {hit3:.2f}   MRR = {mrr:.2f}   индекс: {index_time:.2f} с")


if __name__ == "__main__":
    total_start = time.perf_counter()
    chunks, chunk_vectors = build_index()

    print("=== Поиск ===")
    for k in (1, 3, 5):
        hit_rate, mrr = evaluate(EVAL_SET, chunks, chunk_vectors, k, show_misses=(k == 3))
        print(f"k={k}:  hit@k = {hit_rate:.2f}   MRR = {mrr:.2f}")

    print("\n=== Сравнение нарезок ===")
    start = time.perf_counter()
    compare_strategies(STRATEGIES)
    print(f"Сравнение нарезок: {time.perf_counter() - start:.1f} с")

    print("\n=== Вопросы без ответа ===")
    start = time.perf_counter()
    refusal_rate = check_refusals(NO_ANSWER, chunks, chunk_vectors)
    print(f"\nОтказов: {refusal_rate:.2f}   время: {time.perf_counter() - start:.1f} с")
    print(f"Всего: {time.perf_counter() - total_start:.1f} с")
