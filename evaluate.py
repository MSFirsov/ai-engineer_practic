import sys
import time

import numpy as np

from rag import (
    REFUSAL,
    Chunker,
    Found,
    Searcher,
    ask,
    build_index,
    cache_key,
    chroma_searcher,
    chunk_by_sections,
    chunk_by_sentences,
    get_collection,
    hybrid_search,
    keyword_search,
    rewrite_query,
    search,
    split_into_chunks,
    rerank,
    get_reranker,
)

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

EXACT_SET = [
    {"question": "@tundra_hr_bot", "expected": "@tundra_hr_bot"},
    {"question": "Что за бот tundra_hr_bot?", "expected": "@tundra_hr_bot"},
    {"question": "#it-help", "expected": "напишите в канал #it-help"},
    {"question": "Для чего канал #security?", "expected": "сообщайте в канал #security"},
    {"question": "Кто такой Сорокин?", "expected": "системный администратор"},
    {"question": "Кто такой Огнев?", "expected": "Техническим директором"},
    {"question": "Кто такая Лосева?", "expected": "HR-менеджеру Дарье"},
    {"question": "ThinkPad", "expected": "ThinkPad"},
    {"question": "Что такое ДМС?", "expected": "ДМС"},
    {"question": "Что такое Онего?", "expected": "Онего"},
    {"question": "Что в 15:00?", "expected": "15:00"},
    {"question": "Что происходит в 11:15?", "expected": "11:15"},
    {"question": "хакатон", "expected": "хакатон"},
    {"question": "Что такое бадди?", "expected": "бадди"},
]

FOLLOWUP_SET = [
    {"history": [("Сколько компания компенсирует на обучение?", "До 60 000 рублей в год.")],
     "question": "А на спорт?", "expected": "3 000 рублей"},
    {"history": [("Какой ноутбук можно выбрать?", "MacBook Pro или Lenovo ThinkPad.")],
     "question": "А как часто их меняют?", "expected": "раз в 3 года"},
    {"history": [("Можно ли работать из дома?", "Да, до 3 дней в неделю без согласований.")],
     "question": "А полностью?", "expected": "Полностью удалённый"},
    {"history": [("Когда летний выезд?", "Каждый июль, на Онежское озеро на два дня.")],
     "question": "А новогодний?", "expected": "последнюю пятницу декабря"},
    {"history": [("Сколько длится испытательный срок?", "3 месяца.")],
     "question": "Что дают после него?", "expected": "ДМС"},
    {"history": [("Сколько дней здоровья в году?", "5 дней.")],
     "question": "Они переносятся на следующий год?", "expected": "Дни здоровья не переносятся"},
    {"history": [("Кто генеральный директор?", "Ирина Валенкова.")],
     "question": "А технический?", "expected": "Павел Огнев"},
    {"history": [("Как подать заявление на отпуск?", "Через бота @tundra_hr_bot не позднее чем за 14 дней.")],
     "question": "А кто его согласует?", "expected": "согласует непосредственный"},
    {"history": [("Как вернуть деньги за такси?", "Заполнить таблицу расходов до 5 числа и приложить чеки.")],
     "question": "А когда их вернут?", "expected": "ближайшей зарплатой"},
    {"history": [("Сколько дней отпуска в году?", "28 календарных дней.")],
     "question": "А в офисе есть кот?", "expected": "Байт"},
    {"history": [("Где находится офис?", "В бизнес-центре «Онего», на 4 этаже.")],
     "question": "А кто там живёт?", "expected": "Байт"},
    {"history": [("Куда писать, если сломался ноутбук?", "В канал #it-help.")],
     "question": "А что с ним делать при увольнении?", "expected": "вернуть в течение 5"},
    {"history": [("Сколько дней здоровья в году?", "5 дней."),
                 ("А если болезнь длится дольше?",
                  "Нужно оформить больничный лист и отправить его номер Дарье Лосевой.")],
     "question": "А их можно перенести на следующий год?", "expected": "Дни здоровья не переносятся"},
    {"history": [("Сколько компания компенсирует на обучение?", "До 60 000 рублей в год.")],
     "question": "Во сколько стендап?", "expected": "11:15"},
]


def find_rank(found: Found, expected: str) -> int | None:
    """Ищет, на каком месте в выдаче стоит кусок с ожидаемой подстрокой.

    Сравнение идёт без учёта регистра.

    Args:
        found: Результат search: пары (близость, кусок).
        expected: Подстрока, которая должна быть в правильном куске.

    Returns:
        Место первого подходящего куска, считая с 1, или None, если его нет в выдаче.
    """
    for i, (score, text) in enumerate(found, start=1):
        if expected.lower() in text.lower():
            return i
    return None


def evaluate(eval_set: list[dict[str, str]], chunks: list[str], chunk_vectors: np.ndarray, k: int,
             show_misses: bool = False, searcher: Searcher = search) -> tuple[float, float]:
    """Считает качество поиска на наборе вопросов.

    Args:
        eval_set: Словари с ключами "question" (по нему идёт поиск) и "expected"
            (подстрока правильного куска).
        chunks: Куски документа.
        chunk_vectors: Матрица эмбеддингов кусков.
        k: Сколько кусков брать из выдачи.
        show_misses: Печатать ли вопросы, для которых нужный кусок не попал в top-k.
        searcher: Функция поиска. По умолчанию поиск по смыслу (search).

    Returns:
        Пара (hit@k, MRR): доля вопросов с нужным куском в top-k
        и среднее значение 1/место, где промах считается за 0.
    """
    hits = 0
    rr_sum = 0.0
    for item in eval_set:
        found = searcher(item["question"], chunks, chunk_vectors, k)
        rank = find_rank(found, item["expected"])
        if rank is None:
            if show_misses:
                print(f" промах: {item["question"]}")
        else:
            hits += 1
            rr_sum += 1 / rank
    return hits / len(eval_set), rr_sum / len(eval_set)


def check_refusals(questions: list[str], chunks: list[str], chunk_vectors: np.ndarray) -> float:
    """Проверяет, отказывается ли модель отвечать на вопросы, которых нет в документе.

    Для каждого вопроса вызывает ask и печатает вопрос и ответ. Отказом считается
    ответ, в котором есть фраза REFUSAL.

    Args:
        questions: Вопросы без ответа в документе.
        chunks: Куски документа.
        chunk_vectors: Матрица эмбеддингов кусков.

    Returns:
        Доля вопросов, на которые модель ответила отказом.
    """
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
    """Сравнивает стратегии нарезки на EVAL_SET.

    Для каждой стратегии строит индекс и печатает строку: число кусков, средняя
    длина куска, hit@1, hit@3, MRR и время построения индекса. Перед строкой
    печатаются промахи.

    Args:
        strategies: Словарь «название стратегии → функция нарезки».
    """
    for name, chunker in strategies.items():
        start = time.perf_counter()
        chunks, chunk_vectors = build_index(chunker=chunker)
        index_time = time.perf_counter() - start
        avg_len = sum(len(chunk) for chunk in chunks) / len(chunks)

        hit1, _ = evaluate(EVAL_SET, chunks, chunk_vectors, k=1, show_misses=True)
        hit3, mrr = evaluate(EVAL_SET, chunks, chunk_vectors, k=3, show_misses=True)
        print(f"{name:<18}  кусков: {len(chunks):>3}   ср. длина: {avg_len:>4.0f}   "
              f"hit@1 = {hit1:.2f}   hit@3 = {hit3:.2f}   MRR = {mrr:.2f}   индекс: {index_time:.2f} с")


SEARCHERS: dict[str, Searcher] = {
    "по смыслу": search,
    "по словам": lambda query, chunks, chunk_vectors, k: keyword_search(query, chunks, k),
    "гибрид": hybrid_search,
}


def compare_searchers(searchers: dict[str, Searcher], sets: dict[str, list[dict[str, str]]],
                      chunks: list[str], chunk_vectors: np.ndarray) -> None:
    """Сравнивает способы поиска на нескольких наборах вопросов.

    Для каждого набора и каждого способа печатает строку: hit@1, hit@3 и MRR.

    Args:
        searchers: Словарь «название способа → функция поиска».
        sets: Словарь «название набора → вопросы с ожидаемой подстрокой».
        chunks: Куски документа.
        chunk_vectors: Матрица эмбеддингов кусков.
    """
    for set_name, eval_set in sets.items():
        print(f"{set_name} ({len(eval_set)} вопросов)")
        for name, searcher in searchers.items():
            hit1, _ = evaluate(eval_set, chunks, chunk_vectors, k=1, searcher=searcher)
            hit3, mrr = evaluate(eval_set, chunks, chunk_vectors, k=3, searcher=searcher)
            print(f"  {name:<10} hit@1 = {hit1:.2f}   hit@3 = {hit3:.2f}   MRR = {mrr:.2f}")


def make_query(item: dict, mode: str) -> str:
    """Строит строку для поиска по уточняющему вопросу одним из трёх подходов.

    Args:
        item: Элемент FOLLOWUP_SET с ключами "history", "question" и "expected".
        mode: Подход: "как есть" — сам вопрос, "склейка" — предыдущий вопрос
            пользователя плюс текущий, "переписывание" — результат rewrite_query.

    Returns:
        Строка, по которой будет идти поиск.
    """
    if mode == "склейка":
        return f"{item['history'][-1][0]} {item['question']}"
    if mode == "переписывание":
        return rewrite_query(item["question"], item["history"])
    return item["question"]


def compare_followups(followup_set: list[dict], chunks: list[str], chunk_vectors: np.ndarray) -> None:
    """Сравнивает три подхода к уточняющим вопросам.

    Для каждого подхода один раз строит набор запросов через make_query и печатает
    строку: hit@1, hit@3, MRR и время построения запросов.

    Args:
        followup_set: Уточняющие вопросы с историей и ожидаемой подстрокой.
        chunks: Куски документа.
        chunk_vectors: Матрица эмбеддингов кусков.
    """
    modes = ["как есть", "склейка", "переписывание"]
    for mode in modes:
        start = time.perf_counter()
        eval_set = []
        for item in followup_set:
            eval_set.append({"question": make_query(item, mode), "expected": item["expected"]})
        query_time = time.perf_counter() - start

        hit1, _ = evaluate(eval_set, chunks, chunk_vectors, k=1)
        hit3, mrr = evaluate(eval_set, chunks, chunk_vectors, k=3, show_misses=True)
        print(f"{mode:<15} hit@1 = {hit1:.2f}   hit@3 = {hit3:.2f}   MRR = {mrr:.2f}   запросы: {query_time:.1f} с")


def compare_chroma() -> None:
    """Сравнивает NumPy и сохранённую Chroma на одинаковых кусках.

    Читает справочник для базы сравнения NumPy, но не изменяет Chroma.
    Проверяет совпадение идентификаторов до вычисления метрик.

    Raises:
        ValueError: В базе другой набор кусков или база пуста.
    """
    chunks, chunk_vectors = build_index()
    expected_ids = {cache_key(chunk) for chunk in chunks}
    actual_ids = set(get_collection().get(include=[])["ids"])
    if actual_ids != expected_ids:
        raise ValueError("Набор кусков в Chroma отличается от справочника. Проверь индексацию.")
    compare_searchers(
        {"NumPy": search, "Chroma": chroma_searcher},
        {"обычные вопросы": EVAL_SET, "точные вопросы": EXACT_SET},
        chunks,
        chunk_vectors,
    )


def compare_reranking(eval_set: list[dict[str, str]], chunks: list[str],
                      chunk_vectors: np.ndarray, pool_size: int = 20) -> None:
    """Сравнивает исходную выдачу и реранкинг одних и тех же кандидатов.

    Args:
        eval_set: Непустой набор вопросов и ожидаемых подстрок.
        chunks: Куски справочника.
        chunk_vectors: Их векторы для поиска NumPy.
        pool_size: Число кандидатов первого этапа, не меньше трёх.

    Raises:
        ValueError: Набор пуст или число кандидатов меньше трёх.
    """
    if not eval_set or pool_size < 3:
        raise ValueError("Нужны вопросы и pool_size >= 3")
    totals = {"до": [0, 0, 0.0], "после": [0, 0, 0.0]}
    pool_hits = 0
    search_time = 0.0
    rerank_time = 0.0
    for item in eval_set:
        query, expected = item["question"], item["expected"]
        start = time.perf_counter()
        candidates = search(query, chunks, chunk_vectors, pool_size)
        search_time += time.perf_counter() - start
        if find_rank(candidates, expected) is not None:
            pool_hits += 1
        before = candidates[:3]
        start = time.perf_counter()
        after = rerank(query, candidates, k=3)
        rerank_time += time.perf_counter() - start
        for name, found in (("до", before), ("после", after)):
            rank = find_rank(found, expected)
            if rank is not None:
                totals[name][0] += int(rank == 1)
                totals[name][1] += 1
                totals[name][2] += 1 / rank
        before_rank = find_rank(before, expected)
        after_rank = find_rank(after, expected)
        if before_rank != after_rank:
            print(f"  {query}: {before_rank} -> {after_rank}")
    count = len(eval_set)
    print(f"Кандидаты: hit@{pool_size} = {pool_hits / count:.2f}")
    for name, (hit1, hit3, rr) in totals.items():
        print(f"{name}: hit@1 = {hit1 / count:.2f}  hit@3 = {hit3 / count:.2f}  MRR = {rr / count:.2f}")
    print(f"Поиск: {search_time:.2f} с; реранкинг: {rerank_time:.2f} с; вопросов: {count}")


if __name__ == "__main__":
    if "--rerank" in sys.argv:
        if "--fast" in sys.argv or "--chroma" in sys.argv:
            raise SystemExit("Запускай --rerank отдельно от --fast и --chroma")
        start = time.perf_counter()
        get_reranker()
        print(f"Загрузка реранкера: {time.perf_counter() - start:.2f} с")
        chunks, chunk_vectors = build_index()
        for name, dataset in (("обычные", EVAL_SET), ("точные", EXACT_SET)):
            print(name)
            compare_reranking(dataset, chunks, chunk_vectors)
        raise SystemExit(0)
    if "--chroma" in sys.argv:
        compare_chroma()
        raise SystemExit(0)
    fast = "--fast" in sys.argv
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

    print("\n=== Сравнение поисков ===")
    compare_searchers(SEARCHERS, {"обычные вопросы": EVAL_SET, "точные запросы": EXACT_SET}, chunks, chunk_vectors)

    if not fast:
        print("\n=== Уточняющие вопросы ===")
        start = time.perf_counter()
        compare_followups(FOLLOWUP_SET, chunks, chunk_vectors)
        print(f"Уточняющие вопросы: {time.perf_counter() - start:.1f} с")

        print("\n=== Вопросы без ответа ===")
        start = time.perf_counter()
        refusal_rate = check_refusals(NO_ANSWER, chunks, chunk_vectors)
        print(f"\nОтказов: {refusal_rate:.2f}   время: {time.perf_counter() - start:.1f} с")
    print(f"Всего: {time.perf_counter() - total_start:.1f} с")
