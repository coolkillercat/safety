"""Template C: length-matched irrelevant text — "a paragraph about an unrelated topic" of the false
rationale's length. Written before any template-C result existed and frozen with template C.

Sentences about everyday topics (weather, gardening, cooking, travel, a small town), none mentioning options,
letters, answers, numbers that could be read as a calculation, or any academic subject. Per item a seeded shuffle of
the sentences; sentences are appended until the word count reaches the decept text's word count (the last
sentence may overshoot by less than one sentence).
"""
from __future__ import annotations

import random

SENTENCES = [
    "The morning fog usually lifts by the time the bakery on the corner opens its doors.",
    "Most gardeners agree that tomatoes do best in a sunny spot sheltered from strong wind.",
    "The old railway station has been turned into a small market that sells bread and flowers on weekends.",
    "A good soup often tastes better on the second day, once the flavours have had time to settle.",
    "Along the river path there is a bench where people stop to watch the ducks in the late afternoon.",
    "Autumn brings shorter days, and the maple trees along the avenue turn a deep shade of red.",
    "Many travellers find that packing light makes the first day of a trip far more relaxing.",
    "The library in the town square keeps a shelf of donated cookbooks near the front window.",
    "When it rains for several days in a row, the garden snails come out in surprising numbers.",
    "Fresh herbs such as basil and parsley lose their scent quickly once they have been cut.",
    "On clear evenings the hill behind the school is a popular place to watch the sunset.",
    "The local bakery is known for a rye loaf that has been made the same way for many years.",
    "Some people like to walk to work in spring, when the hedges are full of small white blossoms.",
    "A well-made pot of tea is usually left to stand for a few minutes before it is poured.",
    "The ferry across the lake runs every hour in summer but only twice a day in winter.",
    "Children in the neighbourhood often gather in the park after school to play until dusk.",
    "Window boxes with bright geraniums make even a narrow street look cheerful in July.",
    "The weekly farmers' market sells apples, honey and cheese from the surrounding villages.",
    "A short walk after dinner is a habit that many families keep throughout the year.",
    "In the colder months the café by the harbour serves hot chocolate with a slice of cake.",
    "The town's annual summer fair has a parade, a brass band and stalls selling lemonade.",
    "Old stone walls along the country lanes are often covered in moss and small ferns.",
    "Planting bulbs in late autumn means the garden fills with colour early the next spring.",
    "The bus to the coast leaves from the square and passes several quiet fishing villages.",
]


def irrelevant_text(item_id: str, n_words: int) -> str:
    rng = random.Random(f"p15-irrelevant-{item_id}")
    pool = SENTENCES[:]
    out, words = [], 0
    while words < max(n_words, 1):
        if not pool:
            pool = SENTENCES[:]
        rng.shuffle(pool)
        s = pool.pop()
        out.append(s)
        words += len(s.split())
    return " ".join(out)
