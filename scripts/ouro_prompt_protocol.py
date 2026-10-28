"""Ouro GSM8K prompts and answer scoring shared by evaluation and calibration."""

import re

FEWSHOT_PREFIX = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

"""


def doc_to_text(doc):
    return FEWSHOT_PREFIX + f"Q: {doc['question']}\nA:"


def last_number(text):
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else None


def process_results(doc, results):
    text = results[0]
    gold = doc["answer"].split("####")[-1].strip().replace(",", "")
    strict = re.search(r"The answer is (\-?[0-9\.\,]+).", text)
    prediction = strict.group(1).replace(",", "").rstrip(".") if strict else None
    return {
        "strict_match": prediction == gold,
        "flexible_match": last_number(text) == gold,
    }
