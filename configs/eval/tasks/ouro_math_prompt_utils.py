"""Pinned MATH500 prompt and scoring helpers."""

import re


def doc_to_instruct(doc):
    return (
        "Below is a math problem. Reason step by step, and put your final answer "
        "within \\boxed{}.\n\n"
        f"Problem: {doc['problem']}\n\nSolution:"
    )


def process_results(doc, results):
    from lm_eval.tasks.minerva_math.utils import process_results as score

    return {"math_verify": score(doc, results)["math_verify"]}


def truncate_new_problem(text):
    return re.split(r"(?m)^[ \t]*(?:Question|Problem):", text, maxsplit=1)[0]


def process_results_ouro26(doc, results):
    return process_results(doc, [truncate_new_problem(text) for text in results])
