"""Huginn MATH500 prompt."""


def doc_to_boxed_final(doc: dict) -> str:
    return (
        f"Problem:\n{doc['problem']}\n\nSolution:\n"
        "Reason carefully and show the essential steps. Put the final answer "
        "in \\boxed{} and also state `Final Answer: The final answer is "
        "$ANSWER$. I hope it is correct.`"
    )
