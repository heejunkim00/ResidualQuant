"""EvalPlus prompts shared by evaluation and calibration."""

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)
MAGIC_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"


PROMPT_TEMPLATE = (
    INSTRUCTION_PREFIX + "\n```\n{problem}\n```\n" + RESPONSE_PREFIX + "\n```python\n"
)

STOP_STRINGS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\nmain(",
    "\nprint(",
    "\n```\n",
]

MAX_TOKENS = {"humaneval": 1024, "mbpp": 2048}


def build_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem.strip())


def build_chat_prompt(problem: str, tokenizer) -> str:
    """Render an EvalPlus chat prompt ending inside the Python code fence."""
    task = f"{INSTRUCTION_PREFIX}\n```\n{problem.strip()}\n```\n"
    response = f"{RESPONSE_PREFIX}\n```python\n{MAGIC_SPLITTER}\n```\n"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": task}, {"role": "assistant", "content": response}],
        tokenize=False,
    )
    return rendered.split(MAGIC_SPLITTER)[0]
