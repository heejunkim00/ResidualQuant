"""Build pinned Ouro and Huginn calibration prompts."""

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from code_prompt_protocol import build_chat_prompt, build_prompt


def build_dataset(model_key, domain):
    if domain not in ("gsm8k", "mbpp"):
        raise ValueError("Calibration dataset must be gsm8k or mbpp")
    if model_key not in ("ouro", "ouro26", "huginn"):
        raise ValueError(f"Unsupported calibration model: {model_key}")
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads(
        (root / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
    )
    model = protocol["models"][model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        model["name"],
        revision=model["revision"],
        trust_remote_code=True,
        local_files_only=True,
    )
    extra = {}
    if domain == "mbpp":
        from evalplus.data import get_mbpp_plus

        excluded = {int(key.split("/")[-1]) for key in get_mbpp_plus()}
        dataset = load_dataset("google-research-datasets/mbpp", "full", split="train")
        rows = {int(row["task_id"]): row for row in dataset}
        sample_ids = sorted(set(rows) - excluded)
        assert len(dataset) == 374 and len(sample_ids) == 266
        extra = dict(
            excluded_evaluation_ids=sorted(excluded), available=len(sample_ids)
        )
        source = root / "scripts/code_prompt_protocol.py"
        name = "google-research-datasets/mbpp/full"

        def render(row):
            problem = row["text"] + "\n\n" + "\n".join(row["test_list"])
            prompt = (
                build_chat_prompt(problem, tokenizer)
                if model_key == "huginn"
                else build_prompt(problem)
            )
            return prompt + row["code"] + "\n```\n"

    elif model_key == "huginn":
        import lm_eval
        from lm_eval.tasks import get_task_dict

        task = get_task_dict(["gsm8k_cot"])["gsm8k_cot"]
        task.set_fewshot_seed(0)
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        rows = dict(enumerate(dataset))
        demos = task.get_config("fewshot_config").samples
        excluded = {row["question"].strip() for row in demos}
        sample_ids = [
            index
            for index, row in rows.items()
            if row["question"].strip() not in excluded
        ]
        source = Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k-cot.yaml"
        name = "openai/gsm8k/main"

        def template(messages, **kwargs):
            return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)

        def render(row):
            return (
                task.fewshot_context(
                    row,
                    num_fewshot=8,
                    apply_chat_template=True,
                    fewshot_as_multiturn=True,
                    chat_template=template,
                    system_instruction=(
                        "You are a helpful assistant that can assist users "
                        "with mathematical reasoning."
                    ),
                )
                + row["answer"]
            )
    else:
        from ouro_prompt_protocol import FEWSHOT_PREFIX

        dataset = load_dataset("openai/gsm8k", "main", split="train")
        rows = dict(enumerate(dataset))
        sample_ids = [
            index
            for index, row in rows.items()
            if row["question"].strip() not in FEWSHOT_PREFIX
        ]
        source = root / "scripts/ouro_prompt_protocol.py"
        name = "openai/gsm8k/main"

        def render(row):
            return FEWSHOT_PREFIX + f"Q: {row['question']}\nA:" + row["answer"]

    random.Random(0).shuffle(sample_ids)
    sample_ids = sample_ids[:256]
    if len(sample_ids) != 256:
        raise ValueError("Calibration requires 256 training samples")
    return dict(
        model="huginn" if model_key == "huginn" else "ouro",
        dataset=name,
        split="train",
        domain="code" if model_key == "huginn" and domain == "mbpp" else domain,
        seed=0,
        partitions=dict(fit=sample_ids[:128], holdout=sample_ids[128:]),
        samples=[dict(id=index, text=render(rows[index])) for index in sample_ids],
        tokenizer_revision=model["revision"],
        teacher_forced=True,
        prompt_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        **extra,
    ), tokenizer
