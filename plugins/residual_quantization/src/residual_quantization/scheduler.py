"""Keep full-prompt scheduling for completed-group quantization."""

from vllm.v1.core.sched.scheduler import Scheduler

__all__ = ("FullPromptScheduler",)


class FullPromptScheduler(Scheduler):
    """Admit full prompts with decode requests; disable partial prefills."""

    def __init__(self, vllm_config, *args, **kwargs):
        config = vllm_config.scheduler_config
        if config.max_num_batched_tokens < config.max_model_len:
            raise ValueError("Full prompts require a budget of at least max_model_len")
        if config.long_prefill_token_threshold:
            raise ValueError("Packed KV does not support partial long prefills")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Packed KV does not support cached prompt prefixes")
        config.enable_chunked_prefill = False
        config.chunked_prefill_enabled = False
        super().__init__(vllm_config, *args, **kwargs)
