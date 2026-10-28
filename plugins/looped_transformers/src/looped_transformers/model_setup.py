"""Expose recurrent model structure on its Hugging Face config."""

__all__ = ("prepare_model_config",)


def prepare_model_config(config):
    architectures = config.architectures or ()
    if "OuroForCausalLM" in architectures:
        config.loop_model = "ouro"
        config.n_layers_in_recurrent_block = config.num_hidden_layers
        config.mean_recurrence = int(getattr(config, "total_ut_steps", 4))
        config.n_layers_in_prelude = 0
        config.n_layers_in_coda = 0
    elif "RavenForCausalLM" in architectures:
        config.loop_model = "huginn"
    else:
        raise ValueError(f"Unsupported server model architecture: {architectures}")
    return config
