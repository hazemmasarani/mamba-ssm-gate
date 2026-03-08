# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MAMBA model."""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.distributed as dist

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationMixin
from transformers.integrations import lazy_load_kernel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    auto_docstring,
    logging,
)
from transformers.utils.import_utils import (
    is_mambapy_available,
    is_torchdynamo_compiling,
)

from transformers.models.mamba.configuration_mamba import MambaConfig


logger = logging.get_logger(__name__)

if is_mambapy_available():
    from mambapy.pscan import pscan
else:
    pscan = None


class MambaCache_Gate:
    """
    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PreTrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if
            a smaller batch size is used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.

    Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, MambaForCausalLM_Gate, MambaCache_Gate

        >>> model = MambaForCausalLM_Gate.from_pretrained("state-spaces/mamba-130m-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

        >>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_params = MambaCache_Gate(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
        >>> cache_position = torch.arange(len(inputs["input_ids"][0]), device=model.device)  # sequence length
        >>> outputs = model(**inputs, cache_params=cache_params, cache_position=cache_position, use_cache=True)
        >>> outputs.cache_params
        ```
    """

    is_compileable = True

    # TODO (joao): add layer_device_map arg and update code in `generate` accordingly
    def __init__(
        self,
        config: PreTrainedConfig,
        max_batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
    ):
        self.max_batch_size = max_batch_size
        self._dtype = dtype
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel

        self.conv_states: list[torch.Tensor] = []
        self.ssm_states: list[torch.Tensor] = []
        device = torch.device(device) if device is not None else None
        for _ in range(config.num_hidden_layers):
            conv_state: torch.Tensor = torch.zeros(
                self.max_batch_size,
                self.intermediate_size,
                self.conv_kernel_size,
                device=device,
                dtype=self._dtype,
            )
            ssm_state: torch.Tensor = torch.zeros(
                self.max_batch_size,
                self.intermediate_size,
                self.ssm_state_size,
                device=device,
                dtype=self._dtype,
            )

            torch._dynamo.mark_static_address(conv_state)
            torch._dynamo.mark_static_address(ssm_state)
            self.conv_states.append(conv_state)
            self.ssm_states.append(ssm_state)

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        # This `if` blocks is only reached in multigpu and if `layer_device_map` is not passed. It is used
        # when the cache is initialized in the forward pass (e.g. Mamba)
        if self.conv_states[layer_idx].device != new_conv_state.device:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].to(new_conv_state.device)

        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(device=conv_state.device, dtype=conv_state.dtype)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx].zero_()
        self.ssm_states[layer_idx] += new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]

    def reset(self):
        for layer_idx in range(len(self.conv_states)):
            # In-place ops prevent breaking the static address
            self.conv_states[layer_idx].zero_()
            self.ssm_states[layer_idx].zero_()


class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.ssm_state_size = config.state_size
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act = ACT2FN[config.hidden_act]

        self.use_mambapy = config.use_mambapy

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        # porjection of the output hidden states
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

    # fmt: off
    def slow_forward(self, input_states):
        batch_size, seq_len, _ = input_states.shape
        # 1. Gated MLP's linear projection
        gate = self.in_proj(input_states).transpose(1, 2)  
        gate = self.act(gate)
        gate = gate.contiguous()
        scan_output = torch.zeros(batch_size, self.intermediate_size, seq_len).to(gate.device)
        if dist.is_initialized():
            dist.broadcast(gate, src=0)
            dist.broadcast(scan_output, src=1)
        else:
            print("Not distributed Error")
        scan_output = scan_output * gate

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states
    ):
        return self.slow_forward(hidden_states)


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


class MambaBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MambaPreTrainedModel_Gate(PreTrainedModel):
    config: MambaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["MambaBlock", "MambaMixer"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights."""
        std = self.config.initializer_range
        if isinstance(module, MambaMixer):

            if self.config.rescale_prenorm_residual:
                # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
                #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
                #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
                #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
                #
                # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                p = module.out_proj.weight
                p /= math.sqrt(self.config.num_hidden_layers)

        if isinstance(module, nn.Linear):
            init.normal_(module.weight, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, MambaRMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, std=std)


@dataclass
class MambaOutput(ModelOutput):
    r"""
    cache_params (`MambaCache_Gate`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.

        Includes both the State space model state matrices after the selective scan, and the Convolutional states
    """

    last_hidden_state: torch.FloatTensor | None = None
    cache_params: MambaCache_Gate | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


@dataclass
class MambaCausalLMOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    cache_params (`MambaCache_Gate`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.

        Includes both the State space model state matrices after the selective scan, and the Convolutional states
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    cache_params: MambaCache_Gate | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


class MambaModel_Gate(MambaPreTrainedModel_Gate):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.LongTensor | None = None,
        cache_params: MambaCache_Gate | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | MambaOutput:
        r"""
        cache_params (`MambaCache_Gate`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = MambaCache_Gate(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            hidden_states = mixer_block(
                hidden_states
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class MambaForCausalLM_Gate(MambaPreTrainedModel_Gate, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "backbone.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.backbone = MambaModel_Gate(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: dict[str, Any], num_new_tokens: int = 1, **kwargs
    ) -> dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: MambaCache_Gate | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ):
        # Overwritten -- has custom cache class `MambaCache_Gate`
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if use_cache and cache_params is None:
            # we initialize the `cache_position` to full size of `conv_states` at prefill stage
            # considering padding will be applied when input length is shorter, and truncation
            # will be applied when it is longer, so it will be equivalent to always have it match
            # the length of `cache_params.conv_states`, which is `config.conv_kernel`
            model_inputs["cache_position"] = torch.arange(0, self.backbone.config.conv_kernel, device=input_ids.device)
            if inputs_embeds is not None:
                max_batch_size = inputs_embeds.size(0)
            else:
                max_batch_size = input_ids.size(0)
            model_inputs["cache_params"] = MambaCache_Gate(
                self.backbone.config, max_batch_size, device=self.device, dtype=self.dtype
            )
        elif use_cache and cache_position[0] > 0:
            model_inputs["attention_mask"] = None

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_params: MambaCache_Gate | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,  # for now we need this for generation
    ) -> tuple | MambaCausalLMOutput:
        r"""
        cache_params (`MambaCache_Gate`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

        hidden_states = mamba_outputs[0]
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :].to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )


__all__ = ["MambaForCausalLM_Gate", "MambaModel_Gate", "MambaPreTrainedModel_Gate", "MambaCache_Gate"]
