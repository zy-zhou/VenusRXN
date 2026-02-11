import math
import functools

import einops
import torch
import torch.nn.functional as F
from torch import nn
from attr import dataclass
from huggingface_hub import snapshot_download
from pathlib import Path

from esm.layers.rotary import RotaryEmbedding
from esm.layers.regression_head import RegressionHead
from esm.sdk.api import ESMCInferenceClient
from esm.tokenization import EsmSequenceTokenizer, get_esmc_model_tokenizers
from esm.utils.constants.models import ESMC_300M, ESMC_600M


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
        is_cross_attn: bool = False,
        d_cross_attn: int | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.is_cross_attn = is_cross_attn

        if is_cross_attn:
            d_cross_attn = d_cross_attn or d_model
            self.layernorm_q = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model, bias=bias)
            )
            self.kv = nn.Linear(d_cross_attn, d_model * 2, bias=bias) # assume kv uses post layernorm
        else:
            self.layernorm_qkv = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
            )
            self.rotary = RotaryEmbedding(d_model // n_heads)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()


    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, seq_id, encoder_x=None, encoder_attn_mask=None):
        attn_mask = None

        if self.is_cross_attn:
            query_BLD = self.layernorm_q(x)
            kv_BLD2 = self.kv(encoder_x)
            key_BLD, value_BLD = torch.chunk(kv_BLD2, 2, dim=-1)

            if encoder_attn_mask is not None:
                if encoder_attn_mask.dtype is not torch.bool:
                    encoder_attn_mask = encoder_attn_mask != 0
                attn_mask = encoder_attn_mask[:, None, None, :]
        
        else:
            qkv_BLD3 = self.layernorm_qkv(x)
            query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)

            if seq_id is not None:
                # Where True, enable participation in attention.
                attn_mask = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
                attn_mask = attn_mask.unsqueeze(1)

        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )

        if not self.is_cross_attn:
            query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        # Shortcut, if we don't use attention biases then torch
        # will autoselect flashattention as the implementation
        context_BHLD = F.scaled_dot_product_attention(
            query_BHLD, key_BHLD, value_BHLD, attn_mask
        )

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")

        return self.out_proj(context_BLD)


class UnifiedTransformerBlock(nn.Module):
    """
    A unified transformer block that can optionally incorporate cross-attention.

    This class defines a transformer block that can be configured to use cross-attention
    alongside the standard multi-head self-attention mechanism. It is designed to be a flexible
    component of transformer-based models, allowing for the integration of different modalities
    of information.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    add_cross_attn : bool, optional
        Whether to add cross-attention to the transformer block. Default is False.
    d_cross_attn : int, optional
        The dimensionality of the encoder output for the cross-attention. Default is `d_model`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
        add_cross_attn: bool = False,
        d_cross_attn: int | None = None
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model, n_heads, bias, qk_layernorm=qk_layernorm
        )

        self.add_cross_attn = add_cross_attn
        if add_cross_attn:
            self.cross_attn = MultiHeadAttention(
                d_model, n_heads, bias, qk_layernorm=qk_layernorm,
                is_cross_attn=True, d_cross_attn=d_cross_attn
            )
        
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor,
        encoder_x: torch.Tensor | None = None,
        encoder_attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for the UnifiedTransformerBlock.

        Parameters
        ----------
        x : torch.Tensor[float]
            Input tensor to the transformer block, typically the output from the previous layer.
        sequence_id : torch.Tensor[int]
            Tensor containing sequence IDs for each element in the batch, used for attention masking.
        encoder_x : torch.Tensor[float]
            Sequence of hidden-states at the output of the last layer of another encoder. Used in the cross-attention.
        encoder_attn_mask : torch.Tensor[float]
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        Returns
        -------
        torch.Tensor[float]
            The output tensor after applying the transformer block operations.
        """
        r1 = self.attn(x, sequence_id)
        x = x + r1 / self.scaling_factor

        if self.add_cross_attn and encoder_x is not None:
            r2 = self.cross_attn(x, sequence_id, encoder_x, encoder_attn_mask)
            x = x + r2 / self.scaling_factor

        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3

        return x


class TransformerStack(nn.Module):
    """
    A stack of transformer blocks used in the ESM-C model. Each block is a UnifiedTransformerBlock,
    which can optionally use cross-attention alongside the multi-head self-attention.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        n_layers (int): The number of transformer blocks in the stack.
        scale_residue (bool, optional): Whether to scale the residue connections in each transformer block.
        n_layers_cross_attn (int, optional): The number of transformer blocks that use cross-attention.
        d_cross_attn (int, optional): The dimensionality of the encoder output for the cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
        expansion_ratio: float = 8 / 3,
        n_layers_cross_attn: int = 0,
        d_cross_attn: int | None = None
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    bias=bias,
                    expansion_ratio=expansion_ratio,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                    add_cross_attn=i >= (n_layers - n_layers_cross_attn),
                    d_cross_attn=d_cross_attn
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        encoder_x: torch.Tensor | None = None,
        encoder_attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the TransformerStack.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
            encoder_x (torch.Tensor):
                The encoder output tensor of shape (batch_size, encoder_sequence_length, d_cross_attn).
            encoder_attn_mask (torch.Tensor):
                The encoder attention mask of shape (batch_size, encoder_sequence_length).

        Returns:
            post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
            pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
        """
        hiddens = []
        for block in self.blocks:
            x = block(x, sequence_id, encoder_x, encoder_attn_mask)
            hiddens.append(x)
        return self.norm(x), x, hiddens


@dataclass
class ESMCConfig:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    add_cross_attention: bool = False
    cross_attention_hidden_size: int | None = None


@dataclass
class ESMCOutput:
    last_hidden_state: torch.Tensor | None
    hidden_states: torch.Tensor | None
    sequence_logits: torch.Tensor | None = None
    pooler_output: torch.Tensor | None = None


class ESMC(nn.Module, ESMCInferenceClient):
    """
    ESMC model implementation with optional cross-attention for multimodal input.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
        n_layers_cross_attn (int, optional): The number of transformer layers to equip cross-attention.
        d_cross_attn (int, optional): The dimensionality of the encoder output for the cross-attention.
        add_pooling_layer (bool, optional): Whether to add a pooling layer on the top of the model.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tokenizer: EsmSequenceTokenizer,
        n_layers_cross_attn: int = 0,
        d_cross_attn: int | None = None,
        add_pooling_layer: bool = True
    ):
        super().__init__()
        self.config = ESMCConfig(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            add_cross_attention=n_layers_cross_attn > 0,
            cross_attention_hidden_size=d_cross_attn
        )

        self.embed = nn.Embedding(64, d_model)
        self.transformer = TransformerStack(
            d_model,
            n_heads,
            n_layers,
            n_layers_cross_attn=n_layers_cross_attn,
            d_cross_attn=d_cross_attn
        )
        self.sequence_head = RegressionHead(d_model, 64)
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh()) \
            if add_pooling_layer else None

        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ESMC_600M,
        n_layers_cross_attn: int = 0,
        d_cross_attn: int | None = None,
        add_pooling_layer: bool = True,
        local_files_only: bool = False,
        device: torch.device | None = None
    ):
        import warnings

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name == ESMC_300M:
            with torch.device(device):
                model = cls(
                    d_model=960,
                    n_heads=15,
                    n_layers=30,
                    tokenizer=get_esmc_model_tokenizers(),
                    n_layers_cross_attn=n_layers_cross_attn,
                    d_cross_attn=d_cross_attn,
                    add_pooling_layer=add_pooling_layer
                ).eval()
            path = Path(snapshot_download(
                repo_id="EvolutionaryScale/esmc-300m-2024-12", local_files_only=local_files_only
            ))
            state_dict = torch.load(
                path / "data/weights/esmc_300m_2024_12_v0.pth",
                map_location=device
            )
        elif model_name == ESMC_600M:
            with torch.device(device):
                model = cls(
                    d_model=1152,
                    n_heads=18,
                    n_layers=36,
                    tokenizer=get_esmc_model_tokenizers(),
                    n_layers_cross_attn=n_layers_cross_attn,
                    d_cross_attn=d_cross_attn,
                    add_pooling_layer=add_pooling_layer
                ).eval()
            path = Path(snapshot_download(
                repo_id="EvolutionaryScale/esmc-600m-2024-12", local_files_only=local_files_only
            ))
            state_dict = torch.load(
                path / "data/weights/esmc_600m_2024_12_v0.pth",
                map_location=device
            )
        else:
            raise ValueError(f"Model {model_name} not found in local model registry.")
        
        model_state_dict = model.state_dict()
        missing_keys = [key for key in model_state_dict.keys() if key not in state_dict.keys()]
        unexpected_keys = [key for key in state_dict.keys() if key not in model_state_dict.keys()]
        if unexpected_keys:
            warnings.warn(
                f"Some weights from the model checkpoint were not used when initializing ESMC:\n"
                f"{unexpected_keys}",
                RuntimeWarning,
            )
        if missing_keys:
            warnings.warn(
                f"Some weights of ESMC were not initialized from the model checkpoint and are newly "
                f"initialized:\n{missing_keys}\nYou should probably TRAIN this model on a down-stream "
                "task to be able to use it for predictions and inference.",
                RuntimeWarning,
            )
        model.load_state_dict(state_dict, strict=False)

        if device.type != "cpu":
            model = model.to(torch.bfloat16)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def raw_model(self):
        return self

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_sequence_logits: bool = False
    ) -> ESMCOutput:
        """
        Performs forward pass through the ESMC model. Check utils to see how to tokenize inputs from raw data.

        Args:
            input_ids (torch.Tensor, optional): The amino acid tokens.
            attention_mask (torch.Tensor, optional): The sequence ID.
            encoder_hidden_states (torch.Tensor, optional): The encoder output for cross-attention.
            encoder_attention_mask (torch.Tensor, optional): The attention mask for the encoder output.
            output_sequence_logits (bool, optional): Whether to output sequence logits.

        Returns:
            ESMCOutput: The output of the ESMC model.

        """
        if attention_mask is None:
            # For ESMC, a boolean mask is created in place of attention_mask if not specified.
            attention_mask = input_ids != self.tokenizer.pad_token_id

        x = self.embed(input_ids)
        x, _, hiddens = self.transformer(
            x,
            sequence_id=attention_mask,
            encoder_x=encoder_hidden_states,
            encoder_attn_mask=encoder_attention_mask
        )

        # Stack hidden states into a [n_layers, B, L, D] matrix.
        hiddens = torch.stack(hiddens, dim=0)  # type: ignore

        sequence_logits = self.sequence_head(x) if output_sequence_logits else None
        pooled_output = self.pooler(x[:, 0]) if self.pooler is not None else None
        output = ESMCOutput(
            last_hidden_state=x,
            hidden_states=hiddens,
            sequence_logits=sequence_logits,
            pooler_output=pooled_output
        )
        return output
