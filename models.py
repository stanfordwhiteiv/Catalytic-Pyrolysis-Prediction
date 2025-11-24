"""
Physics-Informed Transformer for TGA Prediction

Core neural network architectures for predicting char yield from thermogravimetric
analysis (TGA) data. The key innovation is using cross-attention to let the model
learn how static material properties influence temperature-dependent degradation,
rather than just concatenating everything and hoping for the best.

After many failed experiments with LSTMs and vanilla transformers, this architecture
finally gave us the results we needed. The cross-attention mechanism was the key.

Author: [Your Name]
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need" (Vaswani et al.)

    Adds position information to embeddings since transformers have no inherent
    notion of sequence order. Uses sin/cos functions at different frequencies
    so the model can learn to attend to relative positions.

    Spent way too long debugging this before realizing I had the dimensions wrong.
    The key insight: pe needs shape (1, max_len, d_model) for proper broadcasting.

    Args:
        d_model: Embedding dimension (must match your transformer hidden dim)
        max_len: Maximum sequence length. 5000 is overkill for TGA but costs
                nothing and prevents edge cases at 3am.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Pre-compute positional encodings once at init, not during forward
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term creates different frequencies for each dimension
        # Even dims get sin, odd dims get cos
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Add batch dim: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer - saved with model but not trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added (same shape)
        """
        # Only use positions up to current sequence length
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayerWithCrossAttention(nn.Module):
    """
    Custom transformer encoder layer with cross-attention to static features.

    This is where the magic happens. Standard transformers just do self-attention,
    but we need the dynamic sequence (temp, prev_char) to "see" the static material
    properties. Cross-attention lets dynamic features query the static ones.

    Architecture (order matters!):
        1. Self-attention: dynamic features attend to themselves
        2. Cross-attention: dynamic attends to static (THE KEY INNOVATION)
        3. Feedforward: standard MLP for nonlinear transformation

    Each sub-layer has residual connections and layer norm. We use Post-LN
    (norm after residual) following the original transformer paper.

    Args:
        d_model: Model dimension. 64 works well, 128 didn't help much.
        nhead: Number of attention heads. Must divide d_model evenly.
               4 heads with d_model=64 = 16 dims per head.
        dim_feedforward: FFN hidden dim. We use 16 which seems tiny but larger
                        values didn't help and just slow things down.
        dropout: Dropout probability. 0.1 is standard, higher hurts performance.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention: sequence attends to itself
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Cross-attention: THE KEY INNOVATION
        # Query = dynamic features, Key/Value = static features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network (2-layer MLP)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

        # Layer norm after each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Separate dropout for each residual (probably overkill but whatever)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # LeakyReLU to avoid dead neurons. Tried GELU, didn't see improvement.
        self.activation = nn.LeakyReLU()

    def forward(
        self,
        src: torch.Tensor,
        static: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the encoder layer.

        Note the shapes - PyTorch MultiheadAttention expects (seq, batch, dim)
        not (batch, seq, dim). Took me embarrassingly long to figure that out.

        Args:
            src: Dynamic sequence (seq_len, batch_size, d_model)
            static: Static features (1, batch_size, d_model)
                   It's (1, batch, dim) because there's one static vector per sample

        Returns:
            Transformed sequence (seq_len, batch_size, d_model)
        """
        # Self-attention block
        src2, _ = self.self_attn(src, src, src)  # Q=K=V=src
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Cross-attention block - src queries static features
        src2, _ = self.cross_attn(src, static, static)  # Q=src, K=V=static
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        return src


class TransformerEncoderWithCrossAttention(nn.Module):
    """
    Stack of transformer encoder layers with cross-attention.

    Just stacks N layers. Each gets same static features but processes
    increasingly refined dynamic representations.

    2 layers is enough for our task. More = more params but diminishing returns.
    Tried 3 and 4 layers, marginal improvement not worth the compute.

    Args:
        encoder_layer: Single encoder layer (reused, not cloned - acts as regularization)
        num_layers: How many layers to stack
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        # Same layer object reused, weights shared across layers
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        static: torch.Tensor
    ) -> torch.Tensor:
        """Pass through all layers sequentially."""
        for layer in self.layers:
            src = layer(src, static)
        return src


class TransformerWithStaticNet(nn.Module):
    """
    The main model: Physics-Informed Transformer with Cross-Attention.

    Predicts char yield from TGA data by combining:
    1. Dynamic features: temperature sequence + previous char values
    2. Static features: material properties that don't change with temperature

    The key insight is that material properties should INFLUENCE how char evolves
    with temperature, not just be concatenated as extra features. Cross-attention
    lets the model learn this relationship properly.

    After trying concatenation, FiLM conditioning, and various other approaches,
    cross-attention gave the best results. Makes intuitive sense - different
    materials should have different temperature-dependent behaviors.

    Simplified architecture:
        Dynamic -> Linear -> PosEnc ──┐
                                      ├─> Cross-Attn Transformer -> Linear -> Output
        Static -> Linear ─────────────┘

    Args:
        dynamic_input_dim: Size of dynamic features (2 for temp + prev_char)
        static_input_dim: Size of static features (varies by dataset)
        hidden_dim: Transformer hidden size. 64 works well.
        num_layers: Number of transformer layers. 2 is enough.
        output_dim: Output size. 1 for single-step prediction.
        nhead: Attention heads. 4 for hidden_dim=64 (16 dims/head).
        dim_feedforward: FFN hidden size. 16 is surprisingly enough.
        max_seq_len: Max sequence length for positional encoding.
        dropout: Dropout rate. Stick with 0.1.

    Example:
        >>> model = TransformerWithStaticNet(
        ...     dynamic_input_dim=2,   # (temperature, prev_char)
        ...     static_input_dim=8,    # material properties
        ...     hidden_dim=64,
        ...     num_layers=2,
        ...     nhead=4
        ... )
        >>> # IMPORTANT: shapes matter!
        >>> dynamic_x = torch.randn(32, 10, 2)   # (batch, seq_len, features)
        >>> static_x = torch.randn(32, 8)        # (batch, features) - NO seq dim!
        >>> output = model(dynamic_x, static_x)  # (batch, 1)
    """

    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        nhead: int = 4,
        dim_feedforward: int = 256,
        max_seq_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project inputs to hidden dimension (simple linear, nothing fancy)
        self.dynamic_input_fc = nn.Linear(dynamic_input_dim, hidden_dim)
        self.static_input_fc = nn.Linear(static_input_dim, hidden_dim)

        # Only dynamic sequence needs positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)

        # Build transformer encoder with our custom cross-attention layers
        encoder_layer = TransformerEncoderLayerWithCrossAttention(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoderWithCrossAttention(
            encoder_layer, num_layers
        )

        # Final output projection
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Watch the tensor shapes! Lots of transposing because PyTorch transformers
        expect (seq, batch, dim) but we store data as (batch, seq, dim).
        I always forget this and waste hours debugging shape mismatches.

        Args:
            dynamic_x: Dynamic input (batch, seq_len, dynamic_dim)
                      Contains temperature and previous char values
            static_x: Static input (batch, static_dim)
                     Material properties - NOTE: no sequence dimension!

        Returns:
            Predictions (batch, output_dim)
        """
        batch_size, seq_len, _ = dynamic_x.size()

        # Project dynamic features and add positional encoding
        dynamic_emb = self.dynamic_input_fc(dynamic_x)   # (batch, seq, hidden)
        dynamic_emb = self.pos_encoder(dynamic_emb)       # Add position info

        # Transpose for transformer: (batch, seq, dim) -> (seq, batch, dim)
        dynamic_emb = dynamic_emb.transpose(0, 1)

        # Project static: (batch, static) -> (batch, hidden) -> (1, batch, hidden)
        static_emb = self.static_input_fc(static_x).unsqueeze(0)

        # Run through transformer with cross-attention
        transformer_out = self.transformer_encoder(dynamic_emb, static_emb)

        # Take last timestep output only: (seq, batch, dim) -> (batch, dim)
        transformer_out = transformer_out[-1, :, :]

        # Final prediction
        out = self.fc(transformer_out)

        return out


class TransformerStandardNet(nn.Module):
    """
    Baseline: Standard Transformer with concatenation (no cross-attention).

    This is what you'd try first - concatenate static features with dynamic
    features at each timestep. Works okay but cross-attention beats it.

    Included for ablation studies to prove cross-attention actually helps.
    (Reviewers love ablations. Give them what they want.)

    Architecture:
        [Dynamic | Static repeated] -> Linear -> Standard Transformer -> Output

    Same args as TransformerWithStaticNet.
    """

    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        nhead: int = 4,
        dim_feedforward: int = 256,
        max_seq_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection takes concatenated features
        total_input_dim = dynamic_input_dim + static_input_dim
        self.input_fc = nn.Linear(total_input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)

        # Standard PyTorch transformer (no cross-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Thank god they finally added this option
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with naive concatenation.

        Static features repeated at each timestep - the "obvious" approach
        that cross-attention improves upon.
        """
        batch_size, seq_len, _ = dynamic_x.size()

        # Expand static to match sequence: (batch, dim) -> (batch, seq, dim)
        static_expanded = static_x.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate along feature dimension
        combined = torch.cat([dynamic_x, static_expanded], dim=2)

        # Project and encode
        x = self.input_fc(combined)
        x = self.pos_encoder(x)

        # Standard transformer (batch_first=True so no transpose needed)
        transformer_out = self.transformer_encoder(x)

        # Take last timestep
        transformer_out = transformer_out[:, -1, :]

        out = self.fc(transformer_out)

        return out


def compute_physics_loss(
    predictions: torch.Tensor,
    dynamic_x: torch.Tensor,
    monotonicity_lambda: float = 10.0,
    over_penalty_lambda: float = 1000.0
) -> tuple:
    """
    Compute physics-informed loss penalties.

    CRITICAL for getting physically plausible predictions. Without these,
    the model sometimes predicts char > 100% or increasing char with temperature
    (both physically impossible for TGA). Reviewers will definitely ask about this.

    Two constraints:
    1. Over-penalty: Char cannot exceed 100% (it's a percentage...)
    2. Monotonicity: Char decreases with temperature (basic TGA physics)

    These are soft constraints (penalties), not hard constraints. The lambdas
    control enforcement strength. After MUCH tuning:
    - over_penalty_lambda=1000: harsh penalty for impossible values
    - monotonicity_lambda=10: gentler, allows small violations

    Args:
        predictions: Model output (batch, 1)
        dynamic_x: Dynamic input (batch, seq_len, dim)
                  prev_char assumed at index 1 of last dimension
        monotonicity_lambda: Weight for monotonicity penalty
        over_penalty_lambda: Weight for over-100 penalty

    Returns:
        Tuple of (over_penalty, mono_penalty) - raw penalties, NOT weighted.
        Caller should weight these before adding to loss.
    """
    # Penalty for predictions > 100 (physically impossible)
    over_penalty = torch.mean(torch.relu(predictions - 100.0))

    # Get last prev_char from input sequence
    prev_chars = dynamic_x[:, :, 1]              # (batch, seq_len)
    last_char = prev_chars[:, -1].unsqueeze(1)   # (batch, 1)

    # Char should decrease: if prediction > last_char, that's bad
    char_diff = predictions - last_char
    monotonicity_penalty = torch.mean(torch.relu(char_diff))

    return over_penalty, monotonicity_penalty


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Useful for model comparison and sanity checks. Our TransformerWithStaticNet
    has ~50k params which is tiny by modern standards but plenty for this task.

    Args:
        model: Any PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
