# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GoalHead model for language-conditioned goal latent prediction.

This module defines the GoalHead architecture that predicts goal latent
representations from initial images and language instructions.
"""

import logging
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPBlock(nn.Module):
    """Simple MLP block with LayerNorm and GELU activation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for fusing visual and text features."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention
        x = self.norm1(query)
        kv = self.norm1(key_value)
        attn_out, _ = self.attention(x, kv, kv, key_padding_mask=key_padding_mask)
        x = query + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning."""
    
    def __init__(self, feature_dim: int, conditioning_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(conditioning_dim, feature_dim)
        self.beta_proj = nn.Linear(conditioning_dim, feature_dim)
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features to modulate [B, ..., feature_dim]
            conditioning: Conditioning vector [B, conditioning_dim]
        """
        gamma = self.gamma_proj(conditioning)
        beta = self.beta_proj(conditioning)
        
        # Expand gamma and beta to match x dimensions
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        return gamma * x + beta


# =============================================================================
# Improved Components for GoalHead v2
# =============================================================================


class AttentionPooling(nn.Module):
    """
    Learnable Attention Pooling (MAP head style).
    
    Uses learnable query tokens to aggregate information from input tokens
    via cross-attention. This is more expressive than mean pooling as it
    learns which tokens to attend to.
    
    Reference: SigLIP2's MAP head for pooled representations
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_queries: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_queries: Number of learnable query tokens (1 for single pooled output)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.query = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        
        # Cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # Optional MLP after pooling
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, D]
            key_padding_mask: Optional mask for padding tokens [B, N]
            
        Returns:
            Pooled output [B, num_queries, D] or [B, D] if num_queries=1
        """
        batch_size = x.shape[0]
        
        # Expand query for batch
        q = self.query.expand(batch_size, -1, -1)
        
        # Cross-attention: query attends to input tokens
        x_norm = self.norm(x)
        pooled, _ = self.attn(q, x_norm, x_norm, key_padding_mask=key_padding_mask)
        pooled = q + pooled
        
        # MLP
        pooled = pooled + self.mlp(self.mlp_norm(pooled))
        
        # Squeeze if single query
        if self.num_queries == 1:
            pooled = pooled.squeeze(1)  # [B, D]
        
        return pooled


class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention block (Flamingo style).
    
    Uses a learnable gate (initialized to zero) to control the contribution
    of cross-attention. This stabilizes training by starting with identity
    and gradually learning to incorporate the attended information.
    
    Reference: Flamingo's gated cross-attention for stable frozen backbone training
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        init_gate: float = 0.0,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            init_gate: Initial value for the gate (0 = start with identity)
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        
        # Learnable gate initialized to init_gate (typically 0 or small)
        self.gate = nn.Parameter(torch.tensor(init_gate))
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Query tokens [B, N_q, D]
            key_value: Key/Value tokens [B, N_kv, D]
            key_padding_mask: Optional mask for key/value [B, N_kv]
            
        Returns:
            Output tokens [B, N_q, D]
        """
        # Normalize
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        
        # Cross-attention with gating
        attn_out, _ = self.attention(q, kv, kv, key_padding_mask=key_padding_mask)
        
        # Apply gate (tanh ensures bounded influence)
        gated_attn = torch.tanh(self.gate) * attn_out
        x = query + gated_attn
        
        # MLP with residual
        x = x + self.mlp(self.norm_out(x))
        
        return x


class QFormerBridge(nn.Module):
    """
    Q-Former style bridge module with learnable queries.
    
    Uses learnable query tokens that cross-attend to both visual and text
    modalities, bridging the gap between frozen encoders.
    
    Reference: BLIP-2's Q-Former for efficient vision-language alignment
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_queries: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gated_attn: bool = True,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_queries: Number of learnable query tokens
            num_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_gated_attn: Whether to use gated cross-attention
        """
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        
        # Cross-attention to visual features
        if use_gated_attn:
            self.visual_attn_layers = nn.ModuleList([
                GatedCrossAttention(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.visual_attn_layers = nn.ModuleList([
                CrossAttentionBlock(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
        
        # Cross-attention to text features
        if use_gated_attn:
            self.text_attn_layers = nn.ModuleList([
                GatedCrossAttention(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.text_attn_layers = nn.ModuleList([
                CrossAttentionBlock(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
        
        # Self-attention for query refinement
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: Visual features [B, N_v, D]
            text_tokens: Text features [B, N_t, D]
            visual_mask: Optional padding mask for visual [B, N_v]
            text_mask: Optional padding mask for text [B, N_t]
            
        Returns:
            Query outputs [B, num_queries, D]
        """
        batch_size = visual_tokens.shape[0]
        
        # Expand queries for batch
        q = self.queries.expand(batch_size, -1, -1)
        
        # Alternating cross-attention to visual and text, with self-attention
        for v_attn, t_attn, s_attn in zip(
            self.visual_attn_layers,
            self.text_attn_layers,
            self.self_attn_layers,
        ):
            # Cross-attend to visual
            q = v_attn(q, visual_tokens, visual_mask)
            # Cross-attend to text
            q = t_attn(q, text_tokens, text_mask)
            # Self-attention for query interaction
            q = s_attn(q)
        
        return self.output_norm(q)


class MultiHopFiLM(nn.Module):
    """
    Multi-hop FiLM conditioning.
    
    Applies FiLM modulation at multiple stages, allowing the model to
    progressively refine features based on language conditioning.
    
    Reference: Multi-stage FiLM for complex language conditioning
    """
    
    def __init__(
        self,
        feature_dim: int,
        conditioning_dim: int,
        hidden_dim: int,
        num_hops: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: Dimension of features to modulate
            conditioning_dim: Dimension of conditioning vector
            hidden_dim: Hidden dimension for intermediate layers
            num_hops: Number of FiLM modulation stages
            dropout: Dropout probability
        """
        super().__init__()
        self.num_hops = num_hops
        
        # Condition encoder with hop-specific projections
        self.condition_proj = nn.Linear(conditioning_dim, hidden_dim)
        
        # FiLM layers for each hop
        self.film_layers = nn.ModuleList([
            FiLMLayer(hidden_dim, hidden_dim)
            for _ in range(num_hops)
        ])
        
        # MLP layers between FiLM modulations
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_hops)
        ])
        
        # Input/output projections
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: Features to modulate [B, D] or [B, N, D]
            conditioning: Conditioning vector [B, C]
            
        Returns:
            Modulated features [B, D] or [B, N, D]
        """
        # Project conditioning
        c = self.condition_proj(conditioning)  # [B, hidden_dim]
        
        # Project input
        x = self.input_proj(features)
        
        # Multi-hop FiLM
        for film, mlp in zip(self.film_layers, self.mlp_layers):
            x = film(x, c)
            x = x + mlp(x)  # Residual connection
        
        # Project output
        return self.output_proj(x)


class ImprovedGoalHead(nn.Module):
    """
    Improved GoalHead with:
    1. Attention pooling instead of mean pooling
    2. Q-Former style learnable queries for cross-modal fusion
    3. Gated cross-attention for training stability
    4. Multi-hop FiLM conditioning option
    
    This addresses key weaknesses in the original GoalHead:
    - Mean pooling loses important spatial/token information
    - Single-query cross-attention is too limited
    - Uncontrolled cross-attention can destabilize training
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        output_dim: int,
        proprio_dim: int = 0,
        hidden_dim: int = 512,
        num_layers: int = 2,
        fusion_mode: str = "qformer",
        num_attention_heads: int = 8,
        num_queries: int = 16,
        dropout: float = 0.1,
        use_proprio: bool = False,
        use_gated_attn: bool = True,
        film_hops: int = 3,
    ):
        """
        Initialize ImprovedGoalHead.
        
        Args:
            visual_dim: Dimension of visual features from encoder
            text_dim: Dimension of text features from encoder
            output_dim: Dimension of output goal latent
            proprio_dim: Dimension of proprioception state (if used)
            hidden_dim: Hidden dimension for layers
            num_layers: Number of layers in Q-Former/cross-attention
            fusion_mode: "qformer", "gated_cross_attn", "multihop_film"
            num_attention_heads: Number of attention heads
            num_queries: Number of learnable queries for Q-Former
            dropout: Dropout probability
            use_proprio: Whether to include proprioception
            use_gated_attn: Whether to use gated attention (stabilizes training)
            film_hops: Number of hops for multi-hop FiLM
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.use_proprio = use_proprio and proprio_dim > 0
        
        # === Improved Pooling: Attention Pooling instead of mean ===
        # Visual attention pooling
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.visual_pool = AttentionPooling(
            embed_dim=hidden_dim,
            num_queries=1,  # Single pooled output
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Text attention pooling
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.text_pool = AttentionPooling(
            embed_dim=hidden_dim,
            num_queries=1,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Proprio projection (if used)
        if self.use_proprio:
            self.proprio_proj = nn.Linear(proprio_dim, hidden_dim // 4)
            proprio_contribution = hidden_dim // 4
        else:
            proprio_contribution = 0
        
        # === Fusion Module ===
        if fusion_mode == "qformer":
            # Q-Former style with learnable queries
            self.fusion = QFormerBridge(
                embed_dim=hidden_dim,
                num_queries=num_queries,
                num_layers=num_layers,
                num_heads=num_attention_heads,
                dropout=dropout,
                use_gated_attn=use_gated_attn,
            )
            # Pool Q-Former outputs
            self.query_pool = AttentionPooling(
                embed_dim=hidden_dim,
                num_queries=1,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
        elif fusion_mode == "gated_cross_attn":
            # Gated cross-attention (simpler than Q-Former)
            self.fusion = nn.ModuleList([
                GatedCrossAttention(hidden_dim, num_attention_heads, dropout)
                for _ in range(num_layers)
            ])
        elif fusion_mode == "multihop_film":
            # Multi-hop FiLM conditioning
            self.fusion = MultiHopFiLM(
                feature_dim=hidden_dim,
                conditioning_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_hops=film_hops,
                dropout=dropout,
            )
        elif fusion_mode == "concat":
            # Fallback to concat (with improved pooling)
            input_dim = hidden_dim * 2 + proprio_contribution
            self.fusion = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        
        # Proprio fusion for non-concat modes
        if self.use_proprio and fusion_mode != "concat":
            self.proprio_fusion = nn.Linear(hidden_dim + proprio_contribution, hidden_dim)
        
        # Output MLP layers
        layers = []
        for i in range(num_layers - 1):
            layers.append(MLPBlock(hidden_dim, hidden_dim, hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.output_mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"🧠 ImprovedGoalHead initialized: fusion={fusion_mode}, "
            f"visual_dim={visual_dim}, text_dim={text_dim}, "
            f"output_dim={output_dim}, num_queries={num_queries}, "
            f"use_gated_attn={use_gated_attn}, use_proprio={self.use_proprio}"
        )
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to predict goal latent.
        
        Args:
            visual_features: Visual features from encoder [B, N, D] or [B, D]
            text_features: Text features from encoder [B, text_dim] or [B, L, text_dim]
            proprio: Optional proprioception state [B, proprio_dim]
            
        Returns:
            z_goal_pred: Predicted goal latent [B, output_dim]
        """
        # Handle visual features shape
        if visual_features.dim() == 6:
            # [B, T, V, H, W, D] -> [B, T*V*H*W, D]
            visual_features = rearrange(visual_features, "b t v h w d -> b (t v h w) d")
        elif visual_features.dim() == 2:
            # [B, D] -> [B, 1, D]
            visual_features = visual_features.unsqueeze(1)
        
        # Handle text features shape
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)  # [B, 1, D]
        
        # Project to hidden dimension
        v_tokens = self.visual_proj(visual_features)  # [B, N, hidden_dim]
        t_tokens = self.text_proj(text_features)  # [B, L, hidden_dim]
        
        # Process proprio if used
        p_feat = None
        if self.use_proprio and proprio is not None:
            p_feat = self.proprio_proj(proprio)  # [B, hidden_dim//4]
        
        # Fusion
        if self.fusion_mode == "qformer":
            # Q-Former: learnable queries attend to both modalities
            query_outputs = self.fusion(v_tokens, t_tokens)  # [B, num_queries, hidden_dim]
            fused = self.query_pool(query_outputs)  # [B, hidden_dim]
            
        elif self.fusion_mode == "gated_cross_attn":
            # Gated cross-attention: visual attends to text
            v_pooled = self.visual_pool(v_tokens)  # [B, hidden_dim]
            v_pooled = v_pooled.unsqueeze(1)  # [B, 1, hidden_dim]
            
            for layer in self.fusion:
                v_pooled = layer(v_pooled, t_tokens)
            fused = v_pooled.squeeze(1)  # [B, hidden_dim]
            
        elif self.fusion_mode == "multihop_film":
            # Multi-hop FiLM: visual modulated by text
            v_pooled = self.visual_pool(v_tokens)  # [B, hidden_dim]
            t_pooled = self.text_pool(t_tokens)  # [B, hidden_dim]
            fused = self.fusion(v_pooled, t_pooled)  # [B, hidden_dim]
            
        elif self.fusion_mode == "concat":
            # Concat with improved attention pooling
            v_pooled = self.visual_pool(v_tokens)  # [B, hidden_dim]
            t_pooled = self.text_pool(t_tokens)  # [B, hidden_dim]
            if p_feat is not None:
                fused = torch.cat([v_pooled, t_pooled, p_feat], dim=-1)
            else:
                fused = torch.cat([v_pooled, t_pooled], dim=-1)
            fused = self.fusion(fused)
        
        # Add proprio for non-concat modes
        if p_feat is not None and self.fusion_mode != "concat":
            fused = self.proprio_fusion(torch.cat([fused, p_feat], dim=-1))
        
        # Output projection
        z_goal_pred = self.output_mlp(fused)  # [B, output_dim]
        
        return z_goal_pred


class GoalHead(nn.Module):
    """
    Goal latent prediction head.
    
    Predicts goal latent z_goal from:
    - I0: Initial image (encoded by frozen visual encoder)
    - T: Language instruction (encoded by frozen text encoder)
    - Optional: Initial proprioception state
    
    Fusion modes:
    - concat: Concatenate features and pass through MLP
    - cross_attn: Use cross-attention between visual and text features
    - film: Use FiLM conditioning of visual features by text
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        output_dim: int,
        proprio_dim: int = 0,
        hidden_dim: int = 512,
        num_layers: int = 2,
        fusion_mode: str = "concat",
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        use_proprio: bool = False,
    ):
        """
        Initialize GoalHead.
        
        Args:
            visual_dim: Dimension of visual features from encoder
            text_dim: Dimension of text features from encoder
            output_dim: Dimension of output goal latent (should match encoder latent dim)
            proprio_dim: Dimension of proprioception state (if used)
            hidden_dim: Hidden dimension for MLP layers
            num_layers: Number of MLP layers
            fusion_mode: How to fuse visual and text features ("concat", "cross_attn", "film")
            num_attention_heads: Number of attention heads for cross_attn mode
            dropout: Dropout probability
            use_proprio: Whether to include proprioception in input
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.proprio_dim = proprio_dim
        self.fusion_mode = fusion_mode
        self.use_proprio = use_proprio and proprio_dim > 0
        
        # Visual feature projection (pool spatial dimensions)
        self.visual_pool = nn.AdaptiveAvgPool1d(1)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Text feature projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Proprio projection (if used)
        if self.use_proprio:
            self.proprio_proj = nn.Linear(proprio_dim, hidden_dim // 4)
            proprio_contribution = hidden_dim // 4
        else:
            proprio_contribution = 0
        
        # Fusion layers
        if fusion_mode == "concat":
            input_dim = hidden_dim * 2 + proprio_contribution
            self.fusion = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_mode == "cross_attn":
            self.fusion = CrossAttentionBlock(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            if self.use_proprio:
                self.proprio_fusion = nn.Linear(hidden_dim + proprio_contribution, hidden_dim)
        elif fusion_mode == "film":
            self.fusion = FiLMLayer(hidden_dim, hidden_dim)
            if self.use_proprio:
                self.proprio_fusion = nn.Linear(hidden_dim + proprio_contribution, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        
        # Output MLP layers
        layers = []
        current_dim = hidden_dim
        for i in range(num_layers - 1):
            layers.append(MLPBlock(current_dim, hidden_dim, hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.output_mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"🧠 GoalHead initialized: fusion={fusion_mode}, "
                   f"visual_dim={visual_dim}, text_dim={text_dim}, "
                   f"output_dim={output_dim}, use_proprio={self.use_proprio}")
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to predict goal latent.
        
        Args:
            visual_features: Visual features from encoder [B, N, D] or [B, T, V, H, W, D]
            text_features: Text features from encoder [B, text_dim] or [B, L, text_dim]
            proprio: Optional proprioception state [B, proprio_dim]
            
        Returns:
            z_goal_pred: Predicted goal latent [B, output_dim]
        """
        batch_size = visual_features.shape[0]
        
        # Process visual features (handle different input shapes)
        if visual_features.dim() == 6:
            # [B, T, V, H, W, D] -> [B, T*V*H*W, D]
            visual_features = rearrange(visual_features, "b t v h w d -> b (t v h w) d")
        elif visual_features.dim() == 2:
            # [B, D] -> [B, 1, D]
            visual_features = visual_features.unsqueeze(1)
        
        # Pool visual features: [B, N, D] -> [B, D]
        v_feat = visual_features.mean(dim=1)  # Simple mean pooling
        v_feat = self.visual_proj(v_feat)  # [B, hidden_dim]
        
        # Process text features
        if text_features.dim() == 3:
            # [B, L, text_dim] -> [B, text_dim] via mean pooling
            t_feat = text_features.mean(dim=1)
        else:
            t_feat = text_features
        t_feat = self.text_proj(t_feat)  # [B, hidden_dim]
        
        # Process proprio if used
        p_feat = None
        if self.use_proprio and proprio is not None:
            p_feat = self.proprio_proj(proprio)  # [B, hidden_dim//4]
        
        # Fuse features
        if self.fusion_mode == "concat":
            if p_feat is not None:
                fused = torch.cat([v_feat, t_feat, p_feat], dim=-1)
            else:
                fused = torch.cat([v_feat, t_feat], dim=-1)
            fused = self.fusion(fused)
        elif self.fusion_mode == "cross_attn":
            # Visual as query, text as key/value
            v_feat = v_feat.unsqueeze(1)  # [B, 1, hidden_dim]
            t_feat = t_feat.unsqueeze(1)  # [B, 1, hidden_dim]
            fused = self.fusion(v_feat, t_feat).squeeze(1)  # [B, hidden_dim]
            if p_feat is not None:
                fused = self.proprio_fusion(torch.cat([fused, p_feat], dim=-1))
        elif self.fusion_mode == "film":
            fused = self.fusion(v_feat, t_feat)  # [B, hidden_dim]
            if p_feat is not None:
                fused = self.proprio_fusion(torch.cat([fused, p_feat], dim=-1))
        
        # Output projection
        z_goal_pred = self.output_mlp(fused)  # [B, output_dim]
        
        return z_goal_pred


class GoalHeadWithEncoders(nn.Module):
    """
    Wrapper that combines GoalHead with frozen visual and text encoders.
    
    This provides a convenient interface for end-to-end goal prediction
    from raw images and text strings.
    """
    
    def __init__(
        self,
        visual_encoder: nn.Module,
        text_encoder: nn.Module,
        goal_head: GoalHead,
        freeze_visual: bool = True,
        freeze_text: bool = True,
    ):
        """
        Initialize GoalHeadWithEncoders.
        
        Args:
            visual_encoder: Pretrained visual encoder (e.g., DINO, VJEPA)
            text_encoder: Pretrained text encoder (e.g., CLIP, SentenceTransformer)
            goal_head: GoalHead module
            freeze_visual: Whether to freeze visual encoder
            freeze_text: Whether to freeze text encoder
        """
        super().__init__()
        
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.goal_head = goal_head
        
        # Freeze encoders
        if freeze_visual:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
            logger.info("🔒 Visual encoder frozen")
        
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()
            logger.info("🔒 Text encoder frozen")
        
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
    
    def train(self, mode: bool = True):
        """Override train to keep frozen encoders in eval mode."""
        super().train(mode)
        if self.freeze_visual:
            self.visual_encoder.eval()
        if self.freeze_text:
            self.text_encoder.eval()
        return self
    
    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass from raw images and text.
        
        Args:
            images: Input images [B, C, H, W] or [B, T, C, H, W]
            text_tokens: Tokenized text [B, L] (depends on text encoder)
            proprio: Optional proprioception [B, proprio_dim]
            
        Returns:
            z_goal_pred: Predicted goal latent [B, output_dim]
        """
        # Encode visual features
        with torch.set_grad_enabled(not self.freeze_visual):
            visual_features = self.visual_encoder(images)
        
        # Encode text features
        with torch.set_grad_enabled(not self.freeze_text):
            text_features = self.text_encoder(text_tokens)
        
        # Predict goal latent
        z_goal_pred = self.goal_head(visual_features, text_features, proprio)
        
        return z_goal_pred
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using the visual encoder."""
        with torch.set_grad_enabled(not self.freeze_visual):
            return self.visual_encoder(images)
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode text using the text encoder."""
        with torch.set_grad_enabled(not self.freeze_text):
            return self.text_encoder(text_tokens)


def compute_goal_loss(
    z_goal_pred: torch.Tensor,
    z_goal_gt: torch.Tensor,
    l2_weight: float = 1.0,
    cosine_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute goal prediction loss.
    
    Supports both pooled [B, D] and patch-level [B, N, D] predictions.
    
    Args:
        z_goal_pred: Predicted goal latent [B, D] or [B, N, D]
        z_goal_gt: Ground truth goal latent [B, D] or [B, N, D]
        l2_weight: Weight for L2 loss
        cosine_weight: Weight for cosine similarity loss
        
    Returns:
        Dictionary with loss terms and total loss
    """
    losses = {}
    
    # Handle patch-level predictions
    if z_goal_pred.dim() == 3 and z_goal_gt.dim() == 3:
        # [B, N, D] - compute loss per patch and average
        l2_loss = F.mse_loss(z_goal_pred, z_goal_gt)
        if cosine_weight > 0:
            # Cosine similarity per patch, then average
            cosine_sim = F.cosine_similarity(z_goal_pred, z_goal_gt, dim=-1).mean()
            cosine_loss = 1.0 - cosine_sim
        else:
            cosine_loss = 0.0
            cosine_sim = None
    else:
        # L2 loss
        l2_loss = F.mse_loss(z_goal_pred, z_goal_gt)
        if cosine_weight > 0:
            cosine_sim = F.cosine_similarity(z_goal_pred, z_goal_gt, dim=-1).mean()
            cosine_loss = 1.0 - cosine_sim
        else:
            cosine_loss = 0.0
            cosine_sim = None
    
    losses["l2_loss"] = l2_loss
    
    # Always compute cosine_sim for monitoring (regardless of weight)
    if cosine_sim is None:
        if z_goal_pred.dim() == 3 and z_goal_gt.dim() == 3:
            cosine_sim = F.cosine_similarity(z_goal_pred, z_goal_gt, dim=-1).mean()
        else:
            cosine_sim = F.cosine_similarity(z_goal_pred, z_goal_gt, dim=-1).mean()
        cosine_loss = 1.0 - cosine_sim
    
    losses["cosine_loss"] = cosine_loss if isinstance(cosine_loss, torch.Tensor) else torch.tensor(cosine_loss, device=z_goal_pred.device)
    losses["cosine_sim"] = cosine_sim
    
    # Total loss (only weighted terms contribute to gradient)
    total_loss = l2_weight * l2_loss
    if cosine_weight > 0:
        total_loss = total_loss + cosine_weight * cosine_loss
    losses["total_loss"] = total_loss
    
    return losses


def compute_goal_loss_v2(
    z_goal_pred: torch.Tensor,
    z_goal_gt: torch.Tensor,
    z_init: Optional[torch.Tensor] = None,
    negative_samples: Optional[torch.Tensor] = None,
    l2_weight: float = 1.0,
    cosine_weight: float = 0.5,
    contrastive_weight: float = 0.1,
    delta_reg_weight: float = 0.01,
    temperature: float = 0.07,
) -> Dict[str, torch.Tensor]:
    """
    Improved goal prediction loss with:
    1. Cosine loss always enabled (direction alignment)
    2. Contrastive loss with hard negatives
    3. Delta norm regularization (for delta predictions)
    
    Args:
        z_goal_pred: Predicted goal latent [B, D] or [B, N, D]
        z_goal_gt: Ground truth goal latent [B, D] or [B, N, D]
        z_init: Initial latent for delta regularization [B, D] or [B, N, D]
        negative_samples: Negative goal latents [B, K, D] for contrastive loss
        l2_weight: Weight for L2 (MSE) loss
        cosine_weight: Weight for cosine similarity loss
        contrastive_weight: Weight for contrastive loss
        delta_reg_weight: Weight for delta norm regularization
        temperature: Temperature for contrastive loss
        
    Returns:
        Dictionary with loss terms and total loss
    """
    losses = {}
    device = z_goal_pred.device
    
    # Flatten patch-level to compute losses
    if z_goal_pred.dim() == 3:
        # [B, N, D] -> [B*N, D] for per-patch losses
        B, N, D = z_goal_pred.shape
        pred_flat = z_goal_pred.view(-1, D)
        gt_flat = z_goal_gt.view(-1, D)
        is_patch_level = True
    else:
        pred_flat = z_goal_pred
        gt_flat = z_goal_gt
        is_patch_level = False
    
    # === L2 Loss ===
    l2_loss = F.mse_loss(z_goal_pred, z_goal_gt)
    losses["l2_loss"] = l2_loss
    
    # === Cosine Loss (always enabled for direction alignment) ===
    cosine_sim = F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean()
    cosine_loss = 1.0 - cosine_sim
    losses["cosine_loss"] = cosine_loss
    losses["cosine_sim"] = cosine_sim
    
    # === Contrastive Loss with Hard Negatives ===
    contrastive_loss = torch.tensor(0.0, device=device)
    if contrastive_weight > 0 and negative_samples is not None:
        # negative_samples: [B, K, D]
        B = z_goal_pred.shape[0]
        
        # Normalize for cosine similarity
        pred_norm = F.normalize(z_goal_pred.view(B, -1), dim=-1)  # [B, D] or [B, N*D]
        gt_norm = F.normalize(z_goal_gt.view(B, -1), dim=-1)
        neg_norm = F.normalize(negative_samples, dim=-1)  # [B, K, D]
        
        # Positive similarity (between pred and gt)
        pos_sim = (pred_norm * gt_norm).sum(dim=-1, keepdim=True) / temperature  # [B, 1]
        
        # Negative similarities
        # For patch-level, pool to single vector for contrastive
        if is_patch_level:
            pred_pooled = z_goal_pred.mean(dim=1)  # [B, D]
            pred_norm = F.normalize(pred_pooled, dim=-1)
        neg_sim = torch.bmm(neg_norm, pred_norm.unsqueeze(-1)).squeeze(-1) / temperature  # [B, K]
        
        # InfoNCE loss
        logits = torch.cat([pos_sim, neg_sim], dim=-1)  # [B, 1+K]
        labels = torch.zeros(B, dtype=torch.long, device=device)  # Positive is always index 0
        contrastive_loss = F.cross_entropy(logits, labels)
        losses["contrastive_loss"] = contrastive_loss
    else:
        losses["contrastive_loss"] = contrastive_loss
    
    # === Delta Norm Regularization ===
    delta_reg_loss = torch.tensor(0.0, device=device)
    if delta_reg_weight > 0 and z_init is not None:
        # Encourage minimal necessary change from initial state
        delta = z_goal_pred - z_init
        delta_norm = delta.norm(dim=-1).mean()
        
        # Also compute expected delta from GT
        gt_delta = z_goal_gt - z_init
        gt_delta_norm = gt_delta.norm(dim=-1).mean()
        
        # Regularize towards GT delta norm (don't over- or under-predict)
        delta_reg_loss = F.mse_loss(delta_norm, gt_delta_norm)
        losses["delta_reg_loss"] = delta_reg_loss
        losses["delta_norm"] = delta_norm
        losses["gt_delta_norm"] = gt_delta_norm
    else:
        losses["delta_reg_loss"] = delta_reg_loss
    
    # === Total Loss ===
    total_loss = (
        l2_weight * l2_loss +
        cosine_weight * cosine_loss +
        contrastive_weight * contrastive_loss +
        delta_reg_weight * delta_reg_loss
    )
    losses["total_loss"] = total_loss
    
    return losses


class PatchLevelGoalHead(nn.Module):
    """
    Patch-level Goal latent prediction head.
    
    Unlike the original GoalHead that outputs a single pooled vector [B, D],
    this version outputs patch-level predictions [B, N, D] to preserve
    spatial structure for planning.
    
    Uses cross-attention to condition each patch on the text instruction.
    """
    
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        output_dim: int,
        num_patches: int = 256,  # 16x16 for ViT-S/14
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize PatchLevelGoalHead.
        
        Args:
            visual_dim: Dimension of visual features from encoder (384 for DINOv2-S)
            text_dim: Dimension of text features from encoder (512 for CLIP ViT-B/32)
            output_dim: Dimension of output goal latent (should match visual_dim)
            num_patches: Number of patches (256 = 16x16 for ViT-S/14)
            hidden_dim: Hidden dimension for transformer layers
            num_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.num_patches = num_patches
        
        # Project visual and text to same hidden dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Learnable patch position embeddings
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)
        
        # Transformer layers with cross-attention
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Self-attention layers for refining patch predictions
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection back to output_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Delta prediction: predict change from initial patches
        self.predict_delta = True  # Predict delta from init instead of absolute
        
        logger.info(f"🧠 PatchLevelGoalHead initialized: "
                   f"visual_dim={visual_dim}, text_dim={text_dim}, "
                   f"output_dim={output_dim}, num_patches={num_patches}, "
                   f"num_layers={num_layers}")
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to predict patch-level goal latent.
        
        Args:
            visual_features: Visual features from encoder [B, N, D]
            text_features: Text features from encoder [B, text_dim]
            proprio: Optional proprioception state (not used in this version)
            
        Returns:
            z_goal_pred: Predicted goal latent [B, N, output_dim]
        """
        batch_size = visual_features.shape[0]
        
        # Handle different input shapes
        if visual_features.dim() == 6:
            # [B, T, V, H, W, D] -> [B, T*V*H*W, D]
            visual_features = rearrange(visual_features, "b t v h w d -> b (t v h w) d")
        elif visual_features.dim() == 2:
            # [B, D] -> [B, 1, D] (pooled input, expand)
            visual_features = visual_features.unsqueeze(1).expand(-1, self.num_patches, -1)
        
        # Store original for residual if predicting delta
        visual_orig = visual_features
        
        # Project to hidden dimension
        x = self.visual_proj(visual_features)  # [B, N, hidden_dim]
        
        # Add positional embeddings
        x = x + self.patch_pos_embed
        
        # Project text features
        if text_features.dim() == 2:
            t = self.text_proj(text_features).unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            t = self.text_proj(text_features)  # [B, L, hidden_dim]
        
        # Apply cross-attention and self-attention layers
        for cross_attn, self_attn in zip(self.layers, self.self_attn_layers):
            # Cross-attention: patches attend to text
            x = cross_attn(query=x, key_value=t)
            # Self-attention: patches attend to each other
            x = self_attn(x)
        
        # Project to output dimension
        delta = self.output_proj(x)  # [B, N, output_dim]
        
        if self.predict_delta:
            # Predict delta from initial patches
            z_goal = visual_orig + delta
        else:
            z_goal = delta
        
        return z_goal


def build_goal_head(config: Dict, visual_encoder=None, text_encoder=None) -> nn.Module:
    """
    Build GoalHead from config dictionary.
    
    Args:
        config: Configuration dictionary with GoalHead parameters
        visual_encoder: Optional visual encoder for wrapper
        text_encoder: Optional text encoder for wrapper
        
    Returns:
        GoalHead, ImprovedGoalHead, PatchLevelGoalHead, or GoalHeadWithEncoders module
    """
    # Determine which model variant to use
    model_version = config.get("version", "v1")  # "v1" (original) or "v2" (improved)
    
    if model_version == "v2" or config.get("use_improved", False):
        # Use ImprovedGoalHead with attention pooling, Q-Former, gated attention
        goal_head = ImprovedGoalHead(
            visual_dim=config.get("visual_dim", 384),
            text_dim=config.get("text_dim", 512),
            output_dim=config.get("output_dim", 384),
            proprio_dim=config.get("proprio_dim", 0),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 2),
            fusion_mode=config.get("fusion_mode", "qformer"),  # Default to qformer for v2
            num_attention_heads=config.get("num_attention_heads", 8),
            num_queries=config.get("num_queries", 16),  # Q-Former queries
            dropout=config.get("dropout", 0.1),
            use_proprio=config.get("use_proprio", False),
            use_gated_attn=config.get("use_gated_attn", True),  # Gated attention for stability
            film_hops=config.get("film_hops", 3),
        )
    elif config.get("patch_level", False):
        # Patch-level prediction
        goal_head = PatchLevelGoalHead(
            visual_dim=config.get("visual_dim", 384),
            text_dim=config.get("text_dim", 512),
            output_dim=config.get("output_dim", 384),
            num_patches=config.get("num_patches", 256),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 3),
            num_attention_heads=config.get("num_attention_heads", 8),
            dropout=config.get("dropout", 0.1),
        )
    else:
        # Original GoalHead (v1)
        goal_head = GoalHead(
            visual_dim=config.get("visual_dim", 384),
            text_dim=config.get("text_dim", 512),
            output_dim=config.get("output_dim", 384),
            proprio_dim=config.get("proprio_dim", 0),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 2),
            fusion_mode=config.get("fusion_mode", "concat"),
            num_attention_heads=config.get("num_attention_heads", 8),
            dropout=config.get("dropout", 0.1),
            use_proprio=config.get("use_proprio", False),
        )
    
    if visual_encoder is not None and text_encoder is not None:
        return GoalHeadWithEncoders(
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            goal_head=goal_head,
            freeze_visual=config.get("freeze_visual", True),
            freeze_text=config.get("freeze_text", True),
        )
    
    return goal_head
