# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Text encoder implementations for language-conditioned goal planning.

Provides wrapper classes for different text encoders with a unified interface.
"""

import logging
import sys
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoderWrapper(nn.Module):
    """Base wrapper class for text encoders."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.embed_dim = None
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode list of text strings to embeddings."""
        raise NotImplementedError
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass on tokenized input."""
        raise NotImplementedError
    
    def tokenize(self, texts: List[str]) -> torch.Tensor:
        """Tokenize text strings."""
        raise NotImplementedError


class SentenceTransformerEncoder(TextEncoderWrapper):
    """
    Wrapper for SentenceTransformers text encoders.
    
    SentenceTransformers provide high-quality sentence embeddings
    and are easier to use than CLIP for pure text tasks.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
        max_seq_length: int = 128,
    ):
        super().__init__(model_name, device)
        self.max_seq_length = max_seq_length
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.model.max_seq_length = max_seq_length
            self.embed_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Loaded SentenceTransformer: {model_name} (dim={self.embed_dim})")
            logger.info(f"   Source: SentenceTransformers library")
        except ImportError:
            raise ImportError(
                "SentenceTransformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode list of text strings to embeddings."""
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )
        return embeddings
    
    def forward(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """Forward pass - accepts either text strings or pre-tokenized input."""
        if isinstance(texts, list):
            return self.encode(texts)
        else:
            # If tensor input, assume it's already processed
            return texts
    
    def tokenize(self, texts: List[str]) -> Dict:
        """Tokenize text strings (returns dict for SentenceTransformers)."""
        return self.model.tokenize(texts)


class CLIPTextEncoder(TextEncoderWrapper):
    """
    Wrapper for CLIP text encoder.
    
    CLIP provides aligned vision-language embeddings which may be
    beneficial for multimodal understanding.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
    ):
        super().__init__(model_name, device)
        
        try:
            import clip
            self.clip = clip
            self.model, _ = clip.load(model_name, device=device)
            self.model.eval()
            # CLIP ViT-B/32 has 512 dim, ViT-L/14 has 768 dim
            self.embed_dim = self.model.text_projection.shape[1]
            logger.info(f"✅ Loaded CLIP text encoder: {model_name} (dim={self.embed_dim})")
            logger.info(f"   Source: OpenAI CLIP (via TorchHub)")
        except ImportError:
            raise ImportError(
                "CLIP is not installed. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            )
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode list of text strings to embeddings."""
        tokens = self.clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()
    
    def encode_ensemble(self, texts: List[str]) -> torch.Tensor:
        """
        Encode multiple texts and return their normalized mean embedding.
        
        This is useful for averaging multiple paraphrases of the same instruction
        to get a more robust text representation.
        
        Process:
        1. Encode each text to get embeddings
        2. L2-normalize each embedding (already done in encode())
        3. Compute mean of normalized embeddings
        4. L2-normalize the mean
        
        Args:
            texts: List of text strings (e.g., multiple paraphrases)
            
        Returns:
            Normalized mean embedding [1, embed_dim]
        """
        # Encode all texts (each is already L2-normalized)
        embeddings = self.encode(texts)  # [N, embed_dim]
        
        # Compute mean
        mean_embedding = embeddings.mean(dim=0, keepdim=True)  # [1, embed_dim]
        
        # Re-normalize the mean
        mean_embedding = mean_embedding / mean_embedding.norm(dim=-1, keepdim=True)
        
        return mean_embedding
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass on tokenized input."""
        if tokens.device != self.device:
            tokens = tokens.to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()
    
    def tokenize(self, texts: List[str]) -> torch.Tensor:
        """Tokenize text strings."""
        return self.clip.tokenize(texts)


class HuggingFaceCLIPEncoder(TextEncoderWrapper):
    """
    CLIP text encoder using HuggingFace transformers library.
    
    This is an alternative to the OpenAI CLIP implementation.
    Uses models like 'openai/clip-vit-base-patch32'.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ):
        super().__init__(model_name, device)
        
        try:
            from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            self.embed_dim = self.model.config.projection_dim
            logger.info(f"✅ Loaded HuggingFace CLIP: {model_name} (dim={self.embed_dim})")
        except ImportError:
            raise ImportError(
                "transformers is not installed. "
                "Install with: pip install transformers"
            )
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode list of text strings to embeddings."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            # Use text_model and text_projection for proper CLIP embeddings
            text_outputs = self.model.text_model(**inputs)
            text_features = self.model.text_projection(text_outputs.pooler_output)
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()
    
    def encode_ensemble(self, texts: List[str]) -> torch.Tensor:
        """
        Encode multiple texts and return their normalized mean embedding.
        """
        embeddings = self.encode(texts)  # [N, embed_dim]
        mean_embedding = embeddings.mean(dim=0, keepdim=True)  # [1, embed_dim]
        mean_embedding = mean_embedding / mean_embedding.norm(dim=-1, keepdim=True)
        return mean_embedding
    
    def forward(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        if isinstance(texts, list):
            return self.encode(texts)
        return texts
    
    def tokenize(self, texts: List[str]) -> Dict:
        """Tokenize text strings."""
        return self.processor(text=texts, return_tensors="pt", padding=True)


class SigLIP2Encoder(TextEncoderWrapper):
    """
    SigLIP 2 text encoder using HuggingFace transformers library.
    
    SigLIP 2 provides improved vision-language alignment with better
    multilingual support and enhanced performance on fine-grained tasks.
    Uses models like 'google/siglip2-large-patch16-512'.
    
    Key differences from CLIP:
    - Uses sigmoid loss instead of softmax (no need for temperature)
    - Better multilingual support
    - Generally higher quality embeddings for fine-grained tasks
    - Default max_position_embeddings=64 (shorter than CLIP's 77)
    
    NOTE: 
    - For non-naflex models: Uses SiglipModel + Siglip2Tokenizer
    - For naflex models: Uses Siglip2Model + Siglip2Tokenizer
    - Siglip2Tokenizer handles lowercasing internally, so external lowercase is disabled by default
    """
    
    def __init__(
        self,
        model_name: str = "google/siglip2-large-patch16-512",
        device: str = "cuda",
        lowercase: bool = False,  # Siglip2Tokenizer handles lowercasing internally
        max_length: int = 64,
    ):
        super().__init__(model_name, device)
        self.lowercase = lowercase
        self.max_length = max_length
        
        try:
            from transformers import Siglip2Tokenizer
            
            # Tokenizer is always Siglip2Tokenizer (handles lowercasing internally)
            self.tokenizer = Siglip2Tokenizer.from_pretrained(model_name)
            
            # Model class depends on whether it's naflex or not
            # naflex models use Siglip2Model, others use SiglipModel (siglip v1 compatible)
            is_naflex = "-naflex" in model_name.lower()
            
            if is_naflex:
                from transformers import Siglip2Model
                self.model = Siglip2Model.from_pretrained(model_name).to(device)
                model_class_name = "Siglip2Model"
            else:
                from transformers import SiglipModel
                self.model = SiglipModel.from_pretrained(model_name).to(device)
                model_class_name = "SiglipModel"
            
            self.model.eval()
            
            # Get embedding dimension from config
            # SiglipModel uses text_config.hidden_size as the output dim
            if hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'hidden_size'):
                self.embed_dim = self.model.config.text_config.hidden_size
            elif hasattr(self.model.config, 'projection_dim'):
                self.embed_dim = self.model.config.projection_dim
            else:
                # Fallback: run a forward pass to determine dim
                with torch.no_grad():
                    test_inputs = self.tokenizer(
                        ["test"], 
                        return_tensors="pt", 
                        padding=True,
                        max_length=self.max_length,
                        truncation=True,
                    )
                    test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
                    test_output = self.model.get_text_features(**test_inputs)
                    # Handle both tensor and ModelOutput returns
                    if hasattr(test_output, 'pooler_output'):
                        test_output = test_output.pooler_output
                    self.embed_dim = test_output.shape[-1]
            
            logger.info(f"✅ Loaded SigLIP2: {model_name} (dim={self.embed_dim})")
            logger.info(f"   Model class: {model_class_name}, Tokenizer: Siglip2Tokenizer")
            logger.info(f"   Max length: {self.max_length}")
            if self.lowercase:
                logger.info(f"   External lowercase: enabled (note: tokenizer also lowercases)")
        except ImportError as e:
            raise ImportError(
                f"Failed to import SigLIP2 classes: {e}. "
                "Install/upgrade with: pip install -U transformers>=4.56 sentencepiece protobuf"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SigLIP2 model '{model_name}': {e}. "
                "Make sure you have the latest transformers installed: "
                "pip install -U git+https://github.com/huggingface/transformers"
            )
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts (lowercase if enabled)."""
        if self.lowercase:
            return [t.lower() for t in texts]
        return texts
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode list of text strings to embeddings."""
        texts = self._preprocess_texts(texts)
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use get_text_features for SigLIP2 text embeddings
            text_features = self.model.get_text_features(**inputs)
            # Handle both tensor and ModelOutput returns (SiglipModel returns ModelOutput)
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
        
        # Normalize features (L2 normalize like CLIP)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()
    
    def encode_ensemble(self, texts: List[str]) -> torch.Tensor:
        """
        Encode multiple texts and return their normalized mean embedding.
        
        This is useful for averaging multiple paraphrases of the same instruction
        to get a more robust text representation.
        
        Process:
        1. Encode each text to get embeddings
        2. L2-normalize each embedding (already done in encode())
        3. Compute mean of normalized embeddings
        4. L2-normalize the mean
        
        Args:
            texts: List of text strings (e.g., multiple paraphrases)
            
        Returns:
            Normalized mean embedding [1, embed_dim]
        """
        # Encode all texts (each is already L2-normalized)
        embeddings = self.encode(texts)  # [N, embed_dim]
        
        # Compute mean
        mean_embedding = embeddings.mean(dim=0, keepdim=True)  # [1, embed_dim]
        
        # Re-normalize the mean
        mean_embedding = mean_embedding / mean_embedding.norm(dim=-1, keepdim=True)
        
        return mean_embedding
    
    def forward(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """Forward pass - accepts either text strings or pre-tokenized input."""
        if isinstance(texts, list):
            return self.encode(texts)
        else:
            # If tensor input, assume it's already processed
            return texts
    
    def tokenize(self, texts: List[str]) -> Dict:
        """Tokenize text strings."""
        texts = self._preprocess_texts(texts)
        return self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_length,
        )


class SimpleTextEncoder(TextEncoderWrapper):
    """
    Simple learned text encoder using word embeddings.
    
    This is a lightweight fallback option that doesn't require
    external pretrained models.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 384,
        hidden_dim: int = 512,
        max_seq_length: int = 64,
        device: str = "cuda",
    ):
        super().__init__("simple", device)
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Simple word embedding + LSTM encoder
        self.word_embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.output_proj = nn.Linear(hidden_dim * 2, embed_dim)
        
        # Simple tokenizer (word-based)
        self._build_vocab()
        
        self.to(device)
        logger.info(f"✅ Initialized SimpleTextEncoder (dim={embed_dim})")
        logger.info(f"   Source: Trained from scratch")
    
    def _build_vocab(self):
        """Build a simple vocabulary."""
        # Common words for robotics instructions
        base_words = [
            "<pad>", "<unk>", "<sos>", "<eos>",
            "the", "a", "to", "of", "and", "in", "on", "at",
            "push", "pull", "open", "close", "pick", "place", "reach",
            "door", "drawer", "button", "window", "handle", "lever",
            "object", "target", "position", "location", "goal",
            "from", "into", "towards", "away", "up", "down", "left", "right",
            "slide", "press", "turn", "rotate", "move", "grasp", "release",
        ]
        self.word2idx = {word: i for i, word in enumerate(base_words)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
    
    def tokenize(self, texts: List[str]) -> torch.Tensor:
        """Tokenize text strings."""
        batch_tokens = []
        for text in texts:
            words = text.lower().replace(",", " ").replace(".", " ").split()
            tokens = [self.word2idx.get(w, 1) for w in words[:self.max_seq_length]]
            # Pad to max length
            tokens = tokens + [0] * (self.max_seq_length - len(tokens))
            batch_tokens.append(tokens)
        return torch.tensor(batch_tokens, device=self.device)
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode list of text strings to embeddings."""
        tokens = self.tokenize(texts)
        return self.forward(tokens)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass on tokenized input."""
        if tokens.device != self.device:
            tokens = tokens.to(self.device)
        
        # Embed tokens
        x = self.word_embed(tokens)  # [B, L, hidden_dim]
        
        # LSTM encoding
        output, (hidden, _) = self.lstm(x)  # hidden: [2*2, B, hidden_dim]
        
        # Use final hidden states (forward + backward)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # [B, hidden_dim*2]
        
        # Project to output dimension
        embeddings = self.output_proj(hidden)  # [B, embed_dim]
        
        return embeddings


def build_text_encoder(config: Dict) -> TextEncoderWrapper:
    """
    Build text encoder from config.
    
    Args:
        config: Configuration dict with keys:
            - type: "sentence_transformer", "clip", "huggingface_clip", "siglip2", or "simple"
            - model_name: Model name for the encoder
            - device: Device to load on
            
    Returns:
        TextEncoderWrapper instance
    """
    encoder_type = config.get("type", "sentence_transformer")
    device = config.get("device", "cuda")
    
    if encoder_type == "sentence_transformer":
        model_name = config.get("model_name", "all-MiniLM-L6-v2")
        return SentenceTransformerEncoder(
            model_name=model_name,
            device=device,
            max_seq_length=config.get("max_seq_length", 128),
        )
    elif encoder_type == "clip":
        model_name = config.get("model_name", "ViT-B/32")
        return CLIPTextEncoder(model_name=model_name, device=device)
    elif encoder_type == "huggingface_clip":
        model_name = config.get("model_name", "openai/clip-vit-base-patch32")
        return HuggingFaceCLIPEncoder(model_name=model_name, device=device)
    elif encoder_type == "siglip2":
        model_name = config.get("model_name", "google/siglip2-large-patch16-512")
        lowercase = config.get("lowercase", False)  # Siglip2Tokenizer handles lowercasing
        return SigLIP2Encoder(model_name=model_name, device=device, lowercase=lowercase)
    elif encoder_type == "simple":
        return SimpleTextEncoder(
            embed_dim=config.get("embed_dim", 384),
            device=device,
        )
    else:
        raise ValueError(f"Unknown text encoder type: {encoder_type}")
