"""
Nano Wave model for Nano Binary Networks.

This module provides a wave-based model for Nano Binary Networks,
implementing the BaseWaveModel interface.
"""

import time
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from torch import Tensor, tensor, float32
from config import WaveConfig
from logger import get_logger
from utils.dictionary_loader import DictionaryLoader
from models.base_wave import BaseWaveModel


class NanoWaveModel(BaseWaveModel):
    """
    Nano Wave model for Nano Binary Networks.

    This class implements a wave-based model for Nano Binary Networks,
    using the BaseWaveModel interface.
    """

    def __init__(self, config: Optional[WaveConfig] = None):
        """
        Initialize the Nano Wave model.

        Args:
            config: Configuration for the wave model. If None, default configuration is used.
        """
        super().__init__(config)

        # Initialize parameters
        self.psi = tensor(0.0, dtype=float32)  # Phase offset
        self.beta = tensor(0.05, dtype=float32)  # Damping factor
        self.A = tensor(600000, dtype=float32)  # Amplitude
        self.f_max = tensor(200, dtype=float32)  # Maximum frequency

        # Initialize lookup table
        self.lookup_table = self._build_lookup_table()

        self.logger.info(f"Initialized NanoWaveModel with parameters: psi={self.psi.item()}, beta={self.beta.item()}, A={self.A.item()}, f_max={self.f_max.item()}")

    def _build_lookup_table(self) -> Dict[str, Tensor]:
        """
        Build a lookup table mapping characters to frequencies.

        Returns:
            Dictionary mapping characters to frequency tensors.
        """
        lookup_table = {}
        for code in range(256):
            char = chr(code) if 32 <= code <= 126 else ''
            if char:
                lookup_table[char] = self._get_frequency(code)

        self.logger.info(f"Built lookup table with {len(lookup_table)} characters")
        return lookup_table

    def _get_frequency(self, code: int) -> Tensor:
        """
        Get the frequency for a character code using a logarithmic scale.

        Args:
            code: ASCII code of the character.

        Returns:
            Tensor representing the frequency.
        """
        return torch.tensor(min(1.0 * (2 ** (code / 32.0)), self.f_max.item()), dtype=float32)

    def train_on_text(self, text: str) -> None:
        """
        Adjust frequencies based on text co-occurrences.

        Args:
            text: Text to train on.
        """
        start_time = time.time()
        self.logger.info(f"Training on text of length {len(text)}")

        # Simple bigram count
        text = text.lower()
        bigrams = [(text[i], text[i + 1]) for i in range(len(text) - 1) if
                   text[i] in self.lookup_table and text[i + 1] in self.lookup_table]

        co_occur = torch.zeros((len(self.lookup_table), len(self.lookup_table)), dtype=float32)
        char_idx = {c: i for i, c in enumerate(self.lookup_table.keys())}

        for a, b in bigrams:
            co_occur[char_idx[a], char_idx[b]] += 1

        # Adjust frequencies: Shift by 0.001 * count / max for "learning"
        max_co = co_occur.max().item()
        if max_co > 0:
            for a in self.lookup_table:
                for b in self.lookup_table:
                    if co_occur[char_idx[a], char_idx[b]] > 0:
                        shift = tensor(0.001, dtype=float32) * (co_occur[char_idx[a], char_idx[b]] / max_co)
                        self.lookup_table[b] += shift  # Simulate resonance
                        self.lookup_table[b] = torch.min(self.lookup_table[b], self.f_max)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Training complete in {elapsed_time:.2f}s: Adjusted frequencies based on {len(bigrams)} co-occurrences")

        # Log performance metrics
        self.log_performance("training", {
            "text_length": len(text),
            "bigrams": len(bigrams),
            "max_co_occurrence": max_co,
            "training_time": elapsed_time
        })

    def compute_wave(self, freqs: Tensor, amplitudes: Optional[Tensor] = None) -> Tensor:
        """
        Compute the wave function for the given frequencies.

        Args:
            freqs: Tensor of frequencies.
            amplitudes: Optional tensor of amplitudes. If None, uses self.A for all frequencies.

        Returns:
            Tensor representing the wave function.
        """
        start_time = time.time()

        # Use default amplitude if not provided
        if amplitudes is None:
            amplitudes = torch.full_like(freqs, self.A.item())

        # Initialize wave
        u = torch.zeros_like(self.t, dtype=float32)

        # Log input parameters
        self.logger.info(f"Computing wave with {freqs.numel()} frequencies, range: [{freqs.min().item():.2f}, {freqs.max().item():.2f}]")

        # Compute wave contributions
        for i, f in enumerate(freqs):
            # Damping relative to first frequency
            d_i = self.beta * torch.abs(f - freqs[0])

            # Wave contribution
            wave_contrib = amplitudes[i] * torch.cos(2 * torch.pi * f * self.t + self.psi) * torch.exp(-d_i * self.t)

            # Add to total wave
            u += wave_contrib

        elapsed_time = time.time() - start_time
        self.logger.info(f"Wave computed in {elapsed_time:.4f}s, u shape: {u.shape}, sample value: {u[0].item():.8f}")

        # Log performance metrics
        self.log_performance("wave_computation", {
            "num_frequencies": freqs.numel(),
            "min_frequency": freqs.min().item(),
            "max_frequency": freqs.max().item(),
            "computation_time": elapsed_time
        })

        return u

    def get_word_frequencies(self, word: str) -> Tensor:
        """
        Get frequencies for characters in a word.

        Args:
            word: Word to get frequencies for.

        Returns:
            Tensor of frequencies.
        """
        return torch.tensor([self.lookup_table[c.lower()] for c in word if c.lower() in self.lookup_table], dtype=float32)

    def analyze_word_pair(self, base: str, context: str) -> Dict[str, Any]:
        """
        Analyze interference between a base word and a context word.

        Args:
            base: Base word.
            context: Context word or phrase.

        Returns:
            Dictionary of analysis results.
        """
        start_time = time.time()

        # Get frequencies for base and context
        freqs1 = self.get_word_frequencies(base)

        # For context, use the last word if it's a phrase
        context_word = context.split()[-1]
        freqs2 = self.get_word_frequencies(context_word)

        results = {}

        if freqs1.numel() > 0 and freqs2.numel() > 0:
            # Compute waves
            u1 = self.compute_wave(freqs1)
            u2 = self.compute_wave(freqs2)

            # Log interference
            interference_metrics = self.log_interference(u1, u2, f"{base} vs {context}", freqs1, freqs2)

            # Store results
            results = {
                "base_word": base,
                "context_word": context_word,
                "base_frequencies": freqs1.tolist(),
                "context_frequencies": freqs2.tolist(),
                "interference_metrics": interference_metrics,
                "analysis_time": time.time() - start_time
            }
        else:
            self.logger.warning(f"Could not analyze {base} vs {context}: insufficient valid characters")
            results = {
                "base_word": base,
                "context_word": context_word,
                "error": "insufficient valid characters",
                "analysis_time": time.time() - start_time
            }

        return results

    def run_experiment(self, text: str, word_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Run an experiment with the given text and word pairs.

        Args:
            text: Text to train on.
            word_pairs: List of (base, context) word pairs to analyze.

        Returns:
            Dictionary of experiment results.
        """
        start_time = time.time()
        self.logger.info(f"Starting experiment with {len(word_pairs)} word pairs")

        # Train on text
        self.train_on_text(text)

        # Analyze word pairs
        results = []
        for base, context in word_pairs:
            result = self.analyze_word_pair(base, context)
            results.append(result)

        # Compile experiment results
        experiment_results = {
            "experiment_name": "nano_wave_experiment",
            "text_length": len(text),
            "num_word_pairs": len(word_pairs),
            "word_pair_results": results,
            "total_time": time.time() - start_time
        }

        # Log experiment results
        self.log_performance("experiment", experiment_results)

        # Save all metrics
        metrics_path = self.save_metrics()
        self.logger.info(f"Experiment complete. Metrics saved to {metrics_path}")

        return experiment_results
