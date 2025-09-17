"""
Dictionary loader for Nano Binary Networks.

This module provides utilities for loading and processing dictionaries
for use in Nano Binary Networks.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import requests

from config import DictionaryConfig
from logger import get_logger


class DictionaryLoader:
    """
    Utility for loading and processing dictionaries for Nano Binary Networks.

    This class provides methods for loading dictionaries from various sources,
    caching them for faster access, and processing them for use in models.
    """

    def __init__(self, config: Optional[DictionaryConfig] = None):
        """
        Initialize the dictionary loader.

        Args:
            config: Configuration for dictionary loading. If None, default configuration is used.
        """
        self.config = config or DictionaryConfig()
        self.logger = get_logger()
        self.logger.info(f"Initializing DictionaryLoader with config: {self.config}")

        # Create cache directory if it doesn't exist
        if self.config.use_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)

    def load_words(self, limit: Optional[int] = None) -> List[str]:
        """
        Load words from the configured source.

        Args:
            limit: Maximum number of words to load. If None, uses the limit from config.

        Returns:
            List of words.
        """
        limit = limit or self.config.limit
        self.logger.info(f"Loading up to {limit} words")

        # Check cache first if enabled
        if self.config.use_cache:
            cache_file = self._get_cache_path(limit)
            if os.path.exists(cache_file):
                self.logger.info(f"Loading words from cache: {cache_file}")
                return self._load_from_cache(cache_file)

        # Load from URL
        words = self._load_from_url(self.config.url, limit)

        # Cache the result if enabled
        if self.config.use_cache:
            cache_file = self._get_cache_path(limit)
            self._save_to_cache(words, cache_file)

        return words

    def _get_cache_path(self, limit: int) -> str:
        """
        Get the path to the cache file for the given limit.

        Args:
            limit: Maximum number of words.

        Returns:
            Path to the cache file.
        """
        # Create a hash of the URL to use in the cache filename
        url_hash = hashlib.md5(self.config.url.encode()).hexdigest()[:8]
        return os.path.join(self.config.cache_dir, f"words_{url_hash}_{limit}.json")

    def _load_from_cache(self, cache_file: str) -> List[str]:
        """
        Load words from a cache file.

        Args:
            cache_file: Path to the cache file.

        Returns:
            List of words.
        """
        try:
            with open(cache_file, 'r') as f:
                words = json.load(f)
            self.logger.info(f"Loaded {len(words)} words from cache")
            return words
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return []

    def _save_to_cache(self, words: List[str], cache_file: str) -> None:
        """
        Save words to a cache file.

        Args:
            words: List of words to save.
            cache_file: Path to the cache file.
        """
        try:
            with open(cache_file, 'w') as f:
                json.dump(words, f)
            self.logger.info(f"Saved {len(words)} words to cache: {cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")

    def _load_from_url(self, url: str, limit: int) -> List[str]:
        """
        Load words from a URL.

        Args:
            url: URL to load words from.
            limit: Maximum number of words to load.

        Returns:
            List of words.
        """
        self.logger.info(f"Loading words from URL: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Process the response based on content type
            content_type = response.headers.get('content-type', '')
            if 'json' in content_type:
                # JSON response
                data = response.json()
                if isinstance(data, list):
                    words = data
                elif isinstance(data, dict):
                    # Try to extract words from common dictionary formats
                    if 'words' in data:
                        words = data['words']
                    else:
                        words = list(data.keys())
                else:
                    words = []
            else:
                # Text response (one word per line)
                words = [line.strip() for line in response.text.splitlines() if line.strip()]

            # Filter and limit
            words = [word.lower() for word in words if word.strip() and word.isalpha()]
            words = words[:limit]

            self.logger.info(f"Loaded {len(words)} words from URL")
            return words
        except Exception as e:
            self.logger.error(f"Failed to load words from URL: {e}")
            return []

    def get_word_frequencies(self, words: List[str]) -> Dict[str, float]:
        """
        Calculate normalized frequencies for words based on their length and position.

        Args:
            words: List of words to calculate frequencies for.

        Returns:
            Dictionary mapping words to their frequencies.
        """
        self.logger.info(f"Calculating frequencies for {len(words)} words")

        # Simple frequency calculation based on word length and position
        frequencies = {}
        max_length = max(len(word) for word in words) if words else 1

        for i, word in enumerate(words):
            # Normalize by position (earlier words get higher frequency)
            position_factor = 1.0 - (i / len(words)) if words else 0

            # Normalize by length (shorter words get higher frequency)
            length_factor = 1.0 - (len(word) / max_length)

            # Combine factors (adjust weights as needed)
            frequency = 0.7 * position_factor + 0.3 * length_factor

            # Scale to desired range (e.g., 0.1 to 200.0)
            min_freq, max_freq = self.config.frequency_range if hasattr(self.config, 'frequency_range') else (0.1, 200.0)
            scaled_frequency = min_freq + frequency * (max_freq - min_freq)

            frequencies[word] = scaled_frequency

        return frequencies
