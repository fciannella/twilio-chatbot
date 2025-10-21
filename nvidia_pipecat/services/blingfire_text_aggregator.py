#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Blingfire-based text aggregator for sentence detection."""

import blingfire as bf
from loguru import logger
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


class BlingfireTextAggregator(BaseTextAggregator):
    """This is a text aggregator that uses blingfire for sentence detection.

    It aggregates text until a complete sentence is detected using blingfire's
    sentence segmentation capabilities.
    """

    def __init__(self):
        """Initialize the BlingfireTextAggregator with an empty text buffer."""
        self._text = ""
        logger.debug("BlingfireTextAggregator: Initialized new instance")

    @property
    def text(self) -> str:
        """Return the currently aggregated text.

        Returns:
            str: The text currently being aggregated.
        """
        return self._text

    async def aggregate(self, text: str) -> str | None:
        """Aggregate text and return a complete sentence when detected.

        Args:
            text (str): The text to be aggregated.

        Returns:
            str | None: A complete sentence if one is detected, otherwise None.
        """
        result: str | None = None

        self._text += text
        logger.debug(f"BlingfireTextAggregator: Aggregating text: '{self._text}'")
        # Use blingfire to split text into sentences
        sentences_text = bf.text_to_sentences(self._text)
        sentences = [s.strip() for s in sentences_text.split("\n") if s.strip()]

        # If we have multiple sentences, return the first complete one
        if len(sentences) > 1:
            result = sentences[0]
            if self._text.lstrip().startswith(sentences[0].strip()):
                self._text = self._text[len(sentences[0]) :]

        return result

    async def handle_interruption(self):
        """Handle interruption by clearing the aggregated text buffer."""
        logger.debug("BlingfireTextAggregator: Handling interruption")
        self._text = ""

    async def reset(self):
        """Reset the aggregated text buffer to empty."""
        logger.debug("BlingfireTextAggregator: Resetting text buffer")
        self._text = ""
