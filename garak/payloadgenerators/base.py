# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes for prompt generators"""

from abc import ABC, abstractmethod
from typing import List

from garak.attempt import Conversation


class PayloadGenerator(ABC):
    """Base class for payload generators that produce harmful prompts for probes"""

    @abstractmethod
    def generate_prompts(self, generator=None) -> List[Conversation]:
        """Generate a list of payloads

        Args:
            generator: The generator object that may be used for dynamic prompt generation

        Returns:
            List of conversations
        """
        return []
