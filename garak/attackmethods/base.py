# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes for attack methods"""

from abc import ABC, abstractmethod
from typing import Iterable, List

from garak.attempt import Conversation, Attempt
from garak.generators import Generator


class AttackMethod(ABC):
    """A way for manipulating conversations so that the input Generator doesn't block the request"""

    @abstractmethod
    def attack(self, prompts: List[Conversation], generator: Generator) -> Iterable[Attempt]:
        """Apply attack method manipulation to prompts and execute against generator.

        This method takes a list of base prompts (conversations) and applies the attack
        method's transformation logic to bypass safety mechanisms. The modified prompts
        are then executed against the provided generator to produce attempts.

        Args:
            prompts: List of base conversations to attack. Each conversation typically
                    contains system prompts and user prompts that need to be modified
                    to bypass safety controls.
            generator: The generator (LLM) to test against. This is used to both
                      potentially transform prompts dynamically and to execute the
                      attacked prompts.

        Returns:
            List of Attempt objects representing the results of executing the attacked
            prompts against the generator. Each attempt contains the original and
            potentially modified prompts, along with the generator's response.
        """
        return []
