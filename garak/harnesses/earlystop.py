# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early stopping harness

The idea here is to start with one (or more) payload generators and then test out the input payloads
with a set of attack methods. After each round of test we carry over all the attempts that have been rejected
by the LLM until we have no more payloads to test, or we run out of available attack methods.
"""
import json
import logging
from typing import List

import tqdm

from garak.attackmethods import AttackMethod
from garak.detectors import Detector
from garak.evaluators import Evaluator
from garak.generators import Generator
from garak.harnesses import Harness
from garak.payloadgenerators import PayloadGenerator
from garak import _config
from garak.attempt import ATTEMPT_COMPLETE, Conversation, Attempt, ATTEMPT_STARTED


class EarlyStopHarness(Harness):

    def create_attempt(self, conversation: Conversation) -> Attempt:
        new_attempt = Attempt(
            probe_classname=(
                    str(self.__class__.__module__).replace("garak.probes.", "")
                    + "."
                    + self.__class__.__name__
            ),
            goal="",  # TODO: goal? Should this come from the payload_generator?
            status=ATTEMPT_STARTED,
            prompt=conversation,
        )
        return new_attempt

    def run(self, model: Generator,
            payload_generators: List[PayloadGenerator],
            attack_methods: List[AttackMethod],
            detectors: List[Detector],
            evaluator: Evaluator,
            buff_names=None):
        """First, use the PayloadGenerator to generate a list of conversations to be tested.
        Then, iterate over the attack_methods and pass the conversations that have been rejected by the LLM to them.
        The process finishes when we don't have any more attack methods or all the conversations have been classified as
        accepted.
        """

        if not payload_generators:
            msg = "No payload generators, nothing to do"
            logging.warning(msg)
            if hasattr(_config.system, "verbose") and _config.system.verbose >= 2:
                print(msg)
            raise ValueError(msg)

        if not attack_methods:
            msg = "No attack methods, nothing to do"
            logging.warning(msg)
            if hasattr(_config.system, "verbose") and _config.system.verbose >= 2:
                print(msg)
            raise ValueError(msg)

        if not buff_names:
            buff_names = []

        self._load_buffs(buff_names)
        self._start_run_hook()

        # Generate initial payloads from all payload generators
        all_conversations = []
        for payload_generator in payload_generators:
            try:
                conversations = payload_generator.generate_prompts(model)
                all_conversations.extend(conversations)
                logging.info(f"Generated {len(conversations)} payloads from {payload_generator.__class__.__name__}")
            except Exception as e:
                logging.error(f"Failed to generate payloads from {payload_generator.__class__.__name__}: {e}")
                continue

        if not all_conversations:
            logging.warning("No conversations generated, nothing to test")
            self._end_run_hook()
            return

        # Convert conversations to attempts for the first round
        accepted_attempts = []
        rejected_attempts = [self.create_attempt(conversation) for conversation in all_conversations]

        # Apply attack methods in sequence
        for attack_method in attack_methods:
            if not rejected_attempts:
                logging.info("No rejected attempts left, stopping early")
                break

            logging.info(f"Applying {attack_method.__class__.__name__} to {len(rejected_attempts)} rejected attempts")

            try:
                # Apply attack method to rejected attempts
                conversations = [attempt.prompt for attempt in rejected_attempts]
                attacked_attempts = list(attack_method.attack(conversations, model))
                rejected_attempts = []

                # Process the results
                for d in detectors:
                    logging.debug("harness: run detector %s", d.detectorname)
                    attempt_iterator = tqdm.tqdm(attacked_attempts, leave=False)
                    detector_probe_name = d.detectorname.replace("garak.detectors.", "")
                    attempt_iterator.set_description("detectors." + detector_probe_name)
                    for attempt in attempt_iterator:
                        if d.skip:
                            continue
                        result = d.detect(attempt)
                        attempt.detector_results[detector_probe_name] = list(result)

                        evaluations = [evaluator.test(float(score)) for score in result if score is not None]
                        # If at least one detector passes, we accept the attempt
                        if any(evaluations):
                            accepted_attempts.append(attempt)
                        else:
                            rejected_attempts.append(attempt)


            except Exception as e:
                logging.error(f"Attack method {attack_method.__class__.__name__} failed: {e}")
                # Continue with rejected attempts for next attack method
                continue

        # Update all attempts to completed status
        for attempt in accepted_attempts + rejected_attempts:
            attempt.status = ATTEMPT_COMPLETE
            _config.transient.reportfile.write(json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n")

        self._end_run_hook()
        logging.info(
            f"Early stopping harness completed: {len(accepted_attempts)} accepted, {len(rejected_attempts)} rejected")
