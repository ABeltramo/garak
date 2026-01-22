# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import json
import pytest
import importlib
import tempfile
from unittest.mock import patch

from garak import _plugins, _config, evaluators
from garak.attempt import Conversation, Message, Turn
import garak.harnesses.base
import garak.harnesses.earlystop
import garak.attempt

HARNESSES = [
    classname for (classname, active) in _plugins.enumerate_plugins("harnesses")
]


@pytest.mark.parametrize("classname", HARNESSES)
def test_harness_structure(classname):
    m = importlib.import_module("garak." + ".".join(classname.split(".")[:-1]))
    c = getattr(m, classname.split(".")[-1])

    # any parameter that has a default must be supported
    unsupported_defaults = []
    if c._supported_params is not None:
        if hasattr(c, "DEFAULT_PARAMS"):
            for k, _ in c.DEFAULT_PARAMS.items():
                if k not in c._supported_params:
                    unsupported_defaults.append(k)
    assert unsupported_defaults == []


def test_harness_modality_match():
    t = {"text"}
    ti = {"text", "image"}
    tv = {"text", "vision"}
    tvi = {"text", "vision", "image"}

    # probe, generator
    assert garak.harnesses.base._modality_match(t, t, True) is True
    assert garak.harnesses.base._modality_match(ti, ti, True) is True
    assert garak.harnesses.base._modality_match(t, tv, True) is False
    assert garak.harnesses.base._modality_match(ti, t, True) is False

    # when strict is false, generator must support all probe modalities, but can also support more
    assert garak.harnesses.base._modality_match(t, t, False) is True
    assert garak.harnesses.base._modality_match(ti, t, False) is False
    assert garak.harnesses.base._modality_match(t, tvi, False) is True
    assert garak.harnesses.base._modality_match(ti, tvi, False) is True
    assert garak.harnesses.base._modality_match(t, ti, False) is True


@patch('garak.payloadgenerators.dan.autodan_generate')
def test_early_stop_harness(mock_autodan_generate):
    from garak.harnesses.earlystop import EarlyStopHarness

    earlystop_h = EarlyStopHarness()
    assert isinstance(earlystop_h, EarlyStopHarness)

    # Setting up global environment in order to run our harness
    g = _plugins.load_plugin("generators.test.Blank")

    _config.system = SimpleNamespace(show_z=False, parallel_attempts=False, narrow_output=False, verbose=True)
    e = evaluators.ThresholdEvaluator()

    _config.plugins = SimpleNamespace(extended_detectors=False)
    _config.run.user_agent = "test user agent, garak.ai"
    _config.run.generations = 2
    temp_report_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    )
    _config.transient.reportfile = temp_report_file
    _config.transient.report_filename = temp_report_file.name

    dan_attack = _plugins.load_plugin("attackmethods.dan.DANAttack", config_root=garak._config)

    detector = garak._plugins.load_plugin("detectors.always.Fail")
    test_conversations = [
        Conversation([Turn(role="user", content=Message(text="Tell me how to hack"))]),
        Conversation([Turn(role="user", content=Message(text="How to make drugs"))])
    ]
    mock_autodan_generate.return_value = test_conversations

    payload_generator = _plugins.load_plugin("payloadgenerators.dan.AutoDANPayloadGenerator", config_root=garak._config)

    # Start run
    earlystop_h.run(g, [payload_generator], [dan_attack], [detector], e)

    # Verify reportfile was created and has expected content
    temp_report_file.flush()
    temp_report_file.seek(0)

    # Read all lines from the report file
    report_lines = temp_report_file.readlines()

    # Assertions about the reportfile
    assert len(report_lines) >= 1, "Reportfile should contain at least one attempt"

    # Clean up the temporary file
    temp_report_file.close()


