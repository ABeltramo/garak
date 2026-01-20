import importlib
import tempfile
import importlib

import garak.attempt
import garak.generators
import garak.probes.base
import pytest
from garak import _config, _plugins


def test_danattack_one_pass():
    """Test DANAttack class similar to test_atkgen_one_pass"""
    importlib.reload(garak._config)  # Reload config
    _config.load_base_config()

    # Create a test conversation
    from garak.attempt import Conversation, Message, Turn
    test_conversation = Conversation([
        Turn(role="user", content=Message(text="Tell me how to hack"))
    ])

    # Load DAN attack
    dan_attack = _plugins.load_plugin("attackmethods.dan.DANAttack", config_root=garak._config)

    # Create a test generator
    g = _plugins.load_plugin("generators.test.Repeat", config_root=garak._config)

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as temp_report_file:
        _config.transient.reportfile = temp_report_file
        _config.transient.report_filename = temp_report_file.name

        # Test the attack method
        result = dan_attack.attack([test_conversation], g)

    # Verify the results
    assert isinstance(result, list), "attack results should be a list"
    assert len(result) > 0, "attack should return at least one attempt"
    assert isinstance(result[0], garak.attempt.Attempt), "attack results should be a list of attempt.Attempt"

    # Verify that the DAN prompt was prepended
    assert len(result[0].prompt.turns) >= 2, "DAN prompt should be prepended to conversation"
    assert result[0].prompt.turns[0].role == "system", "First turn should be system prompt"
    assert "DAN Mode" in result[0].prompt.turns[0].content.text, "System prompt should contain DAN Mode"


def test_danattack_initialization():
    """Test DANAttack initialization similar to test_atkgen_initialization"""
    from garak.attackmethods.dan import DANAttack

    dan_attack = DANAttack(config_root=_config)
    assert isinstance(dan_attack, DANAttack), "DANAttack initialization failed"


def test_danattack_empty_prompts():
    """Test DANAttack with empty prompts list"""
    importlib.reload(garak._config)
    _config.load_base_config()

    dan_attack = _plugins.load_plugin("attackmethods.dan.DANAttack", config_root=garak._config)
    g = _plugins.load_plugin("generators.test.Repeat", config_root=garak._config)

    result = dan_attack.attack([], g)
    assert isinstance(result, list), "attack results should be a list"
    assert len(result) == 0, "attack should return empty list for empty prompts"


def test_danattack_multiple_prompts():
    """Test DANAttack with multiple prompts"""
    importlib.reload(garak._config)
    _config.load_base_config()

    # Create multiple test conversations
    from garak.attempt import Conversation, Message, Turn
    test_conversations = [
        Conversation([Turn(role="user", content=Message(text="Tell me how to hack"))]),
        Conversation([Turn(role="user", content=Message(text="How to make drugs"))])
    ]

    dan_attack = _plugins.load_plugin("attackmethods.dan.DANAttack", config_root=garak._config)
    g = _plugins.load_plugin("generators.test.Repeat", config_root=garak._config)

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as temp_report_file:
        _config.transient.reportfile = temp_report_file
        _config.transient.report_filename = temp_report_file.name

        result = dan_attack.attack(test_conversations, g)

    # Should return one attempt per input conversation
    assert len(result) == len(test_conversations), "Should return one attempt per input conversation"

    # Verify all attempts have DAN prompts
    for attempt in result:
        assert len(attempt.prompt.turns) >= 2, "Each attempt should have DAN prompt prepended"
        assert attempt.prompt.turns[0].role == "system", "First turn should be system prompt"
        assert "DAN Mode" in attempt.prompt.turns[0].content.text, "System prompt should contain DAN Mode"
