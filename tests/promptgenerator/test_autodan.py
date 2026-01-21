import importlib
from unittest.mock import patch

import garak.attempt
import garak.generators
import garak.probes.base
from garak import _config, _plugins
from garak.attempt import Conversation, Message, Turn
from garak.promptgenerator.dan import AutoDANPromptGenerator


def test_generate_prompts_without_generator():
    """Test prompt generation without providing a generator"""
    importlib.reload(garak._config)
    _config.load_base_config()

    generator = AutoDANPromptGenerator()

    result = generator.generate_prompts(generator=None)

    # Verify empty result
    assert result == []


@patch('garak.promptgenerator.dan.autodan_generate')
def test_generate_prompts_with_generator_success(mock_autodan_generate):
    """Test successful prompt generation with a generator"""
    # Setup mock autodan function
    test_conversation = Conversation([
        Turn(role="user", content=Message(text="AutoDAN generated prompt"))
    ])
    mock_autodan_generate.return_value = [test_conversation]

    importlib.reload(garak._config)
    _config.load_base_config()

    generator = AutoDANPromptGenerator()
    test_generator = _plugins.load_plugin("generators.test.Repeat", config_root=garak._config)

    result = generator.generate_prompts(generator=test_generator)

    # Verify autodan_generate was called with correct parameters
    mock_autodan_generate.assert_called_once_with(
        generator=test_generator,
        prompt=generator.goal_str,
        target=generator.target
    )

    # Verify result
    assert len(result) == 1
    assert result[0] == test_conversation
    assert isinstance(result[0], Conversation)
    assert len(result[0].turns) == 1
    assert result[0].turns[0].role == "user"
