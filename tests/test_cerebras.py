import pytest
from llm_cerebras.cerebras import CerebrasModel
from unittest.mock import patch, MagicMock

@pytest.fixture
def cerebras_model():
    return CerebrasModel("cerebras-llama3.1-8b")  # Use the full model ID with prefix

def test_cerebras_model_initialization(cerebras_model):
    assert cerebras_model.model_id == "cerebras-llama3.1-8b"
    assert cerebras_model.can_stream == True
    assert cerebras_model.api_base == "https://api.cerebras.ai/v1"

def test_build_messages(cerebras_model):
    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    conversation = None
    messages = cerebras_model._build_messages(prompt, conversation)
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "Test prompt"}

@patch('llm_cerebras.cerebras.httpx.post')
@patch('llm_cerebras.cerebras.llm.get_key')
def test_execute_non_streaming(mock_get_key, mock_post, cerebras_model):
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_post.return_value = mock_response

    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.options.temperature = 0.7
    prompt.options.max_tokens = None
    prompt.options.top_p = 1
    prompt.options.seed = None
    
    # Make sure prompt doesn't have a schema attribute
    type(prompt).schema = None

    response = MagicMock()
    conversation = None

    result = list(cerebras_model.execute(prompt, False, response, conversation))

    assert result == ["Test response"]
    mock_post.assert_called_once()

if __name__ == "__main__":
    pytest.main()
