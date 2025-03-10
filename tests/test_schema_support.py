import pytest
import json
import httpx
from unittest.mock import patch, MagicMock
from llm_cerebras.cerebras import CerebrasModel

@pytest.fixture
def cerebras_model():
    return CerebrasModel("cerebras-llama3.3-70b")

def test_schema_flag_enabled(cerebras_model):
    """Test that schema support is enabled"""
    assert cerebras_model.supports_schema == True

def test_process_schema_dict(cerebras_model):
    """Test processing a schema that's already a dict"""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    processed = cerebras_model._process_schema(schema)
    assert processed == schema

def test_process_schema_json_string(cerebras_model):
    """Test processing a schema that's a JSON string"""
    schema = json.dumps({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    })
    processed = cerebras_model._process_schema(schema)
    assert processed["type"] == "object"
    assert "name" in processed["properties"]
    assert processed["properties"]["name"]["type"] == "string"
    assert "age" in processed["properties"]
    assert processed["properties"]["age"]["type"] == "integer"

def test_process_schema_concise(cerebras_model):
    """Test processing LLM's concise schema format"""
    schema = "name, age int, bio"
    processed = cerebras_model._process_schema(schema)
    assert processed["type"] == "object"
    assert "name" in processed["properties"]
    assert processed["properties"]["name"]["type"] == "string"
    assert "age" in processed["properties"]
    assert processed["properties"]["age"]["type"] == "integer"
    assert "bio" in processed["properties"]
    assert processed["properties"]["bio"]["type"] == "string"
    assert "name" in processed["required"]
    assert "age" in processed["required"]
    assert "bio" in processed["required"]

def test_process_schema_concise_with_description(cerebras_model):
    """Test processing LLM's concise schema format with descriptions"""
    schema = "name: the person's name, age int: their age in years"
    processed = cerebras_model._process_schema(schema)
    assert processed["properties"]["name"]["description"] == "the person's name"
    assert processed["properties"]["age"]["description"] == "their age in years"

def test_process_schema_concise_newlines(cerebras_model):
    """Test processing LLM's concise schema format with newlines"""
    schema = """
    name: the person's name
    age int: their age in years
    bio: a short biography
    """
    processed = cerebras_model._process_schema(schema)
    assert "name" in processed["properties"]
    assert "age" in processed["properties"]
    assert "bio" in processed["properties"]
    assert processed["properties"]["name"]["description"] == "the person's name"

def test_build_schema_instructions(cerebras_model):
    """Test building schema instructions for the model"""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The person's name"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    instructions = cerebras_model._build_schema_instructions(schema)
    assert "You are a helpful assistant" in instructions
    assert "Your response must follow this schema" in instructions
    assert "name" in instructions
    assert "age" in instructions
    assert "required" in instructions
    assert "The person's name" in instructions

@patch('llm_cerebras.cerebras.httpx.post')
@patch('llm_cerebras.cerebras.llm.get_key')
def test_execute_with_schema_json_object(mock_get_key, mock_post, cerebras_model):
    """Test execution with schema using json_object"""
    # Setup mocks
    mock_get_key.return_value = "fake-api-key"
    
    # Configure mock for a successful json_object response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"name": "Alice", "age": 30}'}}]
    }
    mock_post.return_value = mock_response
    
    # Setup prompt with schema
    prompt = MagicMock()
    prompt.prompt = "Generate a person"
    prompt.schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}
    prompt.options.temperature = 0.7
    prompt.options.max_tokens = None
    prompt.options.top_p = 1
    prompt.options.seed = None
    
    # Execute
    response = MagicMock()
    conversation = None
    
    # Patch the method to simplify the test
    with patch.object(cerebras_model, '_build_messages', return_value=[{"role": "user", "content": "Generate a person"}]):
        result = list(cerebras_model.execute(prompt, False, response, conversation))
    
    # Verify
    assert len(result) == 1
    assert json.loads(result[0]) == {"name": "Alice", "age": 30}
    
    # Check that the request was made with json_object
    assert mock_post.call_count == 1
    call_args = mock_post.call_args[1]
    assert "response_format" in call_args["json"]
    assert call_args["json"]["response_format"] == {"type": "json_object"}
    
    # Verify system message was added with schema instructions
    messages = call_args["json"]["messages"]
    assert len(messages) > 1  # Should have user message + system message
    assert messages[0]["role"] == "system"
    assert "Your response must follow this schema" in messages[0]["content"]

@patch('llm_cerebras.cerebras.httpx.post')
@patch('llm_cerebras.cerebras.llm.get_key')
def test_validate_schema_success(mock_get_key, mock_post, cerebras_model):
    """Test schema validation success"""
    # Setup mocks
    mock_get_key.return_value = "fake-api-key"
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"name": "Alice", "age": 30}'}}]
    }
    mock_post.return_value = mock_response
    
    # Setup prompt with schema
    prompt = MagicMock()
    prompt.prompt = "Generate a person"
    prompt.schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}
    prompt.options.temperature = 0.7
    prompt.options.max_tokens = None
    prompt.options.top_p = 1
    prompt.options.seed = None
    
    # Execute
    response = MagicMock()
    conversation = None
    
    # Patch the method to simplify the test
    with patch.object(cerebras_model, '_build_messages', return_value=[{"role": "user", "content": "Generate a person"}]):
        result = list(cerebras_model.execute(prompt, False, response, conversation))
    
    # Verify
    assert len(result) == 1
    assert json.loads(result[0]) == {"name": "Alice", "age": 30}

@patch('llm_cerebras.cerebras.httpx.post')
@patch('llm_cerebras.cerebras.llm.get_key')
def test_execute_with_concise_schema(mock_get_key, mock_post, cerebras_model):
    """Test execution with concise schema format"""
    # Setup mocks
    mock_get_key.return_value = "fake-api-key"
    
    # Second call succeeds with json_object
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"name": "Bob", "age": 25, "bio": "A software developer"}'}}]
    }
    mock_post.return_value = mock_response
    
    # Setup prompt with concise schema
    prompt = MagicMock()
    prompt.prompt = "Generate a person"
    prompt.schema = "name, age int, bio: a short bio"
    prompt.options.temperature = 0.7
    prompt.options.max_tokens = None
    prompt.options.top_p = 1
    prompt.options.seed = None
    
    # Execute
    response = MagicMock()
    conversation = None
    
    # Patch the method to simplify the test
    with patch.object(cerebras_model, '_build_messages', return_value=[{"role": "user", "content": "Generate a person"}]):
        result = list(cerebras_model.execute(prompt, False, response, conversation))
    
    # Verify
    assert len(result) == 1
    parsed = json.loads(result[0])
    assert "name" in parsed
    assert "age" in parsed
    assert "bio" in parsed
    assert isinstance(parsed["age"], int)
