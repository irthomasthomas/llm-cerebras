"""
Integration tests for llm-cerebras plugin.
These tests ensure that the plugin works correctly with the actual llm CLI.
NOTE: These tests require a valid CEREBRAS_API_KEY env variable to be set
and will make actual API calls to Cerebras.
"""

import os
import pytest
import json
import subprocess
import tempfile
from pathlib import Path

# Skip all tests if no API key is available
pytestmark = pytest.mark.skipif(
    os.environ.get("CEREBRAS_API_KEY") is None,
    reason="CEREBRAS_API_KEY not set in environment variables"
)

def run_llm_command(cmd_args):
    """Run an llm command and return the output"""
    result = subprocess.run(
        ["llm"] + cmd_args,
        capture_output=True,
        text=True,
        check=False
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode

@pytest.mark.integration
def test_basic_completion():
    """Test that a basic completion works"""
    stdout, stderr, returncode = run_llm_command([
        "-m", "cerebras-llama3.1-8b", 
        "Hello, how are you?"
    ])
    assert returncode == 0, f"Command failed with stderr: {stderr}"
    assert stdout, "No output returned"
    assert len(stdout) > 10, "Output too short to be a valid response"

@pytest.mark.integration
def test_schema_basic():
    """Test that a basic schema completion works"""
    stdout, stderr, returncode = run_llm_command([
        "-m", "cerebras-llama3.1-8b",
        "--schema", "name, age int",
        "Generate information about a fictional person"
    ])
    assert returncode == 0, f"Command failed with stderr: {stderr}"
    
    # Attempt to parse as JSON
    try:
        data = json.loads(stdout)
        assert "name" in data, "Response missing 'name' field"
        assert "age" in data, "Response missing 'age' field"
        assert isinstance(data["name"], str), "'name' field is not a string"
        assert isinstance(data["age"], int), "'age' field is not an integer"
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON: {stdout}")

@pytest.mark.integration
def test_schema_multi():
    """Test that a schema-multi completion works"""
    stdout, stderr, returncode = run_llm_command([
        "-m", "cerebras-llama3.1-8b",
        "--schema-multi", "name, age int",
        "Generate information about 2 fictional people"
    ])
    assert returncode == 0, f"Command failed with stderr: {stderr}"
    
    # Attempt to parse as JSON
    try:
        data = json.loads(stdout)
        assert "items" in data, "Response missing 'items' array"
        assert isinstance(data["items"], list), "'items' is not an array"
        assert len(data["items"]) > 0, "'items' array is empty"
        
        # Check the first item
        first_item = data["items"][0]
        assert "name" in first_item, "First item missing 'name' field"
        assert "age" in first_item, "First item missing 'age' field"
        assert isinstance(first_item["name"], str), "'name' field is not a string"
        assert isinstance(first_item["age"], int), "'age' field is not an integer"
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON: {stdout}")

@pytest.mark.integration
def test_complex_schema():
    """Test a more complex schema with nested objects"""
    # Create a temporary file with a complex schema
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        schema_file = f.name
        json.dump({
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "hobbies": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["name", "age", "hobbies"]
                },
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"}
                    },
                    "required": ["city", "country"]
                }
            },
            "required": ["person", "location"]
        }, f)
    
    try:
        stdout, stderr, returncode = run_llm_command([
            "-m", "cerebras-llama3.1-8b",
            "--schema", schema_file,
            "Generate a fictional person with their location"
        ])
        assert returncode == 0, f"Command failed with stderr: {stderr}"
        
        # Attempt to parse as JSON
        try:
            data = json.loads(stdout)
            assert "person" in data, "Response missing 'person' object"
            assert "location" in data, "Response missing 'location' object"
            
            # Check person
            assert "name" in data["person"], "Person missing 'name' field"
            assert "age" in data["person"], "Person missing 'age' field"
            assert "hobbies" in data["person"], "Person missing 'hobbies' field"
            assert isinstance(data["person"]["hobbies"], list), "'hobbies' is not an array"
            
            # Check location
            assert "city" in data["location"], "Location missing 'city' field"
            assert "country" in data["location"], "Location missing 'country' field"
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON: {stdout}")
    finally:
        # Clean up the temporary file
        os.unlink(schema_file)

@pytest.mark.integration
def test_schema_with_description():
    """Test schema with descriptions"""
    stdout, stderr, returncode = run_llm_command([
        "-m", "cerebras-llama3.1-8b",
        "--schema", "name: full name including title, age int: age in years, bio: a short biography",
        "Generate information about a professor"
    ])
    assert returncode == 0, f"Command failed with stderr: {stderr}"
    
    # Attempt to parse as JSON
    try:
        data = json.loads(stdout)
        assert "name" in data, "Response missing 'name' field"
        assert "age" in data, "Response missing 'age' field"
        assert "bio" in data, "Response missing 'bio' field"
        assert isinstance(data["name"], str), "'name' field is not a string"
        assert isinstance(data["age"], int), "'age' field is not an integer"
        assert isinstance(data["bio"], str), "'bio' field is not a string"
        
        # Check if name likely contains a title (Dr., Professor, etc.)
        assert any(title in data["name"] for title in ["Dr.", "Professor", "Prof."]), "Name doesn't contain title despite description"
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON: {stdout}")
