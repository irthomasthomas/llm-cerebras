"""
Automated user tests that simulate typical user workflows with llm-cerebras.
These tests require a properly installed llm environment with llm-cerebras.
"""

import os
import pytest
import json
import subprocess
import tempfile
from pathlib import Path
import re

# Skip tests if SKIP_USER_TESTS is set
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_USER_TESTS") == "1",
    reason="SKIP_USER_TESTS is set"
)

def run_command(cmd):
    """Run a shell command and return output"""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=False
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def check_model_available():
    """Check if cerebras models are available in llm"""
    stdout, stderr, returncode = run_command("llm models list")
    return returncode == 0 and "cerebras" in stdout.lower()

@pytest.mark.user
def test_plugin_installation():
    """Test that the plugin is properly installed and recognized by llm"""
    # Skip if pytest is run from development directory
    if os.path.exists("pyproject.toml") and "llm-cerebras" in open("pyproject.toml").read():
        pytest.skip("Running from development directory")
    
    # Check if plugin is listed
    stdout, stderr, returncode = run_command("llm plugins")
    assert returncode == 0, f"llm plugins command failed: {stderr}"
    assert "cerebras" in stdout, "cerebras plugin not found in llm plugins list"

@pytest.mark.user
def test_models_listing():
    """Test that cerebras models are listed by llm"""
    if not check_model_available():
        pytest.skip("cerebras models not available")
    
    stdout, stderr, returncode = run_command("llm models list | grep -i cerebras")
    assert returncode == 0, "No cerebras models found"
    
    models = stdout.strip().split("\n")
    assert len(models) >= 1, "No cerebras models found"
    
    # Check for expected models
    model_ids = [line.split(" - ")[0].strip() for line in models]
    assert any("cerebras-llama" in model_id for model_id in model_ids), "No llama models found"

@pytest.mark.user
def test_workflow_basic_prompt():
    """Test a basic user workflow with a simple prompt"""
    if not check_model_available():
        pytest.skip("cerebras models not available")
    
    # Test a simple prompt
    stdout, stderr, returncode = run_command("llm -m cerebras-llama3.1-8b 'Write a haiku about programming'")
    assert returncode == 0, f"Command failed: {stderr}"
    assert len(stdout) > 10, "Response too short"
    
    # Haikus typically have three lines
    lines = [line for line in stdout.split("\n") if line.strip()]
    assert 2 <= len(lines) <= 5, f"Response doesn't look like a haiku: {stdout}"

@pytest.mark.user
def test_workflow_schema_prompt():
    """Test a user workflow with a schema prompt"""
    if not check_model_available():
        pytest.skip("cerebras models not available")
    
    # Test a schema prompt
    stdout, stderr, returncode = run_command("""
    llm -m cerebras-llama3.1-8b --schema 'title, year int, director, genre' 'Suggest a sci-fi movie'
    """)
    assert returncode == 0, f"Command failed: {stderr}"
    
    # Try to parse as JSON
    try:
        data = json.loads(stdout)
        assert "title" in data, "Response missing title"
        assert "year" in data, "Response missing year"
        assert "director" in data, "Response missing director"
        assert "genre" in data, "Response missing genre"
        assert isinstance(data["year"], int), "Year is not an integer"
    except json.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON: {stdout}")

@pytest.mark.user
def test_workflow_conversation():
    """Test a conversational workflow with follow-up questions"""
    if not check_model_available():
        pytest.skip("cerebras models not available")
    
    # Create a temporary conversation file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        conversation_file = f.name
    
    try:
        # First question
        cmd1 = f"llm -m cerebras-llama3.1-8b -c {conversation_file} 'What are the three laws of robotics?'"
        stdout1, stderr1, returncode1 = run_command(cmd1)
        assert returncode1 == 0, f"Command failed: {stderr1}"
        assert "law" in stdout1.lower() and "robot" in stdout1.lower(), "Response doesn't mention laws or robots"
        
        # Follow-up question
        cmd2 = f"llm -c {conversation_file} 'Who created these laws?'"
        stdout2, stderr2, returncode2 = run_command(cmd2)
        assert returncode2 == 0, f"Command failed: {stderr2}"
        assert "asimov" in stdout2.lower(), "Response doesn't mention Asimov"
    finally:
        # Clean up
        if os.path.exists(conversation_file):
            os.unlink(conversation_file)

@pytest.mark.user
def test_workflow_schema_template():
    """Test creating and using a schema template"""
    if not check_model_available():
        pytest.skip("cerebras models not available")
    
    # Create a schema template
    template_name = "test_movie_schema"
    
    # Remove template if it exists
    run_command(f"llm templates rm {template_name} 2>/dev/null || true")
    
    try:
        # Create template
        cmd1 = f"""
        llm -m cerebras-llama3.1-8b --schema '
        title: the movie title
        year int: release year
        director: the director
        genre: the primary genre
        ' --system 'You are a helpful assistant that recommends movies' --save {template_name}
        """
        stdout1, stderr1, returncode1 = run_command(cmd1)
        assert returncode1 == 0, f"Template creation failed: {stderr1}"
        
        # Check template exists
        cmd2 = f"llm templates show {template_name}"
        stdout2, stderr2, returncode2 = run_command(cmd2)
        assert returncode2 == 0, f"Template check failed: {stderr2}"
        assert "title" in stdout2, "Template doesn't contain expected schema"
        
        # Use template
        cmd3 = f"llm -m cerebras-llama3.1-8b -t {template_name} 'Suggest a comedy movie'"
        stdout3, stderr3, returncode3 = run_command(cmd3)
        assert returncode3 == 0, f"Template use failed: {stderr3}"
        
        # Try to parse as JSON
        try:
            data = json.loads(stdout3)
            assert "title" in data, "Response missing title"
            assert "year" in data, "Response missing year"
            assert "director" in data, "Response missing director"
            assert "genre" in data, "Response missing genre"
            assert isinstance(data["year"], int), "Year is not an integer"
            assert data["genre"].lower() == "comedy", f"Genre is not comedy: {data['genre']}"
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON: {stdout3}")
    finally:
        # Clean up template
        run_command(f"llm templates rm {template_name} 2>/dev/null || true")
