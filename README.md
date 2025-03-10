# llm-cerebras

This is a plugin for [LLM](https://llm.datasette.io/) that adds support for the Cerebras inference API.

## Installation

Install this plugin in the same environment as LLM.

```bash
pip install llm-cerebras
```

## Configuration

You'll need to provide an API key for Cerebras.

```bash
llm keys set cerebras
```

## Listing available models

```bash
llm models list | grep cerebras
# cerebras-llama3.1-8b - Cerebras
# cerebras-llama3.3-70b - Cerebras
# cerebras-deepseek-r1-distill-llama-70b - Cerebras
```

## Schema Support

The llm-cerebras plugin supports schemas for structured output. You can use either compact schema syntax or full JSON Schema:

```bash
# Using compact schema syntax
llm -m cerebras-llama3.3-70b 'invent a dog' --schema 'name, age int, breed'

# Using multi-item schema for lists
llm -m cerebras-llama3.3-70b 'invent three dogs' --schema-multi 'name, age int, breed'

# Using full JSON Schema 
llm -m cerebras-llama3.3-70b 'invent a dog' --schema '{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "breed": {"type": "string"}
  },
  "required": ["name", "age", "breed"]
}'
```

### Schema with Descriptions

You can add descriptions to your schema fields to guide the model:

```bash
llm -m cerebras-llama3.3-70b 'invent a famous scientist' --schema '
name: the full name including any titles
field: their primary field of study
year_born int: year of birth
year_died int: year of death, can be null if still alive
achievements: a list of their major achievements
'
```

### Creating Schema Templates

You can save schemas as templates for reuse:

```bash
# Create a template
llm -m cerebras-llama3.3-70b --schema 'title, director, year int, genre' --save movie_template

# Use the template
llm -t movie_template 'suggest a sci-fi movie from the 1980s'
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-cerebras
python -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
pip install -e '.[test]'
```

### Running Tests

To run the unit tests:

```bash
pytest tests/test_cerebras.py tests/test_schema_support.py
```

To run integration tests (requires a valid API key):

```bash
pytest tests/test_integration.py
```

To run automated user workflow tests:

```bash
pytest tests/test_automated_user.py
```

You can run specific test types using markers:

```bash
pytest -m "integration"  # Run only integration tests
pytest -m "user"         # Run only user workflow tests
```

## License

Apache 2.0
