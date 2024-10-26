

[![PyPI](https://img.shields.io/pypi/v/llm-cerebras.svg)](https://pypi.org/project/llm-cerebras/)
[![Changelog](https://img.shields.io/github/v/release/irthomasthomas/llm-cerebras?include_prereleases&label=changelog)](https://github.com/irthomasthomas/llm-cerebras/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/irthomasthomas/llm-cerebras/blob/main/LICENSE)

llm plugin to prompt Cerebras hosted models.



Install this plugin in the same environment as [LLM](https://llm.datasette.io/):

    llm install llm-cerebras



You'll need to obtain a Cerebras API key following the instructions [here](https://inference-docs.cerebras.ai/quickstart#step-1-set-up-your-api-key).
Once you have it, configure the plugin like this:

    llm keys set cerebras
    



To use the Cerebras models, run:

    llm -m cerebras-llama3.1-8b "Your prompt here"

Or for the 70B model:

    llm -m cerebras-llama3.1-70b "Your prompt here"



The following options are available:

- `temperature`: Controls randomness. Defaults to 0.7, range 0-1.5.
- `max_tokens`: The maximum number of tokens to generate.
- `top_p`: Alternative to temperature for nucleus sampling. Defaults to 1.
- `seed`: For deterministic sampling.

Example usage with options:

    llm -m cerebras-llama3.1-8b "Your prompt" -o temperature 0.5 -o max_tokens 100



To set up this plugin locally, first checkout the code. Then create a new virtual environment:

    cd llm-cerebras
    python3 -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest

