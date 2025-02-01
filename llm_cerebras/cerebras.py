import llm
import httpx
import json
from pydantic import Field
from typing import Optional, List

@llm.hookimpl
def register_models(register):
    for model_id in CerebrasModel.model_map.keys():
        register(CerebrasModel(model_id))

class CerebrasModel(llm.Model):
    can_stream = True
    model_id: str
    api_base = "https://api.cerebras.ai/v1"

    model_map = {
    "cerebras-llama3.1-8b": "llama3.1-8b",
    "cerebras-llama3.1-70b": "llama3.1-70b",
    "cerebras-deepseek-r1-distill-llama-70b": "DeepSeek-R1-Distill-Llama-70B"
    }

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description="What sampling temperature to use, between 0 and 1.5.",
            ge=0,
            le=1.5,
            default=0.7,
        )
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate.",
            default=None,
        )
        top_p: Optional[float] = Field(
            description="An alternative to sampling with temperature, called nucleus sampling.",
            ge=0,
            le=1,
            default=1,
        )
        seed: Optional[int] = Field(
            description="If specified, our system will make a best effort to sample deterministically.",
            default=None,
        )

    def __init__(self, model_id):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation):
        messages = self._build_messages(prompt, conversation)
        api_key = llm.get_key("", "cerebras", "CEREBRAS_API_KEY")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": self.model_map[self.model_id],
            "messages": messages,
            "stream": stream,
            "temperature": prompt.options.temperature,
            "max_tokens": prompt.options.max_tokens,
            "top_p": prompt.options.top_p,
            "seed": prompt.options.seed,
        }

        url = f"{self.api_base}/chat/completions"

        if stream:
            with httpx.stream("POST", url, json=data, headers=headers, timeout=None) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk != "[DONE]":
                            content = json.loads(chunk)["choices"][0]["delta"].get("content")
                            if content:
                                yield content
        else:
            r = httpx.post(url, json=data, headers=headers, timeout=None)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            yield content

    def _build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        if conversation:
            for response in conversation.responses:
                messages.extend([
                    {"role": "user", "content": response.prompt.prompt},
                    {"role": "assistant", "content": response.text()},
                ])
        messages.append({"role": "user", "content": prompt.prompt})
        return messages