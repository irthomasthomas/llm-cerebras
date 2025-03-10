import llm
import httpx
import json
from pydantic import Field
from typing import Optional, List, Dict, Any, Union
import logging

# Try to import jsonschema for validation
try:
    import jsonschema
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False
    logging.warning("jsonschema not installed, schema validation will be limited")

@llm.hookimpl
def register_models(register):
    for model_id in CerebrasModel.model_map.keys():
        aliases = tuple()
        register(CerebrasModel(model_id), aliases=aliases)

class CerebrasModel(llm.Model):
    can_stream = True
    model_id: str
    api_base = "https://api.cerebras.ai/v1"
    supports_schema = True  # Enable schema support

    model_map = {
        "cerebras-llama3.1-8b": "llama3.1-8b",
        "cerebras-llama3.3-70b": "llama-3.3-70b",
        "cerebras-deepseek-r1-distill-llama-70b": "DeepSeek-R1-Distill-Llama-70B",
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

        # Handle schema using json_object mode
        if hasattr(prompt, 'schema') and prompt.schema:
            # Convert llm's concise schema format to JSON Schema if needed
            schema = self._process_schema(prompt.schema)
            
            # First try the native json_schema approach
            try_native_schema = False  # Set to True to try native schema first
            
            if try_native_schema and not stream:  # json_schema doesn't support streaming
                try:
                    json_schema_data = data.copy()
                    json_schema_data["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "strict": True,
                            "schema": schema
                        }
                    }
                    
                    # Try the API with json_schema format
                    url = f"{self.api_base}/chat/completions"
                    r = httpx.post(url, json=json_schema_data, headers=headers, timeout=None)
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    yield content
                    return
                except httpx.HTTPStatusError:
                    # If json_schema fails, fall back to json_object with instructions
                    logging.info("json_schema format not supported yet, falling back to json_object with instructions")
            
            # Use json_object mode with schema in system message
            data["response_format"] = {"type": "json_object"}
            
            # Add schema instructions via system message if not already present
            schema_instructions = self._build_schema_instructions(schema)
            has_system = any(msg.get("role") == "system" for msg in messages)
            
            if not has_system:
                # Insert system message at the beginning
                messages.insert(0, {"role": "system", "content": schema_instructions})
                data["messages"] = messages
            else:
                # Append schema instructions to existing system message
                for msg in messages:
                    if msg.get("role") == "system":
                        msg["content"] = msg["content"] + "\n\n" + schema_instructions
                        break
                data["messages"] = messages

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
            
            # If we have a schema, validate the response
            if hasattr(prompt, 'schema') and prompt.schema and not stream:
                try:
                    # Parse the JSON content
                    json_content = json.loads(content)
                    # Validate against the schema
                    schema = self._process_schema(prompt.schema)
                    self._validate_schema(json_content, schema)
                    # Return the validated JSON as a string
                    content = json.dumps(json_content)
                except (json.JSONDecodeError, ValueError, jsonschema.exceptions.ValidationError) as e:
                    logging.warning(f"Schema validation failed: {str(e)}")
                    # Continue with the original content
            
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
    
    def _process_schema(self, schema) -> Dict[str, Any]:
        """
        Process schema from llm's format to a proper JSON Schema.
        """
        if isinstance(schema, dict):
            return schema
        
        # If it's a string, check if it's a JSON string
        if isinstance(schema, str):
            try:
                return json.loads(schema)
            except json.JSONDecodeError:
                # This might be using llm's concise schema format
                # For now, convert it to a basic JSON schema
                properties = {}
                required = []
                
                # Handle both comma-separated and newline-separated formats
                if "," in schema and "\n" not in schema:
                    parts = [p.strip() for p in schema.split(",")]
                else:
                    parts = [p.strip() for p in schema.split("\n") if p.strip()]
                
                for part in parts:
                    # Handle field description format: name: description
                    if ":" in part:
                        field_def, description = part.split(":", 1)
                    else:
                        field_def, description = part, ""
                    
                    # Handle type annotations: name int, name float, etc.
                    if " " in field_def:
                        field_name, field_type = field_def.split(" ", 1)
                    else:
                        field_name, field_type = field_def, "string"
                    
                    # Map to JSON schema types
                    type_mapping = {
                        "int": "integer",
                        "float": "number",
                        "str": "string",
                        "string": "string",
                        "bool": "boolean",
                    }
                    json_type = type_mapping.get(field_type.lower(), "string")
                    
                    # Add to properties
                    properties[field_name] = {"type": json_type}
                    if description:
                        properties[field_name]["description"] = description.strip()
                    
                    # All fields are required by default in llm's schema format
                    required.append(field_name)
                
                return {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
        
        # Default empty schema
        return {"type": "object", "properties": {}}
    
    def _build_schema_instructions(self, schema: Dict[str, Any]) -> str:
        """
        Generate instructions for the model to follow the schema.
        """
        instructions = "You are a helpful assistant that returns responses in JSON format. "
        instructions += "Your response must follow this schema exactly:\n"
        
        # Format the schema as a readable instruction
        if schema.get("type") == "object" and "properties" in schema:
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            instructions += "{\n"
            for prop_name, prop_details in properties.items():
                prop_type = prop_details.get("type", "string")
                prop_desc = prop_details.get("description", "")
                is_required = prop_name in required
                
                instructions += f'  "{prop_name}": {prop_type}'
                if prop_desc:
                    instructions += f" // {prop_desc}"
                if is_required:
                    instructions += " (required)"
                instructions += ",\n"
            instructions += "}\n"
        else:
            # Fallback to JSON representation
            instructions += json.dumps(schema, indent=2)
        
        instructions += "\nYour response must be valid JSON and follow this schema exactly. Do not include any explanations or text outside of the JSON structure."
        return instructions
    
    def _validate_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """
        Validate the response against the schema.
        """
        if HAVE_JSONSCHEMA:
            try:
                jsonschema.validate(instance=data, schema=schema)
                return True
            except jsonschema.exceptions.ValidationError as e:
                raise ValueError(f"Schema validation failed: {str(e)}")
        else:
            # Basic validation if jsonschema is not available
            if schema.get("type") == "object" and "properties" in schema:
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                
                # Check required fields
                for field in required:
                    if field not in data:
                        raise ValueError(f"Required field '{field}' is missing from response")
                
                # Check field types (simplified)
                for field, value in data.items():
                    if field in properties:
                        prop_type = properties[field].get("type")
                        if prop_type == "string" and not isinstance(value, str):
                            raise ValueError(f"Field '{field}' should be a string")
                        elif prop_type == "integer" and not isinstance(value, int):
                            raise ValueError(f"Field '{field}' should be an integer")
                        elif prop_type == "number" and not isinstance(value, (int, float)):
                            raise ValueError(f"Field '{field}' should be a number")
                        elif prop_type == "boolean" and not isinstance(value, bool):
                            raise ValueError(f"Field '{field}' should be a boolean")
                        elif prop_type == "array" and not isinstance(value, list):
                            raise ValueError(f"Field '{field}' should be an array")
                        elif prop_type == "object" and not isinstance(value, dict):
                            raise ValueError(f"Field '{field}' should be an object")
            
            return True
