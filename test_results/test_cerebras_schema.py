from cerebras.cloud.sdk import Cerebras
import os
import json
from cerebras.cloud.sdk.types.chat.completion_create_params import ResponseFormatResponseFormatJsonSchemaTyped

client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

# Define a simple schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

print("Testing JSON Object format first (simpler)...")
try:
    # Test with json_object type first (simpler)
    response_format_json = {"type": "json_object"}
    
    print(f"Using response_format: {response_format_json}")
    
    response1 = client.chat.completions.create(
        model="llama-3.3-70b",
        messages=[
            {"role": "user", "content": "Generate information about a fictional person in JSON format with name and age"}
        ],
        response_format=response_format_json
    )
    print("JSON Object Response:")
    print(response1.choices[0].message.content)
    print("\n---\n")
except Exception as e:
    print(f"Error with json_object: {e}")

print("\nTesting JSON Schema format...")
try:
    # Test with json_schema type
    response_format_schema = {
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "schema": schema
        }
    }
    
    print(f"Using response_format: {json.dumps(response_format_schema, indent=2)}")
    
    response2 = client.chat.completions.create(
        model="llama-3.3-70b",
        messages=[
            {"role": "user", "content": "Generate information about a fictional person with name and age"}
        ],
        response_format=response_format_schema
    )
    print("JSON Schema Response:")
    print(response2.choices[0].message.content)
except Exception as e:
    print(f"Error with json_schema: {e}")
    
    # Try with the typed approach using SDK types
    print("\nTrying with typed SDK objects...")
    try:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            ResponseFormatResponseFormatJsonSchemaJsonSchemaTyped,
            ResponseFormatResponseFormatJsonSchemaTyped
        )
        
        json_schema = ResponseFormatResponseFormatJsonSchemaJsonSchemaTyped(
            strict=True,
            schema=schema
        )
        
        response_format = ResponseFormatResponseFormatJsonSchemaTyped(
            type="json_schema",
            json_schema=json_schema
        )
        
        print(f"Using typed response_format: {response_format}")
        
        response3 = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {"role": "user", "content": "Generate information about a fictional person with name and age"}
            ],
            response_format=response_format
        )
        print("JSON Schema Response (typed):")
        print(response3.choices[0].message.content)
    except Exception as e:
        print(f"Error with typed json_schema: {e}")
