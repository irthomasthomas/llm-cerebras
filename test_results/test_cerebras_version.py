import cerebras
import cerebras.cloud.sdk
from cerebras.cloud.sdk import Cerebras

# Try to find version information
print(f"Cerebras package: {cerebras}")
print(f"Cerebras SDK module: {cerebras.cloud.sdk}")

# Check available types and modules
print("\nModules in cerebras.cloud.sdk:")
for item in dir(cerebras.cloud.sdk):
    if not item.startswith("_"):
        print(f"  - {item}")
        
# Check the types module
print("\nChecking types module:")
from cerebras.cloud.sdk import types
print(f"Available types: {dir(types)}")

# Look at completion_create_params module
print("\nChecking completion_create_params module:")
import cerebras.cloud.sdk.types.completion_create_params as params
print(f"Attributes: {dir(params)}")

# Test importing the ResponseFormat
try:
    from cerebras.cloud.sdk.types.chat.completion_create_params import ResponseFormat
    print("\nResponseFormat imported successfully")
    print(f"ResponseFormat: {ResponseFormat}")
except ImportError as e:
    print(f"\nImport error: {e}")
