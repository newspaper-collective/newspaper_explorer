# LLM Utilities Documentation

Comprehensive utilities for interacting with Large Language Models (LLMs) in the Newspaper Explorer project.

## Overview

The LLM utilities provide a flexible, production-ready framework for:
- **Configurable API access** with URL and token management
- **Automatic retries** with exponential backoff
- **Response validation** using Pydantic schemas
- **Centralized prompt templates** for consistency
- **Type-safe responses** with full IDE support

## Architecture

### Core Components

1. **`llm.py`** - Main client with retry logic and validation
2. **`prompts.py`** - Centralized prompt templates
3. **`schemas.py`** - Pydantic response schemas
4. **`config.py`** - Configuration management (extended)

### Features

- ✅ **Configurable endpoints** - Works with any OpenAI-compatible API
- ✅ **Retry logic** - Exponential backoff for rate limits and transient errors
- ✅ **Response validation** - Automatic Pydantic validation against schemas
- ✅ **Temperature control** - Per-request or global temperature settings
- ✅ **Token limits** - Configurable max_tokens per request
- ✅ **Context manager** - Automatic session cleanup
- ✅ **Logging** - Comprehensive debug and info logging

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Required
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key-here

# Optional (with defaults)
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

### Supported API Endpoints

Any OpenAI-compatible API:
- **OpenAI**: `https://api.openai.com/v1`
- **Azure OpenAI**: `https://your-resource.openai.azure.com/openai/deployments/your-deployment`
- **Local models** (e.g., Ollama): `http://localhost:11434/v1`
- **Other providers**: Any API following OpenAI's chat completion format

## Usage Examples

### Basic Entity Extraction

```python
from newspaper_explorer.utils.llm import LLMClient
from newspaper_explorer.utils.prompts import get_prompt
from newspaper_explorer.utils.schemas import EntityResponse

# Text to analyze
text = "Kaiser Wilhelm II empfing Bernhard von Bülow in Berlin."

# Get prompt template
prompt = get_prompt("entity_extraction")
formatted = prompt.format(text=text)

# Make request with validation
with LLMClient() as client:
    response = client.complete(
        prompt=formatted["user"],
        system_prompt=formatted["system"],
        response_schema=EntityResponse,
    )

# Type-safe access
print(response.persons)      # ["Kaiser Wilhelm II", "Bernhard von Bülow"]
print(response.locations)    # ["Berlin"]
```

### Custom Parameters

```python
# Override defaults for specific request
with LLMClient(
    model_name="gpt-4o",
    temperature=0.3,
    max_retries=5,
    retry_delay=2.0,
) as client:
    response = client.complete(
        prompt="Summarize: ...",
        max_tokens=500,  # Override for this request
    )
```

### Without Schema Validation

```python
# Get raw string response
with LLMClient() as client:
    response = client.complete(
        prompt="Translate to English: Das Wetter ist schön.",
        system_prompt="You are a translator.",
    )
    # response is str, not validated model
```

### Error Handling

```python
from newspaper_explorer.utils.llm import LLMClient, LLMRetryError, LLMValidationError

try:
    with LLMClient(max_retries=3) as client:
        response = client.complete(
            prompt="...",
            response_schema=EntityResponse,
        )
except LLMRetryError as e:
    # All retry attempts exhausted
    logger.error(f"Request failed: {e}")
except LLMValidationError as e:
    # Response doesn't match schema
    logger.error(f"Invalid response format: {e}")
```

## Available Prompts

Use `get_prompt(name)` to retrieve templates:

| Name | Purpose | Schema |
|------|---------|--------|
| `entity_extraction` | Extract persons, locations, organizations | `EntityResponse` |
| `topic_classification` | Classify text into predefined topics | `TopicClassificationResponse` |
| `topic_generation` | Generate topic labels for text | `TopicGenerationResponse` |
| `emotion_analysis` | Analyze sentiment and emotions | `EmotionAnalysisResponse` |
| `concept_extraction` | Extract key concepts and relationships | `ConceptExtractionResponse` |
| `summarization` | Summarize text with key points | `SummarizationResponse` |
| `text_quality` | Assess OCR quality and readability | `TextQualityResponse` |

### List Available Prompts

```python
from newspaper_explorer.utils.prompts import list_prompts

print(list_prompts())
# ['entity_extraction', 'topic_classification', ...]
```

## Response Schemas

All schemas are Pydantic models with validation:

### EntityResponse

```python
{
    "persons": ["name1", "name2"],
    "locations": ["place1"],
    "organizations": ["org1"]
}
```

### EmotionAnalysisResponse

```python
{
    "sentiment": "positive",  # positive|negative|neutral|mixed
    "emotions": ["pride", "hope"],
    "intensity": 0.8,
    "tone": "patriotic"
}
```

### ConceptExtractionResponse

```python
{
    "concepts": ["Industrialisierung", "Modernisierung"],
    "relationships": [
        {
            "source": "Industrialisierung",
            "target": "Urbanisierung",
            "type": "leads_to"
        }
    ]
}
```

See `schemas.py` for all available schemas.

## Creating Custom Prompts

### Define a Template

```python
from newspaper_explorer.utils.prompts import PromptTemplate

my_prompt = PromptTemplate(
    system="You are an expert in historical analysis.",
    user="Analyze this text: {text}\n\nFocus on: {focus_area}"
)

# Use it
formatted = my_prompt.format(
    text="...",
    focus_area="economic indicators"
)
```

### Add to Central Registry

Edit `prompts.py` to add permanent prompts:

```python
MY_NEW_PROMPT = PromptTemplate(
    system="...",
    user="..."
)

# Update get_prompt() function to include it
```

## Creating Custom Schemas

### Define with Pydantic

```python
from pydantic import BaseModel, Field

class MyResponse(BaseModel):
    field1: str = Field(description="...")
    field2: int = Field(ge=0, le=100, description="...")
```

### Use with LLMClient

```python
response = client.complete(
    prompt="...",
    response_schema=MyResponse,
)
# Returns validated MyResponse instance
```

## Retry Logic

### Default Behavior

- **Max retries**: 3 attempts
- **Retry delay**: 1 second (doubles each attempt)
- **Retryable errors**: 429, 500, 502, 503, 504, timeouts
- **Non-retryable**: 400, 401, 403, parsing errors

### Customization

```python
client = LLMClient(
    max_retries=5,
    retry_delay=2.0,  # Initial delay in seconds
    timeout=60.0,     # Request timeout
)
```

### Exponential Backoff

Delays between retries:
- Attempt 1: `retry_delay` (e.g., 1s)
- Attempt 2: `retry_delay * 2` (e.g., 2s)
- Attempt 3: `retry_delay * 4` (e.g., 4s)
- ...

## Best Practices

### 1. Use Context Managers

```python
# ✅ Good - automatic cleanup
with LLMClient() as client:
    response = client.complete(...)

# ❌ Avoid - manual cleanup required
client = LLMClient()
response = client.complete(...)
client.close()  # Easy to forget
```

### 2. Use Schema Validation

```python
# ✅ Good - type-safe, validated
response = client.complete(
    prompt="...",
    response_schema=EntityResponse,
)
print(response.persons)  # IDE autocomplete works

# ❌ Avoid - manual parsing, error-prone
response = client.complete(prompt="...")
data = json.loads(response)  # Manual validation
```

### 3. Use Centralized Prompts

```python
# ✅ Good - consistent, reusable
prompt = get_prompt("entity_extraction")
formatted = prompt.format(text=text)

# ❌ Avoid - duplicated, inconsistent
prompt = "Extract entities from: {text}"
```

### 4. Handle Errors Gracefully

```python
# ✅ Good - specific error handling
try:
    response = client.complete(...)
except LLMRetryError:
    # Fallback logic
    use_cached_result()
except LLMValidationError:
    # Try with different prompt
    response = client.complete(simplified_prompt)
```

### 5. Adjust Temperature by Task

- **Deterministic tasks** (extraction, classification): `temperature=0.0-0.3`
- **Balanced** (summarization, analysis): `temperature=0.5-0.7`
- **Creative tasks** (generation, brainstorming): `temperature=0.8-1.2`

## Integration with Analysis Pipeline

### Example: Batch Entity Extraction

```python
import polars as pl
from newspaper_explorer.data.loading import DataLoader
from newspaper_explorer.utils.llm import LLMClient
from newspaper_explorer.utils.prompts import get_prompt
from newspaper_explorer.utils.schemas import EntityResponse

# Load data
loader = DataLoader(source_name="der_tag")
df = loader.load_source()

# Extract entities for sample
prompt_template = get_prompt("entity_extraction")

results = []
with LLMClient(temperature=0.3) as client:
    for row in df.head(10).iter_rows(named=True):
        text = row["text"]
        prompts = prompt_template.format(text=text)

        try:
            response = client.complete(
                prompt=prompts["user"],
                system_prompt=prompts["system"],
                response_schema=EntityResponse,
            )
            results.append({
                "line_id": row["line_id"],
                "persons": response.persons,
                "locations": response.locations,
            })
        except Exception as e:
            logger.warning(f"Failed for {row['line_id']}: {e}")

# Convert to DataFrame
entities_df = pl.DataFrame(results)
```

## Testing

Run the example file:

```bash
python src/newspaper_explorer/utils/llm_examples.py
```

Requires `.env` with valid `LLM_BASE_URL` and `LLM_API_KEY`.

## Troubleshooting

### "base_url must be provided"

Set `LLM_BASE_URL` in `.env`:
```bash
LLM_BASE_URL=https://api.openai.com/v1
```

### "api_key must be provided"

Set `LLM_API_KEY` in `.env`:
```bash
LLM_API_KEY=sk-your-key-here
```

### Rate Limit Errors

Increase retry settings:
```python
client = LLMClient(max_retries=10, retry_delay=5.0)
```

### Validation Errors

Check LLM response format. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Timeout Errors

Increase timeout:
```python
client = LLMClient(timeout=120.0)
```

## Future Enhancements

Potential additions:
- [ ] Streaming responses for long outputs
- [ ] Batch request optimization
- [ ] Caching layer for repeated prompts
- [ ] Cost tracking and budget limits
- [ ] Multi-model ensemble support
- [ ] Async/await support for concurrent requests
