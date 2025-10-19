# LLM Utils Implementation Summary

## What We Built

A comprehensive, production-ready LLM utilities framework for the Newspaper Explorer project with:

### 1. Core LLM Client (`utils/llm.py`)
- ✅ **Configurable API access** - URL and API key from environment
- ✅ **Model parameters** - `model_name`, `temperature`, `max_tokens`
- ✅ **Retry logic** - Exponential backoff with configurable `max_retries` and `retry_delay`
- ✅ **Request library** - Uses `requests` for maximum flexibility
- ✅ **Response validation** - Automatic Pydantic schema validation
- ✅ **Context manager** - Clean session management with `with` statement
- ✅ **Error handling** - Custom exceptions: `LLMError`, `LLMRetryError`, `LLMValidationError`
- ✅ **Logging** - Comprehensive debug and info logging
- ✅ **OpenAI-compatible** - Works with any OpenAI-format API

### 2. Prompt Templates (`utils/prompts.py`)
- ✅ **Centralized management** - All prompts in one place
- ✅ **Variable substitution** - Template formatting with `.format(**kwargs)`
- ✅ **Pre-built prompts** for common tasks:
  - Entity extraction (persons, locations, organizations)
  - Topic classification and generation
  - Emotion/sentiment analysis
  - Concept extraction with relationships
  - Text summarization
  - OCR quality assessment
- ✅ **Helper functions** - `get_prompt(name)`, `list_prompts()`

### 3. Response Schemas (`utils/schemas.py`)
- ✅ **Pydantic models** - Type-safe response validation
- ✅ **Matching schemas** for all prompt templates:
  - `EntityResponse`
  - `TopicClassificationResponse`
  - `TopicGenerationResponse`
  - `EmotionAnalysisResponse`
  - `ConceptExtractionResponse`
  - `SummarizationResponse`
  - `TextQualityResponse`
- ✅ **Field validation** - Ensures required fields, types, and constraints
- ✅ **Default handling** - Gracefully handles missing optional fields
- ✅ **Helper functions** - `get_schema(name)`, `list_schemas()`

### 4. Configuration (`utils/config.py`)
- ✅ **Extended Config class** with LLM settings:
  - `llm_base_url` - API endpoint
  - `llm_api_key` - Authentication token
  - `llm_model` - Model identifier
  - `llm_temperature` - Default temperature
  - `llm_max_tokens` - Default token limit
- ✅ **`.env` support** - All settings configurable via environment variables
- ✅ **Getter method** - `config.get(key, default)` for flexible access

### 5. Documentation
- ✅ **Comprehensive guide** (`docs/LLM.md`) with:
  - Architecture overview
  - Configuration instructions
  - Usage examples
  - Best practices
  - Troubleshooting
  - Integration patterns
- ✅ **Example file** (`utils/llm_examples.py`) with working demos
- ✅ **Environment template** (`.env.example`) with LLM variables

## Key Features

### Retry Logic with Exponential Backoff
```python
client = LLMClient(
    max_retries=3,      # Number of retry attempts
    retry_delay=1.0,    # Initial delay (doubles each retry)
    timeout=30.0        # Request timeout
)
```

Retries on:
- Rate limits (429)
- Server errors (500, 502, 503, 504)
- Timeouts

### Response Validation
```python
# Automatically validates against Pydantic schema
response = client.complete(
    prompt="Extract entities from: Berlin 1901",
    response_schema=EntityResponse
)

# Type-safe access with IDE autocomplete
print(response.persons)      # List[str]
print(response.locations)    # List[str]
```

### Flexible Configuration
```python
# From environment (.env)
client = LLMClient()  # Uses LLM_* vars

# Override programmatically
client = LLMClient(
    base_url="https://custom-api.com/v1",
    api_key="custom-key",
    model_name="gpt-4o",
    temperature=0.3
)

# Per-request overrides
response = client.complete(
    prompt="...",
    temperature=0.8,    # Override for this request
    max_tokens=1000
)
```

### Centralized Prompts
```python
# Get pre-built prompt
prompt = get_prompt("entity_extraction")
formatted = prompt.format(text="German newspaper text...")

# Use with client
response = client.complete(
    prompt=formatted["user"],
    system_prompt=formatted["system"],
    response_schema=EntityResponse
)
```

## What You Asked For (Checklist)

✅ **Configurable with URL and API token** - Via environment or constructor  
✅ **Uses requests library** - For maximum flexibility  
✅ **Temperature** - Global default + per-request override  
✅ **max_tokens** - Global default + per-request override  
✅ **Retries** - Exponential backoff with configurable attempts  
✅ **retry_delay** - Initial delay (doubles each retry)  
✅ **max_retries** - Number of retry attempts  
✅ **model_name** - Configurable model identifier  
✅ **Central prompt management** - `prompts.py` with templates  
✅ **Response schemas** - `schemas.py` with Pydantic models  
✅ **Validation** - Automatic validation against schemas  
✅ **Logging** - Comprehensive logging throughout  
✅ **Error handling** - Custom exceptions for different error types  

## Usage Example

```python
from newspaper_explorer.utils.llm import LLMClient
from newspaper_explorer.utils.prompts import get_prompt
from newspaper_explorer.utils.schemas import EntityResponse

# Text to analyze
text = "Kaiser Wilhelm II in Berlin, 1901."

# Get prompt template
prompt = get_prompt("entity_extraction")
formatted = prompt.format(text=text)

# Make request with automatic retry and validation
with LLMClient(temperature=0.3, max_retries=5) as client:
    response = client.complete(
        prompt=formatted["user"],
        system_prompt=formatted["system"],
        response_schema=EntityResponse,
    )

# Type-safe, validated response
print(f"Persons: {response.persons}")
print(f"Locations: {response.locations}")
```

## Files Created/Modified

### New Files
1. `src/newspaper_explorer/utils/llm.py` - Main LLM client (300+ lines)
2. `src/newspaper_explorer/utils/prompts.py` - Prompt templates (250+ lines)
3. `src/newspaper_explorer/utils/schemas.py` - Response schemas (200+ lines)
4. `src/newspaper_explorer/utils/llm_examples.py` - Usage examples
5. `docs/LLM.md` - Comprehensive documentation (500+ lines)

### Modified Files
1. `src/newspaper_explorer/utils/config.py` - Added LLM configuration
2. `.env.example` - Added LLM environment variables

## Environment Setup

Add to `.env`:

```bash
# LLM API Configuration
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

## Next Steps

To use this in your analysis pipeline:

1. **Set environment variables** - Copy `.env.example` to `.env` and configure
2. **Install dependencies** - Ensure `requests` and `pydantic` are installed
3. **Import and use**:
   ```python
   from newspaper_explorer.utils.llm import LLMClient
   from newspaper_explorer.utils.prompts import get_prompt
   from newspaper_explorer.utils.schemas import EntityResponse
   ```

4. **Integrate into analysis** - Use in entity extraction, topic modeling, etc.

## Testing

Run examples to verify setup:
```bash
python src/newspaper_explorer/utils/llm_examples.py
```

## Benefits

1. **Type Safety** - Pydantic validation ensures correct response format
2. **Consistency** - Centralized prompts ensure consistent LLM behavior
3. **Reliability** - Automatic retries handle transient failures
4. **Flexibility** - Works with any OpenAI-compatible API
5. **Maintainability** - Single source of truth for prompts and schemas
6. **Developer Experience** - IDE autocomplete for response fields
7. **Production Ready** - Proper error handling, logging, and cleanup

## What's NOT Included (Future Work)

- Streaming responses
- Async/await support
- Request caching
- Cost tracking
- Batch optimization
- Multi-model ensemble

These can be added as needed based on usage patterns.
