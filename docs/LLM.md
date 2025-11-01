# LLM Utilities Documentation

Comprehensive utilities for interacting with Large Language Models (LLMs) in the Newspaper Explorer project.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Core Usage](#core-usage)
- [Metadata Support](#metadata-support)
- [Available Prompts & Schemas](#available-prompts--schemas)
- [Custom Prompts & Schemas](#custom-prompts--schemas)
- [Retry Logic](#retry-logic)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Setup (One Time)

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

### 2. Basic Usage

```python
from newspaper_explorer.llm.client import LLMClient
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse

# Text to analyze
text = "Kaiser Wilhelm II empfing Bernhard von Bülow in Berlin."

# Format prompt
prompt = ENTITY_EXTRACTION
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

### 3. Temperature Guidelines

- **0.0-0.3**: Deterministic (extraction, classification)
- **0.5-0.7**: Balanced (analysis, summarization)
- **0.8-1.2**: Creative (generation, brainstorming)

---

## Configuration

### Environment Variables

All LLM settings are configured via `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_BASE_URL` | Yes | - | API endpoint URL |
| `LLM_API_KEY` | Yes | - | Authentication token |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model identifier |
| `LLM_TEMPERATURE` | No | `0.7` | Default temperature |
| `LLM_MAX_TOKENS` | No | `2000` | Default token limit |

### Supported API Endpoints

Any OpenAI-compatible API:

- **OpenAI**: `https://api.openai.com/v1`
- **Azure OpenAI**: `https://your-resource.openai.azure.com/openai/deployments/your-deployment`
- **Local models** (e.g., Ollama): `http://localhost:11434/v1`
- **Other providers**: Any API following OpenAI's chat completion format

### Programmatic Override

```python
# Override environment settings
client = LLMClient(
    base_url="https://custom-api.com/v1",
    api_key="custom-key",
    model_name="gpt-4o",
    temperature=0.3,
    max_retries=5,
    retry_delay=2.0,
)

# Per-request override
response = client.complete(
    prompt="...",
    temperature=0.8,    # Override for this request
    max_tokens=1000
)
```

---

## Core Usage

### Architecture

**Direct imports** - No wrapper files. Import prompts and schemas directly:

```python
# Prompts
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.llm.prompts.topic_analysis import TOPIC_CLASSIFICATION
from newspaper_explorer.llm.prompts.emotion_analysis import EMOTION_ANALYSIS

# Schemas
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse
from newspaper_explorer.llm.schemas.topic_analysis import TopicClassificationResponse
from newspaper_explorer.llm.schemas.emotion_analysis import EmotionAnalysisResponse
```

### Basic Pattern

```python
from newspaper_explorer.llm.client import LLMClient
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse

# 1. Format prompt
prompt = ENTITY_EXTRACTION
formatted = prompt.format(text="Your text here...")

# 2. Make request with validation
with LLMClient() as client:
    response = client.complete(
        prompt=formatted["user"],
        system_prompt=formatted["system"],
        response_schema=EntityResponse,
    )

# 3. Use type-safe response
print(response.persons)      # List[str]
print(response.locations)    # List[str]
```

### Without Schema Validation

Get raw string response without validation:

```python
with LLMClient() as client:
    response = client.complete(
        prompt="Translate: Das Wetter ist schön.",
        system_prompt="You are a translator.",
    )
    # response is str, not validated model
```

### Error Handling

```python
from newspaper_explorer.llm.client import LLMRetryError, LLMValidationError

try:
    response = client.complete(
        prompt="...",
        response_schema=EntityResponse,
    )
except LLMRetryError:
    # All retry attempts exhausted
    logger.error("Request failed after retries")
except LLMValidationError:
    # Response doesn't match schema
    logger.error("Invalid response format")
### Batch Processing

```python
import polars as pl
from newspaper_explorer.llm.client import LLMClient
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse

results = []
with LLMClient(temperature=0.3) as client:
    for row in df.iter_rows(named=True):
        try:
            formatted = ENTITY_EXTRACTION.format(text=row["text"])
            response = client.complete(
                prompt=formatted["user"],
                system_prompt=formatted["system"],
                response_schema=EntityResponse,
            )
            results.append({
                "line_id": row["line_id"],
                "persons": response.persons,
                "locations": response.locations,
            })
        except Exception as e:
            logger.warning(f"Failed: {e}")

results_df = pl.DataFrame(results)
```

---

## Metadata Support

### Why Use Metadata?

Passing contextual information (source, date, newspaper title) improves LLM analysis quality by providing historical context.

### Standard Metadata Fields

| Field | Description | Example |
|-------|-------------|---------|
| `source` | Source identifier | `"der_tag"` |
| `newspaper_title` | Full newspaper title | `"Der Tag"` |
| `date` | Publication date (ISO) | `"1920-01-15"` |
| `year_volume` | Year/volume info | `"1920/15"` |
| `page_number` | Page number | `3` |

### Basic Usage

```python
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION

# Create metadata
metadata = {
    "source": "Der Tag",
    "date": "1920-01-15",
    "page_number": 3,
}

# Format with metadata
formatted = ENTITY_EXTRACTION.format(text="...", metadata=metadata)

# Metadata is available for variable substitution in prompts
```

### Three Metadata Patterns

**1. Variable Substitution Only**

```python
prompt = PromptTemplate(
    system="Analyzing {source} from {date}, page {page_number}.",
    user="{text}"
)
```

**2. Auto Context Only**

```python
prompt = PromptTemplate(
    system="You are a historian.",
    user="{text}",
    include_metadata=True  # Appends context section
)
# Adds:
# Context:
# - Source: Der Tag
# - Publication Date: 1920-01-15
# - Page Number: 3
```

**3. Both (Recommended)**

```python
prompt = PromptTemplate(
    system="Analyzing {source}.",
    user="{text}",
    include_metadata=True  # Variables + context section
)
```

### DataFrame Integration

Metadata is automatically extracted from DataFrame columns:

```python
from newspaper_explorer.analysis.entities.llm_extraction import LLMEntityExtractor

# DataFrame has: line_id, text, source, newspaper_title, date, page_number
extractor = LLMEntityExtractor(source_name="der_tag")
results = extractor.extract_from_dataframe(df)
# Metadata automatically passed to each LLM call
```

---

## Available Prompts & Schemas

Import prompts and schemas directly from their modules:

### Entity Extraction

```python
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse

# Schema:
# {
#     "persons": ["name1", "name2"],
#     "locations": ["place1"],
#     "organizations": ["org1"]
# }
```

### Topic Analysis

```python
from newspaper_explorer.llm.prompts.topic_analysis import TOPIC_CLASSIFICATION, TOPIC_GENERATION
from newspaper_explorer.llm.schemas.topic_analysis import (
    TopicClassificationResponse,
    TopicGenerationResponse,
)
```

### Emotion Analysis

```python
from newspaper_explorer.llm.prompts.emotion_analysis import EMOTION_ANALYSIS
from newspaper_explorer.llm.schemas.emotion_analysis import EmotionAnalysisResponse

# Schema:
# {
#     "sentiment": "positive|negative|neutral|mixed",
#     "emotions": ["pride", "hope"],
#     "intensity": 0.8,
#     "tone": "patriotic"
# }
```

### Concept Extraction

```python
from newspaper_explorer.llm.prompts.concept_extraction import CONCEPT_EXTRACTION
from newspaper_explorer.llm.schemas.concept_extraction import ConceptExtractionResponse

# Schema:
# {
#     "concepts": ["Industrialisierung", "Modernisierung"],
#     "relationships": [
#         {"source": "...", "target": "...", "type": "leads_to"}
#     ]
# }
```

### Summarization

```python
from newspaper_explorer.llm.prompts.summarization import SUMMARIZATION
from newspaper_explorer.llm.schemas.summarization import SummarizationResponse
```

### Text Quality

```python
from newspaper_explorer.llm.prompts.text_quality import TEXT_QUALITY
from newspaper_explorer.llm.schemas.text_quality import TextQualityResponse
```

---

## Custom Prompts & Schemas

### Creating Custom Prompts

Save to `llm/prompts/my_analysis.py`:

```python
from newspaper_explorer.llm.prompts.base import PromptTemplate

MY_CUSTOM_PROMPT = PromptTemplate(
    system="You are analyzing historical {source} newspapers from {date}.",
    user="Task: {task}\n\nText: {text}",
    include_metadata=True  # Optional: auto-append metadata
)
```

Use it:

```python
from newspaper_explorer.llm.prompts.my_analysis import MY_CUSTOM_PROMPT

formatted = MY_CUSTOM_PROMPT.format(
    text="...",
    task="Identify economic indicators",
    metadata={"source": "Der Tag", "date": "1920-01-15"}
)
```

### Creating Custom Schemas

Save to `llm/schemas/my_analysis.py`:

```python
from pydantic import BaseModel, Field

class MyCustomResponse(BaseModel):
    """Custom analysis response."""
    
    field1: str = Field(description="...")
    field2: int = Field(ge=0, le=100, description="Score 0-100")
    field3: list[str] = Field(default_factory=list)
```

Use it:

```python
from newspaper_explorer.llm.schemas.my_analysis import MyCustomResponse

response = client.complete(
    prompt="...",
    response_schema=MyCustomResponse,
)
# Returns validated MyCustomResponse instance
```

---

## Retry Logic

### Default Behavior

- **Max retries**: 3 attempts
- **Retry delay**: 1 second (doubles each attempt)
- **Retryable errors**: 429 (rate limit), 500, 502, 503, 504, timeouts
- **Non-retryable**: 400 (bad request), 401 (unauthorized), 403 (forbidden)

### Exponential Backoff

Delays between retries double each attempt:

- Attempt 1: `retry_delay` (e.g., 1s)
- Attempt 2: `retry_delay * 2` (e.g., 2s)
- Attempt 3: `retry_delay * 4` (e.g., 4s)

### Customization

```python
client = LLMClient(
    max_retries=5,
    retry_delay=2.0,  # Initial delay in seconds
    timeout=60.0,     # Request timeout
)
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# ✅ Good - automatic cleanup
with LLMClient() as client:
    response = client.complete(...)

# ❌ Avoid - manual cleanup
client = LLMClient()
response = client.complete(...)
client.close()
```

### 2. Always Use Schema Validation

```python
# ✅ Good - type-safe with IDE support
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse

response = client.complete(
    prompt="...",
    response_schema=EntityResponse,
)
print(response.persons)  # IDE autocomplete

# ❌ Avoid - manual parsing
response = client.complete(prompt="...")
data = json.loads(response)
```

### 3. Always Pass Metadata

```python
# ✅ Good - provides context
metadata = {
    "source": row.get("source"),
    "newspaper_title": row.get("newspaper_title"),
    "date": row.get("date"),
}
formatted = prompt.format(text=text, metadata=metadata)

# ❌ Less ideal - missing context
formatted = prompt.format(text=text)
```

### 4. Handle Errors Gracefully

```python
from newspaper_explorer.llm.client import LLMRetryError, LLMValidationError

try:
    response = client.complete(...)
except LLMRetryError:
    # Fallback to cached or default
    logger.error("Request failed after retries")
except LLMValidationError:
    # Log and skip or use simpler prompt
    logger.error("Invalid response format")
```

### 5. Test With and Without Metadata

Compare results to verify metadata improves quality:

```python
# Without metadata
result1 = extract_entities(text="...")

# With metadata  
result2 = extract_entities(
    text="...",
    metadata={"source": "Der Tag", "date": "1920-01-15"}
)
```

---

## Troubleshooting

### "base_url must be provided"

**Cause**: Missing `LLM_BASE_URL` environment variable.

**Solution**: Add to `.env`:

```bash
LLM_BASE_URL=https://api.openai.com/v1
```

### "api_key must be provided"

**Cause**: Missing `LLM_API_KEY` environment variable.

**Solution**: Add to `.env`:

```bash
LLM_API_KEY=sk-your-key-here
```

### Rate Limit Errors (429)

**Cause**: Too many requests to API.

**Solution**: Increase retry settings:

```python
client = LLMClient(max_retries=10, retry_delay=5.0)
```

### Validation Errors

**Cause**: LLM response doesn't match expected schema.

**Solution**: Enable debug logging to see raw response:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Timeout Errors

**Cause**: Request takes too long.

**Solution**: Increase timeout:

```python
client = LLMClient(timeout=120.0)
```

### Import Errors

**Cause**: Using old import patterns.

**Solution**: Use direct imports:

```python
# ✅ Correct
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.llm.schemas.entity_extraction import EntityResponse

# ❌ Old pattern (removed)
from newspaper_explorer.llm.prompts import get_prompt
from newspaper_explorer.llm.schemas import EntityResponse
```

---

## Module Structure

```
src/newspaper_explorer/llm/
├── client.py                    # LLMClient with retry & validation
├── prompts/
│   ├── base.py                 # PromptTemplate base class
│   ├── entity_extraction.py    # ENTITY_EXTRACTION
│   ├── topic_analysis.py       # TOPIC_CLASSIFICATION, TOPIC_GENERATION
│   ├── emotion_analysis.py     # EMOTION_ANALYSIS
│   ├── concept_extraction.py   # CONCEPT_EXTRACTION
│   ├── summarization.py        # SUMMARIZATION
│   └── text_quality.py         # TEXT_QUALITY
└── schemas/
    ├── entity_extraction.py    # EntityResponse
    ├── topic_analysis.py       # Topic*Response
    ├── emotion_analysis.py     # EmotionAnalysisResponse
    ├── concept_extraction.py   # ConceptExtractionResponse
    ├── summarization.py        # SummarizationResponse
    └── text_quality.py         # TextQualityResponse
```

**Key Points**:

- **No `__init__.py` files** - Use explicit imports
- **Direct imports** - Import from module, not wrapper
- **One file per concept** - Each prompt/schema in its own file
