# LLM Utils Quick Reference

## Setup (One Time)

```bash
# Add to .env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key-here
```

## Basic Usage

```python
from newspaper_explorer.utils.llm import LLMClient
from newspaper_explorer.utils.prompts import get_prompt
from newspaper_explorer.utils.schemas import EntityResponse

# Get prompt
prompt = get_prompt("entity_extraction")
formatted = prompt.format(text="Your text here...")

# Make request
with LLMClient() as client:
    response = client.complete(
        prompt=formatted["user"],
        system_prompt=formatted["system"],
        response_schema=EntityResponse,
    )

# Use response
print(response.persons)  # Type-safe!
```

## Available Prompts

| Prompt Name | Use For | Schema |
|-------------|---------|--------|
| `entity_extraction` | Extract persons, locations, orgs | `EntityResponse` |
| `topic_classification` | Classify into topics | `TopicClassificationResponse` |
| `topic_generation` | Generate topic labels | `TopicGenerationResponse` |
| `emotion_analysis` | Analyze sentiment/emotions | `EmotionAnalysisResponse` |
| `concept_extraction` | Extract concepts + relationships | `ConceptExtractionResponse` |
| `summarization` | Summarize text | `SummarizationResponse` |
| `text_quality` | Assess OCR quality | `TextQualityResponse` |

## Custom Parameters

```python
# Override defaults
client = LLMClient(
    model_name="gpt-4o",
    temperature=0.3,
    max_tokens=1000,
    max_retries=5,
    retry_delay=2.0,
)

# Or per request
response = client.complete(
    prompt="...",
    temperature=0.8,  # Override just for this
)
```

## Without Schema (Raw String)

```python
# No validation, returns string
response = client.complete(
    prompt="Translate: Guten Tag",
    system_prompt="You are a translator.",
)
print(response)  # str
```

## Error Handling

```python
from newspaper_explorer.utils.llm import LLMRetryError, LLMValidationError

try:
    response = client.complete(...)
except LLMRetryError:
    # Failed after retries
    logger.error("Request failed")
except LLMValidationError:
    # Response doesn't match schema
    logger.error("Invalid format")
```

## Temperature Guidelines

- **0.0-0.3**: Deterministic (extraction, classification)
- **0.5-0.7**: Balanced (analysis, summarization)
- **0.8-1.2**: Creative (generation, brainstorming)

## List Available Options

```python
from newspaper_explorer.utils.prompts import list_prompts
from newspaper_explorer.utils.schemas import list_schemas

print(list_prompts())   # All available prompts
print(list_schemas())   # All available schemas
```

## Common Patterns

### Batch Processing

```python
import polars as pl

results = []
with LLMClient() as client:
    for row in df.iter_rows(named=True):
        try:
            response = client.complete(...)
            results.append({...})
        except Exception as e:
            logger.warning(f"Failed: {e}")

results_df = pl.DataFrame(results)
```

### Custom Prompt

```python
from newspaper_explorer.utils.prompts import PromptTemplate

my_prompt = PromptTemplate(
    system="You are an expert.",
    user="Analyze: {text}"
)

formatted = my_prompt.format(text="...")
response = client.complete(
    prompt=formatted["user"],
    system_prompt=formatted["system"],
)
```

## Troubleshooting

**"base_url must be provided"**  
→ Set `LLM_BASE_URL` in `.env`

**"api_key must be provided"**  
→ Set `LLM_API_KEY` in `.env`

**Rate limit errors**  
→ Increase `max_retries` and `retry_delay`

**Timeout errors**  
→ Increase `timeout` parameter

**Validation errors**  
→ Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

## Files

- **Client**: `src/newspaper_explorer/utils/llm.py`
- **Prompts**: `src/newspaper_explorer/utils/prompts.py`
- **Schemas**: `src/newspaper_explorer/utils/schemas.py`
- **Examples**: `src/newspaper_explorer/utils/llm_examples.py`
- **Docs**: `docs/LLM.md`
