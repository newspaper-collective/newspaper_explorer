"""
LLM client utilities with configurable endpoints, retries, and validation.

Provides a flexible HTTP-based client for interacting with LLM APIs (OpenAI-compatible).
Supports retry logic, response validation against Pydantic schemas, and centralized configuration.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import requests
from pydantic import BaseModel, ValidationError

from newspaper_explorer.utils.config import get_config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMValidationError(LLMError):
    """Raised when LLM response fails schema validation."""

    pass


class LLMRetryError(LLMError):
    """Raised when all retry attempts are exhausted."""

    pass


class LLMClient:
    """
    Flexible LLM client using requests library for OpenAI-compatible APIs.

    Supports:
    - Configurable base URL and API key
    - Retry logic with exponential backoff
    - Response validation against Pydantic schemas
    - Temperature, max_tokens, and other model parameters

    Example:
        ```python
        from newspaper_explorer.utils.llm import LLMClient
        from newspaper_explorer.utils.schemas import EntityResponse

        client = LLMClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model_name="gpt-4o-mini",
            temperature=0.7
        )

        response = client.complete(
            prompt="Extract entities from: Berlin ist die Hauptstadt.",
            response_schema=EntityResponse
        )
        ```
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
    ):
        """
        Initialize LLM client.

        Args:
            base_url: Base URL for API endpoint (e.g., "https://api.openai.com/v1").
                     If None, reads from config or environment.
            api_key: API authentication key. If None, reads from config or environment.
            model_name: Model identifier (e.g., "gpt-4o-mini", "gpt-4").
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random).
            max_tokens: Maximum tokens in response.
            max_retries: Number of retry attempts on failure.
            retry_delay: Initial delay between retries (seconds, exponential backoff).
            timeout: Request timeout in seconds.
        """
        config = get_config()

        self.base_url = (base_url or config.get("llm_base_url", "")).rstrip("/")
        self.api_key = api_key or config.get("llm_api_key", "")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        if not self.base_url:
            raise ValueError("base_url must be provided or configured in environment")
        if not self.api_key:
            raise ValueError("api_key must be provided or configured in environment")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        logger.debug(
            f"Initialized LLMClient: model={model_name}, "
            f"temp={temperature}, max_tokens={max_tokens}"
        )

    def complete(
        self,
        prompt: str,
        response_schema: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Union[str, T]:
        """
        Send completion request to LLM with optional validation.

        Args:
            prompt: User prompt/instruction.
            response_schema: Optional Pydantic model for response validation.
                           If provided, returns validated model instance.
            system_prompt: Optional system message for model behavior.
            temperature: Override default temperature for this request.
            max_tokens: Override default max_tokens for this request.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            Raw string response if no schema provided, otherwise validated model instance.

        Raises:
            LLMRetryError: If all retry attempts fail.
            LLMValidationError: If response fails schema validation after retries.
        """
        messages = self._build_messages(system_prompt, prompt)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs,
        }

        # If schema provided, request JSON response
        if response_schema:
            payload["response_format"] = {"type": "json_object"}

        response_text = self._request_with_retry(payload)

        # Validate against schema if provided
        if response_schema:
            return self._validate_response(response_text, response_schema)

        return response_text

    def _build_messages(
        self, system_prompt: Optional[str], user_prompt: str
    ) -> List[Dict[str, str]]:
        """Build messages array for chat completion."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _request_with_retry(self, payload: Dict[str, Any]) -> str:
        """
        Execute API request with exponential backoff retry logic.

        Args:
            payload: Request payload dictionary.

        Returns:
            Response text from API.

        Raises:
            LLMRetryError: If all retries are exhausted.
        """
        endpoint = f"{self.base_url}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM request attempt {attempt + 1}/{self.max_retries}")

                response = self.session.post(
                    endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]

                logger.debug(f"LLM request successful (attempt {attempt + 1})")
                return content

            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                if response.status_code in (429, 500, 502, 503, 504):
                    # Retry on rate limit or server errors
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2**attempt)
                        logger.info(f"Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                raise LLMRetryError(f"HTTP error after {attempt + 1} attempts: {e}")

            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                raise LLMRetryError(f"Timeout after {attempt + 1} attempts: {e}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                raise LLMRetryError(f"Request failed after {attempt + 1} attempts: {e}")

            except (KeyError, ValueError) as e:
                # Response parsing errors - don't retry
                logger.error(f"Failed to parse API response: {e}")
                raise LLMError(f"Invalid API response format: {e}")

        raise LLMRetryError(f"All {self.max_retries} retry attempts exhausted")

    def _validate_response(self, response_text: str, schema: Type[T]) -> T:
        """
        Validate response text against Pydantic schema.

        Args:
            response_text: Raw JSON response string.
            schema: Pydantic model class.

        Returns:
            Validated model instance.

        Raises:
            LLMValidationError: If validation fails.
        """
        import json

        try:
            data = json.loads(response_text)
            validated = schema.model_validate(data)
            logger.debug(f"Response validated against {schema.__name__}")
            return validated

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in response: {e}")
            raise LLMValidationError(f"Response is not valid JSON: {e}")

        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            raise LLMValidationError(f"Response does not match schema: {e}")

    def close(self):
        """Close the underlying session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
