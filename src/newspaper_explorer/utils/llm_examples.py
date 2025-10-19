"""
Example usage of LLM utilities with prompts and schemas.

Demonstrates how to use the LLMClient with centralized prompt templates
and response validation.
"""

from newspaper_explorer.utils.llm import LLMClient
from newspaper_explorer.utils.prompts import get_prompt
from newspaper_explorer.utils.schemas import EntityResponse, EmotionAnalysisResponse


def example_entity_extraction():
    """Extract entities from German newspaper text."""
    # Sample text
    text = """
    Berlin, 15. Januar 1901. Kaiser Wilhelm II empfing heute den Reichskanzler 
    Bernhard von Bülow im Berliner Stadtschloss. Die Besprechung dauerte zwei Stunden 
    und befasste sich hauptsächlich mit der Außenpolitik des Deutschen Reiches.
    """

    # Get prompt template
    prompt_template = get_prompt("entity_extraction")
    prompts = prompt_template.format(text=text)

    # Initialize client (reads from .env or config)
    with LLMClient(temperature=0.3) as client:
        # Make request with validation
        response = client.complete(
            prompt=prompts["user"],
            system_prompt=prompts["system"],
            response_schema=EntityResponse,
        )

        # Response is validated EntityResponse instance
        print(f"Persons: {response.persons}")
        print(f"Locations: {response.locations}")
        print(f"Organizations: {response.organizations}")


def example_emotion_analysis():
    """Analyze emotional tone of newspaper text."""
    text = """
    Mit großer Freude und Stolz empfing das deutsche Volk die Nachricht vom Sieg 
    unserer Truppen. Die patriotische Begeisterung kennt keine Grenzen.
    """

    prompt_template = get_prompt("emotion_analysis")
    prompts = prompt_template.format(text=text)

    with LLMClient(temperature=0.5) as client:
        response = client.complete(
            prompt=prompts["user"],
            system_prompt=prompts["system"],
            response_schema=EmotionAnalysisResponse,
        )

        print(f"Sentiment: {response.sentiment}")
        print(f"Emotions: {', '.join(response.emotions)}")
        print(f"Intensity: {response.intensity:.2f}")
        print(f"Tone: {response.tone}")


def example_without_validation():
    """Use LLM without schema validation (returns raw string)."""
    with LLMClient(model_name="gpt-4o", temperature=0.8) as client:
        response = client.complete(
            prompt="Übersetze ins Englische: Das Wetter ist schön heute.",
            system_prompt="Du bist ein Übersetzer.",
        )

        # Response is raw string
        print(f"Translation: {response}")


def example_custom_parameters():
    """Use custom parameters for specific request."""
    text = "Eine sehr lange Zeitungsartikel..."

    with LLMClient() as client:
        # Override default parameters for this request
        response = client.complete(
            prompt=f"Fasse zusammen: {text}",
            temperature=0.3,  # More deterministic for summaries
            max_tokens=500,  # Limit summary length
        )

        print(f"Summary: {response}")


def example_error_handling():
    """Demonstrate error handling with retries."""
    from newspaper_explorer.utils.llm import LLMRetryError, LLMValidationError

    try:
        with LLMClient(max_retries=5, retry_delay=2.0) as client:
            response = client.complete(
                prompt="Analyze this: ...",
                response_schema=EntityResponse,
            )
    except LLMRetryError as e:
        print(f"Failed after retries: {e}")
    except LLMValidationError as e:
        print(f"Response validation failed: {e}")


if __name__ == "__main__":
    # Run examples (requires .env with LLM_BASE_URL and LLM_API_KEY)
    print("=== Entity Extraction ===")
    example_entity_extraction()

    print("\n=== Emotion Analysis ===")
    example_emotion_analysis()

    print("\n=== Without Validation ===")
    example_without_validation()
