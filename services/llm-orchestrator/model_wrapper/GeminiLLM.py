import google.generativeai as genai
import asyncio
from typing import Dict, Any

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLM:

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        logger.info("Gemini model initialized: %s", model_name)

    async def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> Dict[str, Any]:
        try:
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=generation_config
            )

            # Extract text and metadata
            text = response.text

            # Token counting (Gemini provides usage metadata)
            usage = {
                'input_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                'output_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else len(text.split()),
                'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else len(text.split())
            }

            return {
                'text': text,
                'usage': usage
            }

        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            raise