import httpx
from typing import Dict, Any

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralLLM:
    def __init__(self, api_url: str, client: httpx.AsyncClient):
        self.api_url = api_url
        self.client = client
        logger.info(f"Mistral client initialized: {api_url}")

    async def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> Dict[str, Any]:
        try:
            response = await self.client.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    'model': 'mistral-8x7b',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': temperature,
                    'max_tokens': max_tokens
                },
                timeout=60.0
            )

            response.raise_for_status()
            data = response.json()

            text = data['choices'][0]['message']['content']
            usage = data.get('usage', {})

            return {
                'text': text,
                'usage': {
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }
            }

        except Exception as e:
            logger.error(f"Mistral generation error: {str(e)}")
            raise
