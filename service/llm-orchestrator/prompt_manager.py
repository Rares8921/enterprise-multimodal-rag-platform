import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptManager:

    def __init__(self, prompts_dir: str = "specialized_prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}

    def _build_template(self, prompt_data: dict) -> str:
        return f"""ROLE:
{prompt_data.get('role', '')}

RULES & CONSTRAINTS:
{prompt_data.get('rules', '')}

TASK:
{prompt_data.get('task', '')}

OUTPUT FORMAT:
{prompt_data.get('output_format', '')}

CONTEXT:
{{context}}

USER QUESTION:
{{query}}

RESPONSE:"""

    def get_prompt_template(self, doc_type: str) -> str:
        if doc_type in self._cache:
            return self._cache[doc_type]

        file_path = self.prompts_dir / f"{doc_type}.json"

        # Fallback to generic if specialized prompt doesn't exist
        if not file_path.exists():
            logger.warning(f"Prompt '{doc_type}' not found. Falling back to generic.")
            file_path = self.prompts_dir / "generic.json"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)

            template = self._build_template(prompt_data)
            self._cache[doc_type] = template
            return template

        except Exception as e:
            logger.error(f"Failed to load prompt template from {file_path}: {e}")
            raise

# Usage example:
# prompt_manager = PromptManager()
# template = prompt_manager.get_prompt_template('legal_contract')
# final_prompt = template.format(context=retrieved_chunks, query=user_input)