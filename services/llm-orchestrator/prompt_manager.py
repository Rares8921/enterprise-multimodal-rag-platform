import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptManager:

    def __init__(self, prompts_dir: str = "specialized_prompts"):
        prompts_path = Path(prompts_dir)
        if not prompts_path.is_absolute():
            service_relative = Path(__file__).resolve().parent / prompts_path
            if service_relative.exists():
                prompts_path = service_relative
        self.prompts_dir = prompts_path
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

    def get_prompt_template(self, doc_type: str, agent: str | None = None) -> str:
        agent_key = agent or "default"
        cache_key = f"{doc_type}__{agent_key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        candidate_files: list[Path] = []

        # 1) Most specific: doc_type + agent
        if agent and agent != "default":
            candidate_files.append(self.prompts_dir / f"{doc_type}__{agent}.json")
            # 2) Generic agent prompt as fallback
            candidate_files.append(self.prompts_dir / f"generic__{agent}.json")

        # 3) doc_type default
        candidate_files.append(self.prompts_dir / f"{doc_type}.json")
        # 4) global generic
        candidate_files.append(self.prompts_dir / "generic.json")

        file_path = next((p for p in candidate_files if p.exists()), self.prompts_dir / "generic.json")
        if not file_path.exists():
            raise FileNotFoundError(f"No prompt templates found in {self.prompts_dir}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)

            template = self._build_template(prompt_data)
            self._cache[cache_key] = template
            return template

        except Exception as e:
            logger.error(f"Failed to load prompt template from {file_path}: {e}")
            raise

# Usage example:
# prompt_manager = PromptManager()
# template = prompt_manager.get_prompt_template('legal_contract')
# final_prompt = template.format(context=retrieved_chunks, query=user_input)