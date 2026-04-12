import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
from typing import Dict, List, Any
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _save_element(elements: Dict, element_type: str, element: Dict):
    if element_type in elements:
        elements[element_type].append(element)


def _merge_boxes(box1: List[int], box2: List[int]) -> List[int]:
    return [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3])
    ]


class LayoutParser:
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        global device

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device

        logger.info(f"Loading LayoutLM model on {self.device}")

        # Load processor and model
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            0: "O", # Outside
            1: "B-TITLE", # Begin-Title
            2: "I-TITLE", # Inside-Title
            3: "B-SECTION",
            4: "I-SECTION",
            5: "B-TABLE",
            6: "I-TABLE",
            7: "B-LIST",
            8: "I-LIST",
            9: "B-HEADER",
            10: "I-HEADER",
            11: "B-FOOTER",
            12: "I-FOOTER",
            13: "B-SIGNATURE",
            14: "I-SIGNATURE"
        }

        logger.info("LayoutLM model loaded successfully")

    @torch.no_grad()
    def parse_document(self, image: Image.Image, words: List[str], boxes: List[List[int]]) -> Dict[str, Any]:
        # WARN: MAKE SURE TO RUN IN A THREAD POOL
        # parse document layout and extract structured elements
        try:
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                return_tensors="pt", # pytorch
                padding="max_length",
                truncation=True,
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # inference
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()

            # predictions to labels
            tokens = self.processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())

            # Parse structure
            structure = self._parse_structure(tokens, predictions, words, boxes)

            return structure
        except Exception as e:
            logger.error(f"Layout parsing error: {str(e)}")
            raise

    def _parse_structure(self, tokens: List[str], predictions: List[int], words: List[str], boxes: List[List[int]]) -> Dict[str, Any]:
        # token predictions -> structured elements
        elements = {
            'titles': [],
            'sections': [],
            'tables': [],
            'lists': [],
            'headers': [],
            'footers': [],
            'signatures': [],
            'body_text': []
        }

        current_element = None
        current_type = None

        for token, pred_id, word, box in zip(tokens, predictions, words, boxes):
            if token.startswith('##'):
                continue

            label = self.id2label.get(pred_id, "O")

            if label.startswith('B-'):
                if current_element:
                    _save_element(elements, current_type, current_element)

                current_type = label[2:].lower() + 's'
                current_element = {
                    'text': word,
                    'bbox': box,
                    'tokens': [token]
                }

            elif label.startswith('I-') and current_element:
                current_element['text'] += ' ' + word
                current_element['tokens'].append(token)
                current_element['bbox'] = _merge_boxes(current_element['bbox'], box)

            elif label == 'O':
                if current_element:
                    _save_element(elements, current_type, current_element)
                    current_element = None
                    current_type = None

                elements['body_text'].append({
                    'text': word,
                    'bbox': box
                })

        if current_element:
            _save_element(elements, current_type, current_element)

        return elements

