import asyncio
import logging
import numpy as np
from PIL import Image
import easyocr

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(self, settings):
        self.settings = settings
        # parse comma-separated languages or default to English
        langs = settings.ocr_languages.split(',') if getattr(settings, 'ocr_languages', None) else ['en']

        logger.info(f"Initializing EasyOCR with languages: {langs}")
        # Let EasyOCR automatically detect and use GPU if available
        self.reader = easyocr.Reader(langs, gpu=True)

    def _extract_text(self, image: Image.Image) -> str:
        # WARN: synchronous function
        img_array = np.array(image.convert('RGB'))

        results = self.reader.readtext(img_array)

        # Post-processing extracts just the text strings, filter empty
        lines = [res[1].strip() for res in results if res[1].strip()]
        return '\n'.join(lines)

    async def process_image(self, image: Image.Image) -> str:
        """
        Process image with OCR asynchronously

        Args:
            image: PIL Image object

        Returns:
            Extracted text
        """
        try:
            # Run CPU/GPU bound OCR task in a separate thread
            text = await asyncio.to_thread(self._extract_text, image)
            return text
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            raise
