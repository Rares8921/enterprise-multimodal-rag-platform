import torch
import numpy as np
from typing import List, Dict, Any
import logging

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for document chunks"""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading embedding model on {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding model loaded: {model_name} (dim={self.embedding_dim})")

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text chunks
            batch_size: Batch size for processing

        Returns:
            Numpy array of embeddings (N x embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 norm for cos similarity
        )

    def chunk_document(self, text: str, layout_structure: Dict, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Chunk text into chunks of size chunk_size
        :param text: full document text
        :param layout_structure: parsed layout structure
        :param chunk_size: target chunk size in tokens
        :param overlap: overlap between chunks
        :return: list of chunks with metadata
        """

        chunks = []
        # chunk by semantic units (sections, paragraphs) for preserving the context
        for page_structure in layout_structure.get('structures', []):
            page_num = page_structure['page_number']

            # Process titles
            for title in page_structure.get('titles', []):
                chunks.append({
                    'text': title['text'],
                    'page': page_num,
                    'type': 'title',
                    'bbox': title.get('bbox'),
                })

            # Sections
            for section in page_structure.get('sections', []):
                # Split long sec.
                section_text = section['text']
                if len(section_text.split()) > chunk_size:
                    words = section_text.split()
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk_text = ' '.join(words[i:i + chunk_size])
                        chunks.append({
                            'text': chunk_text,
                            'page': page_num,
                            'type': 'section',
                            'bbox': section.get('bbox'),
                        })
                else:
                    chunks.append({
                        'text': section_text,
                        'page': page_num,
                        'type': section,
                        'bbox': section.get('bbox')
                    })

            # tables
            for table in page_structure.get('tables', []):
                chunks.append({
                    'text': table['text'],
                    'page': page_num,
                    'type': 'table',
                    'bbox': table.get('bbox')
                })

            # remaining text or body text
            body_texts = [bt['text'] for bt in page_structure.get('body_text', [])]
            if body_texts:
                combined_text = ' '.join(body_texts)
                words = combined_text.split()

                for i in range(0, len(words), chunk_size - overlap):
                    chunk_text = ' '.join(words[i:i + chunk_size])
                    chunks.append({
                        'text': chunk_text,
                        'page': page_num,
                        'type': 'body',
                    })

        return chunks

