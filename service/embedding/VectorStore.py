from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    "Pinecone vector store"

    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int):
        logger.info(f"Initializing Pinecone: {index_name}")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension

        # Create index if it doesn't exist
        if index_name not in [idx['name'] for idx in self.pc.list_indexes()]:
            logger.info(f"Creating index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=environment
                )
            )

        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")

    def upsert_vectors(self, vectors: List[Dict[str, Any]], namespace: str = ""):
        """

        :param vectors: list of (id, vector, metadata) tuples
        :param namespace: pinecone namespace (i.e tenant_id)
        :return: None
        """

        try:
            pinecone_vectors = [
                (
                    vec['id'],
                    vec['values'].tolist() if isinstance(vec['values'], np.ndarray) else vec['values'],
                    vec.get('metadata', {})
                )
                for vec in vectors
            ]

            self.index.upsert(vectors=pinecone_vectors, namespace=namespace)
            logger.info(f"Upserted {len(vectors)} vectors to namespace: {namespace}")
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise

    def query(self, query_vector: np.ndarray, namespace: str = "", top_k: int = 10, filter: Dict = None) -> List[Dict]:
        # query similar vectors
        try:
            results = self.index.query(
                vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=True
            )

            return results['matches']
        except Exception as e:
            logger.error(f"Error querying vectors: {str(e)}")
            raise

    def delete_namespace(self, namespace: str):
        # delete all vectors in a namespace
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted namespace: {namespace}")
        except Exception as e:
            logger.error(f"Error deleting namespace: {str(e)}")