import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from core.config import CONFIG

logger = logging.getLogger(__name__)

class QdrantMemory:
    def __init__(self):
        self.collection_name = "trades_journal"
        
        try:
            # Connect to Qdrant on Linux LAN
            self.client = QdrantClient(host=CONFIG.QDRANT_HOST, port=CONFIG.QDRANT_PORT)
            logger.info(f"Connected to Qdrant at {CONFIG.QDRANT_HOST}:{CONFIG.QDRANT_PORT}")
            
            # Setup FastEmbed (runs locally on this machine)
            self.client.set_model("BAAI/bge-small-en-v1.5")
            
            # Ensure collection exists
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize QdrantMemory: {e}")
            self.client = None
            
    def _ensure_collection(self):
        """Creates the collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating new Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.client.get_fastembed_vector_params()
            )

memory_db = QdrantMemory()
