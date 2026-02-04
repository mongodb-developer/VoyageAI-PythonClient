import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import List, Dict
import voyageai
from pymongo import MongoClient
from datetime import datetime

class MultiKeyEmbeddingPipeline:
    def __init__(
        self,
        api_keys: <span class="hljs-type">List[str],
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        batch_size: int = 128,  # VoyageAI max batch size
        workers_per_key: int = 3
    </span>):
        self.api_keys = api_keys
        self.key_cycle = cycle(api_keys)
        self.batch_size = batch_size
        self.mongo_client = MongoClient(mongo_uri)
        self.collection = self.mongo_client[db_name][collection_name]
        self.executor = ThreadPoolExecutor(max_workers=len(api_keys) * workers_per_key)
        
        # Create VoyageAI clients for each key
        self.clients = [voyageai.Client(api_key=key) for key in api_keys]
        self.client_cycle = cycle(self.clients)
        
    def get_next_client(self) -> voyageai.Client:
        """Round-robin client selection to distribute load across keys"""
        return next(self.client_cycle)
    
    def embed_batch(self, documents: <span class="hljs-type">List[Dict], client: voyageai.Client</span>) -> List[Dict]:
        """Embed a batch of documents using specified client"""
        try:
            texts = [doc['text_field'] for doc in documents]  # Adjust field name
            embeddings = client.embed(
                texts, 
                model="voyage-2",  # or your specific model
                input_type="document"
            ).embeddings
            
            # Attach embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc['embedding'] = embedding
                doc['embedded_at'] = datetime.utcnow()
            
            return documents
        except Exception as e:
            print(f"Error embedding batch: <span class="hljs-subst">{e}"</span>)
            raise
    
    def update_mongodb(self, documents: <span class="hljs-type">List[Dict]</span>):
        """Bulk update MongoDB with embeddings"""
        from pymongo import UpdateOne
        
        operations = [
            UpdateOne(
                {'_id': doc['_id']},
                {'$set': {
                    'embedding': doc['embedding'],
                    'embedded_at': doc['embedded_at']
                }}
            )
            for doc in documents
        ]
        
        self.collection.bulk_write(operations, ordered=False)
    
    def process_batch(self, documents: <span class="hljs-type">List[Dict]</span>):
        """Process a single batch: embed + update MongoDB"""
        client = self.get_next_client()
        embedded_docs = self.embed_batch(documents, client)
        self.update_mongodb(embedded_docs)
        return len(embedded_docs)
    
    async def run(self, query_filter: <span class="hljs-type">Dict = None, skip: int = 0</span>):
        """Main processing loop with parallel execution"""
        query = query_filter or {'embedding': {'$exists': False}}  # Only unembed docs
        
        cursor = self.collection.find(query).skip(skip).batch_size(self.batch_size * 10)
        total_processed = 0
        batch = []
        futures = []
        
        for doc in cursor:
            batch.append(doc)
            
            if len(batch) >= self.batch_size:
                # Submit batch to thread pool
                future = self.executor.submit(self.process_batch, batch.copy())
                futures.append(future)
                batch = []
                
                # Control concurrency (adjust based on memory)
                if len(futures) >= len(self.api_keys) * 10:
                    # Wait for some to complete
                    completed = 0
                    for f in futures[:len(self.api_keys) * 5]:
                        completed += f.result()
                    total_processed += completed
                    futures = futures[len(self.api_keys) * 5:]
                    print(f"Processed <span class="hljs-subst">{total_processed} documents..."</span>)
        
        # Process remaining batch
        if batch:
            total_processed += self.process_batch(batch)
        
        # Wait for all remaining futures
        for f in futures:
            total_processed += f.result()
        
        print(f"Total processed: <span class="hljs-subst">{total_processed}"</span>)
        return total_processed


# Usage
if __name__ == "__main__":
    # Setup multiple API keys
    API_KEYS = [
        os.getenv('VOYAGE_API_KEY_1'),
        os.getenv('VOYAGE_API_KEY_2'),
        os.getenv('VOYAGE_API_KEY_3'),
        os.getenv('VOYAGE_API_KEY_4'),
        # Add more keys as needed
    ]
    
    pipeline = MultiKeyEmbeddingPipeline(
        api_keys=API_KEYS,
        mongo_uri=os.getenv('MONGO_URI'),
        db_name='your_db',
        collection_name='your_collection',
        batch_size=128,
        workers_per_key=3
    )
    
    # Run the pipeline
    asyncio.run(pipeline.run())
