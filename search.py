import os
import uuid
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

class VectorSearchSystem:
    def __init__(
        self, 
        collection_name: str = "documents", 
        embedding_model: str = "all-MiniLM-L6-v2",
        connection_mode: str = "http",
        location: Optional[str] = None
    ):
        """
        Initialize the vector search system.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Name of the SentenceTransformer model to use
            connection_mode: One of "memory", "local", or "http"
            location: Path for local storage or URL for HTTP connection
        """
        # Initialize embedding model
        self.model = SentenceTransformer(embedding_model)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        self.collection_name = collection_name
        
        # Initialize Qdrant client based on connection mode
        if connection_mode == "memory":
            # In-memory mode for testing
            self.client = QdrantClient(":memory:")
        elif connection_mode == "local":
            # Local persistence mode
            if location:
                self.client = QdrantClient(path=location)
            else:
                self.client = QdrantClient(path="./local_vector_db")
        else:
            # HTTP connection to Qdrant server
            if location:
                self.client = QdrantClient(location)
            else:
                self.client = QdrantClient("http://localhost:6333")
        
        # Create collection if it doesn't exist
        self._create_collection(collection_name)
    
    def _create_collection(self, collection_name: str):
        """Create a Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        if not any(collection.name == collection_name for collection in collections):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000,  # Index after 20k vectors for better performance
                )
            )
            print(f"Created new collection: {collection_name}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text."""
        return self.model.encode(text).tolist()
    
    def add_document(
        self, 
        doc_id: str, 
        document: str, 
        metadata: Dict[str, Any] = None,
        chunk_size: int = 0
    ):
        """
        Add a document to the vector database.
        
        Args:
            doc_id: Unique identifier for the document
            document: Text content of the document
            metadata: Additional information about the document
            chunk_size: If > 0, split document into chunks of this size
        """
        # Prepare metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add document information to metadata
        base_payload = {
            "doc_id": doc_id,
            **metadata
        }
        
        # Handle chunking if requested
        if chunk_size > 0 and len(document) > chunk_size:
            chunks = self._chunk_document(document, chunk_size)
            points = []
            
            for i, chunk in enumerate(chunks):
                # Clone the metadata for each chunk
                chunk_payload = base_payload.copy()
                # Add chunk-specific information
                chunk_payload.update({
                    "text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                
                # Generate embedding for this chunk
                embedding = self._get_embedding(chunk)
                
                # Create point
                points.append(
                    models.PointStruct(
                        id=f"{doc_id}_{i}",
                        vector=embedding,
                        payload=chunk_payload
                    )
                )
            
            # Upload all chunks
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Document '{doc_id}' added to collection as {len(chunks)} chunks.")
            
        else:
            # Single document case
            # Generate embedding
            embedding = self._get_embedding(document)
            
            # Add complete text to payload
            payload = {
                "text": document,
                **base_payload
            }
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),  # Generate a random UUID
                        vector=embedding,
                        payload={
                            "doc_id": doc_id,  # Store original ID in payload
                            "text": document,
                            **metadata
                        }
                    )
                ]
            )
            print(f"Document '{doc_id}' added to collection.")
    
    def _chunk_document(self, text: str, chunk_size: int, overlap: int = 100) -> List[str]:
        """Split document into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to find a natural breaking point (sentence end)
            if end < len(text):
                # Find the last period, question mark, or exclamation point
                for punct in ['.', '!', '?', '\n\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start + chunk_size // 2:  # Only break if we've processed at least half the chunk
                        end = last_punct + 1
                        break
            
            # Add this chunk
            chunks.append(text[start:end])
            
            # Move start position, with overlap
            start = max(start + chunk_size - overlap, end - overlap)
        
        return chunks
    
    def add_document_from_file(
        self, 
        file_path: str, 
        metadata: Dict[str, Any] = None,
        chunk_size: int = 0
    ):
        """
        Read a document from a file and add it to the vector database.
        
        Args:
            file_path: Path to the document file
            metadata: Additional information about the document
            chunk_size: If > 0, split document into chunks of this size
        """
        # Use filename as document ID
        doc_id = os.path.basename(file_path)
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Generate file metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add file information to metadata
        file_metadata = {
            "filename": doc_id,
            "path": file_path,
            "size_bytes": os.path.getsize(file_path),
            "last_modified": os.path.getmtime(file_path),
            **metadata
        }
        
        # Add document
        self.add_document(doc_id, content, file_metadata, chunk_size)
    
    def search(
        self, 
        query: str, 
        limit: int = 5,
        filter_condition: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            filter_condition: Optional filter to apply during search
            
        Returns:
            List of matching documents with their similarity scores
        """
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Prepare filter if provided
        search_filter = None
        if filter_condition:
            search_filter = Filter(**filter_condition)
        
        # Search in Qdrant
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        ).points
        
        # Format results
        formatted_results = []
        for result in results:
            # Extract text and document ID from payload
            payload = result.payload or {}
            text = payload.get("text", "")
            doc_id = payload.get("doc_id", str(result.id))
            
            # Create a clean metadata dictionary without text and doc_id
            metadata = {k: v for k, v in payload.items() 
                      if k not in ["text", "doc_id"]}
            
            # Format the result
            formatted_results.append({
                "id": result.id,
                "doc_id": doc_id,
                "text": text,
                "score": result.score,
                "metadata": metadata
            })
        
        return formatted_results
    
    def search_by_metadata(
        self, 
        field: str, 
        value: Any, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for documents by metadata field value.
        
        Args:
            field: Metadata field name
            value: Value to match
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        # Create a filter condition
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            ]
        )
        
        # Retrieve points using scroll API
        results, next_page_offset = self.client.scroll(
            collection_name=self.collection_name,
            filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        formatted_results = []
        for point in results:
            # Extract text and document ID from payload
            payload = point.payload or {}
            text = payload.get("text", "")
            doc_id = payload.get("doc_id", str(point.id))
            
            # Create a clean metadata dictionary without text and doc_id
            metadata = {k: v for k, v in payload.items() 
                      if k not in ["text", "doc_id"]}
            
            # Format the result
            formatted_results.append({
                "id": point.id,
                "doc_id": doc_id,
                "text": text,
                "metadata": metadata
            })
        
        return formatted_results
    
    def delete_document(self, doc_id: str):
        """Delete a document from the collection."""
        # Try to find if this document was split into chunks
        results = self.client.scroll(
            collection_name=self.collection_name,
            filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=1000  # Assuming no more than 1000 chunks per document
        )[0]
        
        if results:
            # Get IDs of all chunks
            point_ids = [point.id for point in results]
            
            # Delete all chunks
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            print(f"Document '{doc_id}' with {len(point_ids)} chunks deleted from collection.")
        else:
            # Try direct ID deletion as fallback
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[doc_id]
                )
            )
            print(f"Document '{doc_id}' deleted from collection.")
    
    def clear_collection(self):
        """Delete all documents from the collection."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter()
        )
        print(f"All documents deleted from collection '{self.collection_name}'.")


if __name__ == "__main__":
    import argparse
    import json
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Search for documents in Qdrant vector database')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--collection', default='documents', help='Qdrant collection name')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of results')
    parser.add_argument('--mode', default='http', choices=['memory', 'local', 'http'], 
                        help='Qdrant connection mode')
    parser.add_argument('--location', default=None, 
                        help='Path for local storage or URL for HTTP connection')
    parser.add_argument('--filter', default=None, dest='filter_json',
                        help='JSON string with filter conditions')
    
    args = parser.parse_args()
    
    # Initialize the search system with command-line parameters
    search_system = VectorSearchSystem(
        collection_name=args.collection,  # Use the collection name from arguments
        connection_mode=args.mode,
        location=args.location
    )
    
    # Parse filter if provided
    filter_condition = None
    if args.filter_json:
        try:
            filter_condition = json.loads(args.filter_json)
        except json.JSONDecodeError:
            print("Warning: Invalid filter JSON. Proceeding without filter.")
    
    # Perform search
    print(f"Searching for '{args.query}' in collection '{args.collection}'...")
    results = search_system.search(
        args.query, 
        limit=args.limit,
        filter_condition=filter_condition
    )
    
    # Display results
    if not results:
        print("No matching documents found.")
    else:
        print("\nSearch Results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['doc_id']} (Score: {result['score']:.4f})")
            
            # Print a snippet of the text (first 200 characters)
            text = result['text']
            snippet = text[:200] + "..." if len(text) > 200 else text
            print(f"   {snippet}")
            
            # Print metadata
            if result['metadata']:
                print(f"   Metadata: {result['metadata']}")
            
            print()