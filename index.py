import os
import argparse
import json
import hashlib
from search import VectorSearchSystem
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct

def index_documents(folder_path, collection_name, connection_mode="http", location=None, chunk_size=0, overwrite=False):
    """
    Index all text documents in a folder to a Qdrant collection.
    
    Args:
        folder_path: Path to the folder containing documents
        collection_name: Name of the Qdrant collection
        connection_mode: One of "memory", "local", or "http"
        location: Path for local storage or URL for HTTP connection
        chunk_size: If > 0, split documents into chunks of this size
        overwrite: If True, overwrite existing documents with the same ID
    """
    # Initialize the vector search system
    search_system = VectorSearchSystem(
        collection_name=collection_name,
        connection_mode=connection_mode,
        location=location
    )
    
    # Process each file in the folder
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        # Define supported extensions
        supported_extensions = ['.txt', '.md', '.csv', '.json', '.html', '.py', '.js']
        if not any(filename.lower().endswith(ext) for ext in supported_extensions):
            print(f"Skipping unsupported file: {filename}")
            continue
        
        try:
            # Create a stable ID based on the filename
            file_id = hashlib.md5(filename.encode()).hexdigest()
            
            # Check if document with this ID already exists
            if not overwrite:
                point = search_system.client.retrieve(
                    collection_name=collection_name,
                    ids=[file_id]
                )
                
                if point:  # If the document exists
                    print(f"Skipping existing file: {filename}")
                    skipped_count += 1
                    continue
            
            # Extract file metadata
            file_stats = os.stat(file_path)
            metadata = {
                "filename": filename,
                "path": file_path,
                "size_bytes": file_stats.st_size,
                "last_modified": file_stats.st_mtime,
                "extension": os.path.splitext(filename)[1]
            }
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Generate embedding
            embedding = search_system._get_embedding(content)
            
            # Upload to Qdrant (upsert will overwrite if ID already exists)
            search_system.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=file_id,
                        vector=embedding,
                        payload={
                            "doc_id": filename,
                            "text": content,
                            **metadata
                        }
                    )
                ]
            )
            print(f"Indexed document: {filename}")
            success_count += 1
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            failed_count += 1
    
    print(f"\nIndexing complete.")
    print(f"Successfully indexed: {success_count} documents")
    print(f"Skipped existing: {skipped_count} documents")
    if failed_count > 0:
        print(f"Failed to index: {failed_count} documents")

def index_single_document(file_path, collection_name, connection_mode="http", location=None, chunk_size=0, metadata=None, overwrite=False):
    """
    Index a single document to a Qdrant collection.
    
    Args:
        file_path: Path to the document file
        collection_name: Name of the Qdrant collection
        connection_mode: One of "memory", "local", or "http"
        location: Path for local storage or URL for HTTP connection
        chunk_size: If > 0, split document into chunks of this size
        metadata: Additional metadata to store with the document
        overwrite: If True, overwrite existing document with the same ID
    """
    if not os.path.isfile(file_path):
        print(f"Error: {file_path} is not a valid file")
        return
        
    # Initialize the vector search system
    search_system = VectorSearchSystem(
        collection_name=collection_name,
        connection_mode=connection_mode,
        location=location
    )
    
    filename = os.path.basename(file_path)
    
    try:
        # Create a stable ID based on the filename
        file_id = hashlib.md5(filename.encode()).hexdigest()
        
        # Check if document with this ID already exists
        if not overwrite:
            point = search_system.client.retrieve(
                collection_name=collection_name,
                ids=[file_id]
            )
            
            if point:  # If the document exists
                print(f"Document '{filename}' already exists in collection. Skipping.")
                return
        
        # Extract file metadata
        file_stats = os.stat(file_path)
        file_metadata = {
            "filename": filename,
            "path": file_path,
            "size_bytes": file_stats.st_size,
            "last_modified": file_stats.st_mtime,
            "extension": os.path.splitext(file_path)[1]
        }
        
        # Merge with additional metadata if provided
        if metadata:
            file_metadata.update(metadata)
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Generate embedding
        embedding = search_system._get_embedding(content)
        
        # Upload to Qdrant (upsert will overwrite if ID already exists)
        search_system.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=file_id,
                    vector=embedding,
                    payload={
                        "doc_id": filename,
                        "text": content,
                        **file_metadata
                    }
                )
            ]
        )
        print(f"Successfully indexed {filename}")
        
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index documents to Qdrant vector database')
    parser.add_argument('path', help='Path to folder containing documents or a single document file')
    parser.add_argument('--collection', default='documents', help='Qdrant collection name')
    parser.add_argument('--mode', default='http', choices=['memory', 'local', 'http'], 
                        help='Qdrant connection mode')
    parser.add_argument('--location', default=None, 
                        help='Path for local storage or URL for HTTP connection')
    parser.add_argument('--chunk-size', type=int, default=0,
                        help='If > 0, split documents into chunks of this size')
    parser.add_argument('--metadata', type=str, default=None,
                        help='JSON string with additional metadata to add to all documents')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite documents that already exist in the collection')
    
    args = parser.parse_args()
    
    # Parse metadata if provided
    custom_metadata = None
    if args.metadata:
        try:
            custom_metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Warning: Invalid metadata JSON. Proceeding without custom metadata.")
    
    # Check if path is a directory or a file
    if os.path.isdir(args.path):
        print(f"Indexing documents from folder: {args.path}")
        index_documents(
            args.path, 
            args.collection,
            connection_mode=args.mode,
            location=args.location,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite
        )
    elif os.path.isfile(args.path):
        print(f"Indexing single document: {args.path}")
        index_single_document(
            args.path,
            args.collection,
            connection_mode=args.mode,
            location=args.location,
            chunk_size=args.chunk_size,
            metadata=custom_metadata,
            overwrite=args.overwrite
        )
    else:
        print(f"Error: {args.path} is neither a valid directory nor a valid file")