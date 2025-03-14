# list_documents.py
from qdrant_client import QdrantClient

client = QdrantClient(path="./local_vector_db")
collection_name = "test_collection"

# Get all points
results, next_page = client.scroll(
    collection_name=collection_name,
    limit=100,
    with_payload=True
)

print(f"Found {len(results)} documents in collection:")
for i, point in enumerate(results):
    print(f"\nDocument {i+1}:")
    print(f"ID: {point.id}")
    doc_id = point.payload.get("doc_id", "N/A")
    print(f"Doc ID: {doc_id}")
    text = point.payload.get("text", "N/A")
    print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
    
    # Print other metadata
    other_metadata = {k: v for k, v in point.payload.items() if k not in ["text", "doc_id"]}
    if other_metadata:
        print("Metadata:")
        for key, value in other_metadata.items():
            print(f"  {key}: {value}")