from search import VectorSearchSystem

# Initialize the system in local mode
search_system = VectorSearchSystem(
    collection_name="test_collection",
    connection_mode="local",
    location="./local_vector_db"  # This directory will be created automatically
)

# Add some test documents
print("Adding test documents...")
search_system.add_document(
    "doc1", 
    "Qdrant is a vector database designed for storing and searching vector embeddings.",
    {"source": "test", "category": "database"}
)

search_system.add_document(
    "doc2", 
    "Vector embeddings are numerical representations that capture semantic meaning.",
    {"source": "test", "category": "concept"}
)

search_system.add_document(
    "doc3", 
    "Python is a programming language widely used for machine learning and data analysis.",
    {"source": "test", "category": "language"}
)

# Perform a search
print("\nSearching for 'vector database'...")
results = search_system.search("vector database", limit=2)

# Display results
print("\nSearch results:")
for i, result in enumerate(results):
    print(f"{i+1}. {result['doc_id']} (Score: {result['score']:.4f})")
    print(f"   {result['text']}")
    print(f"   Metadata: {result['metadata']}")

print("\nTest complete! Vector data has been stored in './local_vector_db'")