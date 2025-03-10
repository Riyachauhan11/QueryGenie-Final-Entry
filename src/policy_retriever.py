import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model 
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="../data/chroma_db")
policy_collection = chroma_client.get_collection("company_policies")

def retrieve_policy(query):
    """Retrieves the most relevant policy chunk, filtering out weak matches dynamically."""
    query_embedding = embedding_model.encode(query).tolist()

    results = policy_collection.query(
        query_embeddings=[query_embedding],
        n_results=3,  # Retrieve top 3 matches
        include=["documents", "distances"]
    )

    if results["documents"]:
        best_scores = [1 - d[0] for d in results["distances"]]  # Normalize distances
        avg_score = sum(best_scores) / len(best_scores)  # Compute dynamic threshold

        best_match = results["documents"][0][0]  # Always return the top match
        best_score = best_scores[0]  # Best score among results

        dynamic_threshold = avg_score * 0.8  

        if best_score >= dynamic_threshold:
            return best_match  

    return "NO_MATCH"  # No relevant policy found

if __name__ == "__main__":
    query = "how to contact company?"
    print(retrieve_policy(query))
