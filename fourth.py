from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents (your knowledge base)
documents = [
    "I don't know",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning is a type of machine learning.",
    "Deep learning is used in most modern AI applications.",
    "Artificial intelligence is the science of programming smart machines.",
    "Natural language processing is used in AI applications.",
]

# User query
user_query = "What is deep learning?"

# Create TF-IDF vectors for documents and the query
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([user_query] + documents)

# Calculate cosine similarity between the user query and documents
cosine_similarities = cosine_similarity(
    tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Sort documents by similarity score
most_similar_document_index = cosine_similarities.argmax()
most_similar_document = documents[most_similar_document_index]

# Print the most relevant document as the answer
print("User Query:", user_query)
print("Most Relevant Document:", most_similar_document)
