from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning is a type of machine learning.",
    "Natural language processing is used in AI applications.",
]

# User query
query = "Tell me about machine learning."


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([query] + documents)

# Calculate cosine similarity between the query and documents
cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Sort documents by similarity score
most_similar_document = documents[cosine_similarities.argmax()]

# Print the most relevant document
print("Most Relevant Document:", most_similar_document)