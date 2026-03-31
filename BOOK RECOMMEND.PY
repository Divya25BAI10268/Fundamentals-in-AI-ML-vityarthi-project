import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset (you can modify or expand)
data = {
    "title": [
        "Harry Potter", "Percy Jackson", "Lord of the Rings",
        "The Fault in Our Stars", "The alchemist", "Pride and prejudice"
    ],
    "genre": [
        "fantasy magic adventure",
        "fantasy mythology adventure",
        "fantasy epic adventure",
        "fantasy adventure",
        "romance fantasy",
        "romance drama"
    ]
}

df = pd.DataFrame(data)

# Convert text to vectors
cv = CountVectorizer()
matrix = cv.fit_transform(df["genre"])

# Compute similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(title):
    if title not in df["title"].values:
        return "Item not found!"

    index = df[df["title"] == title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nBecause you liked '{title}', you may also like:\n")
    
    for i in scores[1:4]:  # top 3 recommendations
        print(df.iloc[i[0]]["title"])

# User input
user_input = input("Enter a movie/book name: ")
recommend(user_input)
