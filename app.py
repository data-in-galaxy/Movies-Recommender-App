import streamlit as st
st.set_page_config(page_title="Netflix Shows", page_icon="🍿", layout="wide")    

import pandas as pd
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load models and dataset
netflix_metadata = joblib.load('models/netflix_metadata.df')
tfidf_matrix = joblib.load('models/tfidf_mat.tf')
tfidf = joblib.load('models/vectorizer.tf')
tfidf_meta_matrix = joblib.load('models/tfidf_meta_mat.tf')
tfidf_meta = joblib.load('models/vectorizer_meta.tf')

# define functions
def get_keywords_recommendations(keywords):
    
    keywords = keywords.lower()
    if "," in keywords:
        keywords = keywords.split(",")
        keywords = " ".join(["".join(k.strip().split()) for k in keywords])
    else: 
        keywords = keywords.replace(" ", "")
   
    # If there are no spaces in keywords, use regex to find exact matches
    regex_pattern = r'\b' + re.escape(keywords) + r'\b'
    matches = netflix_metadata[netflix_metadata['soup'].str.contains(regex_pattern, case=False, regex=True)]
    
    # If there are exact matches, return them
    if not matches.empty:
        recomm = matches['title'][:3].tolist()
    else:
        # If no exact matches, calculate cosine similarity
        keywords_vector = tfidf_meta.transform([keywords]) # vectorize keywords
        result = cosine_similarity(keywords_vector, tfidf_meta_matrix) # compute cosine similarity
        similar_key_movies = sorted(list(enumerate(result[0])), reverse=True, key=lambda x: x[1]) # sort top n similar movies
        recomm = [netflix_metadata.iloc[i[0]].title for i in similar_key_movies[1:4]]
    
    return recomm
    
    # extract names from dataframe and return movie names
    recomm = [netflix_metadata.iloc[i[0]].title for i in similar_key_movies[1:4]]
    return recomm


from sklearn.metrics.pairwise import linear_kernel
cosine_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title):
    # get index from dataframe:
    index = netflix_metadata[netflix_metadata['title'] == title].index[0]
    
    # sort top n similar movies
    similar_titles = sorted(list(enumerate(cosine_matrix[index])), reverse=True, key=lambda x: x[1])
    
    # extract names from dataframe and return movie names
    recomm = []
    for i in  similar_titles[1:6]:
        recomm.append(netflix_metadata.iloc[i[0]].title)
        
    return recomm


# App Layout
def main():
    
    st.image("images/netflix.jpg")
    st.title("Movie Finder 🍿 🤖")
    movies = []

    with st.sidebar:
        st.header("Get Recommendations by 👇")
        search_type = st.radio("", ('Movie Title', 'Keywords'))

# call functions based on selectbox
    if search_type == 'Movie Title': 
        st.subheader("Select Movie 🎬")   
        movie_name = st.selectbox('', netflix_metadata.title)
        if st.button('Recommend 🚀'):
            with st.spinner('Wait for it...'):
                movies = get_recommendations(movie_name)
                        
    else:
        st.subheader('Enter Cast / Crew / Tags / Genre  🌟')
        keyword = st.text_input('', 'Daniel Craig')
        if st.button('Recommend 🚀'):
            with st.spinner('Wait for it...'):
                movies = get_keywords_recommendations(keyword)
                
# Display the recommended movies
    st.subheader("Recommended Movies 🎥")
    if movies:
        for i, movie in enumerate(movies):
            st.write(f"{i + 1}. {movie}")
    

if __name__ == "__main__":
    main()
    
