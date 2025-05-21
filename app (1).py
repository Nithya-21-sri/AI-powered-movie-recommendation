import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Title
st.title("Recommendation System: Content-Based, Collaborative, and Hybrid")

# Upload
uploaded_file = st.file_uploader("Upload NM DATASET.xlsx", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Preview data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Combine content features
    df['combined_features'] = df['Category'].astype(str) + ' ' + df['Item_ID'].astype(str)

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Index map
    indices = pd.Series(df.index, index=df['Item_ID']).drop_duplicates()

    # Content-based function
    def content_based_recommend(item_id, num_recommendations=10):
        if item_id not in indices:
            return pd.DataFrame({'Error': [f"Item_ID '{item_id}' not found."]})
        idx = indices[item_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        item_indices = [i[0] for i in sim_scores]
        return df[['Item_ID', 'Category']].iloc[item_indices]

    # Collaborative filtering
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['User_ID', 'Item_ID', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    rmse_val = rmse(predictions)

    # Predict function
    def predict_rating(user_id, item_id):
        return model.predict(user_id, item_id).est

    # Hybrid recommend
    def hybrid_recommend(user_id, item_id, top_n=10, weight_cb=0.5, weight_cf=0.5):
        if item_id not in indices:
            return pd.DataFrame({'Error': [f"Item_ID '{item_id}' not found."]})

        idx = indices[item_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n*2+1]

        hybrid_scores = []
        for i, score in sim_scores:
            candidate_id = df['Item_ID'].iloc[i]
            cb_score = score
            cf_score = predict_rating(user_id, candidate_id)
            final_score = (weight_cb * cb_score) + (weight_cf * (cf_score / 5))
            hybrid_scores.append((candidate_id, final_score))

        top_recommendations = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(top_recommendations, columns=['Recommended Item_ID', 'Score'])

    # User inputs
    st.sidebar.subheader("Enter Recommendation Parameters")
    user_id = st.sidebar.text_input("User_ID", value="User_913")
    item_id = st.sidebar.text_input("Item_ID", value="Item_52")
    num_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

    if st.sidebar.button("Generate Recommendations"):
        st.subheader("📌 Content-Based Recommendations")
        st.dataframe(content_based_recommend(item_id, num_recommendations))

        st.subheader("📌 Collaborative Filtering Prediction")
        rating = predict_rating(user_id, item_id)
        st.write(f"Predicted Rating by {user_id} for {item_id}: **{rating:.3f}**")

        st.subheader("📌 Hybrid Recommendations")
        st.dataframe(hybrid_recommend(user_id, item_id, num_recommendations))

        st.success(f"Model RMSE on test set: {rmse_val:.4f}")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Title
st.title("Recommendation System: Content-Based, Collaborative, and Hybrid")

# Upload
uploaded_file = st.file_uploader("Upload NM DATASET.xlsx", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Preview data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Combine content features
    df['combined_features'] = df['Category'].astype(str) + ' ' + df['Item_ID'].astype(str)

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Index map
    indices = pd.Series(df.index, index=df['Item_ID']).drop_duplicates()

    # Content-based function
    def content_based_recommend(item_id, num_recommendations=10):
        if item_id not in indices:
            return pd.DataFrame({'Error': [f"Item_ID '{item_id}' not found."]})
        idx = indices[item_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        item_indices = [i[0] for i in sim_scores]
        return df[['Item_ID', 'Category']].iloc[item_indices]

    # Collaborative filtering
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['User_ID', 'Item_ID', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    rmse_val = rmse(predictions)

    # Predict function
    def predict_rating(user_id, item_id):
        return model.predict(user_id, item_id).est

    # Hybrid recommend
    def hybrid_recommend(user_id, item_id, top_n=10, weight_cb=0.5, weight_cf=0.5):
        if item_id not in indices:
            return pd.DataFrame({'Error': [f"Item_ID '{item_id}' not found."]})

        idx = indices[item_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n*2+1]

        hybrid_scores = []
        for i, score in sim_scores:
            candidate_id = df['Item_ID'].iloc[i]
            cb_score = score
            cf_score = predict_rating(user_id, candidate_id)
            final_score = (weight_cb * cb_score) + (weight_cf * (cf_score / 5))
            hybrid_scores.append((candidate_id, final_score))

        top_recommendations = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(top_recommendations, columns=['Recommended Item_ID', 'Score'])

    # User inputs
    st.sidebar.subheader("Enter Recommendation Parameters")
    user_id = st.sidebar.text_input("User_ID", value="User_913")
    item_id = st.sidebar.text_input("Item_ID", value="Item_52")
    num_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

    if st.sidebar.button("Generate Recommendations"):
        st.subheader("📌 Content-Based Recommendations")
        st.dataframe(content_based_recommend(item_id, num_recommendations))

        st.subheader("📌 Collaborative Filtering Prediction")
        rating = predict_rating(user_id, item_id)
        st.write(f"Predicted Rating by {user_id} for {item_id}: **{rating:.3f}**")

        st.subheader("📌 Hybrid Recommendations")
        st.dataframe(hybrid_recommend(user_id, item_id, num_recommendations))

        st.success(f"Model RMSE on test set: {rmse_val:.4f}")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Title
st.title("Recommendation System: Content-Based, Collaborative, and Hybrid")

# Upload
uploaded_file = st.file_uploader("Upload NM DATASET.xlsx", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Preview data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Combine content features
    df['combined_features'] = df['Category'].astype(str) + ' ' + df['Item_ID'].astype(str)

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Index map
    indices = pd.Series(df.index, index=df['Item_ID']).drop_duplicates()

    # Content-based function
    def content_based_recommend(item_id, num_recommendations=10):
        if item_id not in indices:
            return pd.DataFrame({'Error': [f"Item_ID '{item_id}' not found."]})
        idx = indices[item_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        item_indices = [i[0] for i in sim_scores]
        return df[['Item_ID', 'Category']].iloc[item_indices]

    # Collaborative filtering
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['User_ID', 'Item_ID', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    rmse_val = rmse(predictions)

    # Predict function
    def predict_rating(user_id, item_id):
        return model.predict(user_id, item_id).est

    # Hybrid recommend
    def hybrid_recommend(user_id, item_id, top_n=10, weight_cb=0.5, weight_cf=0.5):
        if item_id not in indices:
            return pd.DataFrame({'Error': [f"Item_ID '{item_id}' not found."]})

        idx = indices[item_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n*2+1]

        hybrid_scores = []
        for i, score in sim_scores:
            candidate_id = df['Item_ID'].iloc[i]
            cb_score = score
            cf_score = predict_rating(user_id, candidate_id)
            final_score = (weight_cb * cb_score) + (weight_cf * (cf_score / 5))
            hybrid_scores.append((candidate_id, final_score))

        top_recommendations = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(top_recommendations, columns=['Recommended Item_ID', 'Score'])

    # User inputs
    st.sidebar.subheader("Enter Recommendation Parameters")
    user_id = st.sidebar.text_input("User_ID", value="User_913")
    item_id = st.sidebar.text_input("Item_ID", value="Item_52")
    num_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

    if st.sidebar.button("Generate Recommendations"):
        st.subheader("📌 Content-Based Recommendations")
        st.dataframe(content_based_recommend(item_id, num_recommendations))

        st.subheader("📌 Collaborative Filtering Prediction")
        rating = predict_rating(user_id, item_id)
        st.write(f"Predicted Rating by {user_id} for {item_id}: **{rating:.3f}**")

        st.subheader("📌 Hybrid Recommendations")
        st.dataframe(hybrid_recommend(user_id, item_id, num_recommendations))

        st.success(f"Model RMSE on test set: {rmse_val:.4f}")