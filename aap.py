# ----------------------------------------------
# K-Means Clustering Streamlit Web App
# ----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ----------------------------------------------
# Page Config
# ----------------------------------------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("Mall Customers Segmentation using K-Means Clustering")
st.markdown("This interactive app clusters mall customers based on **Annual Income** and **Spending Score**.")

# ----------------------------------------------
# Load the Dataset
# ----------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

dataset = load_data()

# Clean Gender
if 'Gender' in dataset.columns:
    dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})

# ----------------------------------------------
# Sidebar - User Input
# ----------------------------------------------
st.sidebar.header("Predict Cluster for a New Customer")
income_input = st.sidebar.slider("Annual Income (k$):", 0, 150, 50)
score_input = st.sidebar.slider("Spending Score (1-100):", 0, 100, 50)

# ----------------------------------------------
# Feature Selection
# ----------------------------------------------
X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']]

# ----------------------------------------------
# KMeans + Elbow Method
# ----------------------------------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Apply KMeans with Optimal k = 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
dataset['Cluster'] = y_kmeans
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
centroids['Cluster'] = centroids.index

# ----------------------------------------------
# Tabs for Navigation
# ----------------------------------------------
tab1, tab2, tab3 = st.tabs(["Visualizations", "Elbow Method", "Predict Cluster"])

with tab1:
    st.subheader("Customer Segments Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']

    for i in range(5):
        ax.scatter(
            X[y_kmeans == i]['Annual Income (k$)'],
            X[y_kmeans == i]['Spending Score (1-100)'],
            s=100,
            c=colors[i],
            label=f'Cluster {i}'
        )

    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=300,
        c='yellow',
        label='Centroids',
        marker='X'
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("K-Means Customer Segments")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Statistical Summary per Cluster")
    cluster_summary = dataset.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].agg(
        ['mean', 'median', 'min', 'max', 'std']
    ).round(2)
    cluster_summary.columns = ['_'.join(col) for col in cluster_summary.columns]
    st.dataframe(cluster_summary)

with tab2:
    st.subheader("Elbow Method for Optimal k")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal')
    ax2.set_title("Elbow Method")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("WCSS")
    ax2.axvline(x=5, color='red', linestyle=':', label='Optimal k=5')
    ax2.legend()
    st.pyplot(fig2)

with tab3:
    st.subheader("Enter Customer Info")

    new_data = np.array([[income_input, score_input]])
    predicted_cluster = kmeans.predict(new_data)[0]

    st.success(f"Predicted Cluster: **{predicted_cluster}**")

    st.subheader("Cluster Centroid Info")
    st.write(centroids.loc[predicted_cluster])

# ----------------------------------------------
# Footer
# ----------------------------------------------
st.markdown("---")
st.markdown("**Project Summary**: This app uses K-Means to segment customers by spending behavior & income. Great for targeted marketing!")
st.caption("Built with  by Khushi Saini | Powered by Streamlit ")
