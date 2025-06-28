# mall_customer_k_means_clustering
# 🧠 Mall Customer Segmentation using K-Means Clustering

This project demonstrates how to use **K-Means Clustering** for segmenting customers based on **Annual Income** and **Spending Score** from a mall customer dataset.
It uses **unsupervised learning** to group similar customer behaviors and is presented through an interactive **Streamlit app**.

---

## 📂 Dataset: Mall_Customers.csv

**Source**: [Kaggle - Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)

### 📊 Features:
- `CustomerID`: Unique ID for each customer
- `Gender`: Male or Female
- `Age`: Age of the customer
- `Annual Income (k$)`: Annual income in thousands
- `Spending Score (1-100)`: Score assigned by the mall based on customer behavior and spending nature

### 🔍 Goal:
To identify different segments of customers for better business targeting and decision-making.

---

## 🔬 Unsupervised Learning Technique: K-Means Clustering

K-Means is used to find **natural groupings** in data without pre-labeled outcomes.

### Steps:
1. Load and clean the data
2. Visualize the features
3. Apply Elbow Method to find optimal `k`
4. Train KMeans with `k=5`
5. Visualize clusters and centroids
6. Build a Streamlit app to predict clusters for new inputs

---

## 💡 How to Run the App

1. Clone the repository:
```bash
git clone https://github.com/your-username/mall-customer-clustering.git
cd mall-customer-clustering
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

---

## 📜 File Structure
```bash
mall-customer-clustering/
├── app.py                  # Streamlit web app
├── Mall_Customers.csv      # Dataset
├── README.md
└── requirements.txt
```

---

## 🧾 Code: app.py (Main App Script)
```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st

# Load data
data = pd.read_csv("Mall_Customers.csv")
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X)

# Streamlit UI
st.title("Mall Customer Segmentation App")
st.write("This app segments mall customers using K-Means clustering.")

# User input
income = st.slider("Annual Income (k$)", 15, 137, 70)
score = st.slider("Spending Score (1-100)", 1, 100, 50)

# Predict cluster
input_data = np.array([[income, score]])
cluster = kmeans.predict(input_data)[0]
st.success(f"The predicted cluster for this customer is: Cluster {cluster}")

# Show centroids
centroids = kmeans.cluster_centers_
st.subheader("Cluster Centroids")
st.write(pd.DataFrame(centroids, columns=['Annual Income (k$)', 'Spending Score (1-100)']))
```

---

## ✅ Project by Skillcreaft Technology
> Designed for learning and real-world application of unsupervised ML using KMeans and Streamlit.

---

## 📃 License
MIT License
