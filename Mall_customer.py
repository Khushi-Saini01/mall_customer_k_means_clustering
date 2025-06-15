# ---------------------------------------------------
#  K-Means Clustering on Mall Customers Dataset 
# ---------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')  # Clean output

# -----------------------------
# Load the Dataset
# -----------------------------
dataset = pd.read_csv("Mall_Customers.csv")
print(" First 5 Rows:\n", dataset.head())

print("\n Dataset Info:")
print(dataset.info())

print("\n Null Values:\n", dataset.isnull().sum())

print("\n Summary Stats:\n", dataset.describe())

# -----------------------------
#  Clean the Dataset (if needed)
# -----------------------------
# Convert Gender to numerical for optional use
if 'Gender' in dataset.columns:
    dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})

# -----------------------------
# Basic Visualization
# -----------------------------
sns.pairplot(data=dataset[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.suptitle("🔍 Pairplot of Features", y=1.02)
plt.show()

# -----------------------------
# Feature Selection
# We'll use 'Annual Income' and 'Spending Score'
# -----------------------------
X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------------
# Elbow Method to Determine Optimal k
# -----------------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.axvline(x=5, color='red', linestyle=':', label='Optimal k=5')
plt.legend()
plt.show()

# -----------------------------
# Apply KMeans with Optimal k = 5
# -----------------------------
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels
dataset['Cluster'] = y_kmeans

# -----------------------------
# Visualize the Clusters
# -----------------------------
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(5):
    plt.scatter(
        X[y_kmeans == i]['Annual Income (k$)'],
        X[y_kmeans == i]['Spending Score (1-100)'],
        s=100,
        c=colors[i],
        label=f'Cluster {i}'
    )

# Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='yellow',
    label='Centroids',
    marker='X'
)

plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
#  Pairplot with Cluster Labels
# -----------------------------
sns.pairplot(data=dataset[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']], hue="Cluster", palette="husl")
plt.suptitle("Pairplot Colored by Cluster", y=1.02)
plt.show()

# -----------------------------
# Compare with Original Feature: Gender
# -----------------------------
original_df = pd.read_csv("Mall_Customers.csv")

if 'Gender' in original_df.columns:
    sns.pairplot(data=original_df[['Gender', 'Annual Income (k$)', 'Spending Score (1-100)']], hue="Gender", palette='Set1')
    plt.suptitle(" Pairplot Colored by Gender", y=1.02)
    plt.show()

# -----------------------------
#  Spending Category Bucketing
# -----------------------------
def categorize_spending(score):
    if score < 40:
        return 'Low'
    elif score < 70:
        return 'Medium'
    else:
        return 'High'

original_df['Spending Category'] = original_df['Spending Score (1-100)'].apply(categorize_spending)

sns.pairplot(data=original_df[['Annual Income (k$)', 'Spending Score (1-100)', 'Spending Category']],
             hue="Spending Category", palette='Set2')
plt.suptitle(" Pairplot by Spending Category", y=1.02)
plt.show()

# -----------------------------
# Cluster Summary (Optional Insight)
# -----------------------------
print("\nCluster Centroids:")
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
print(centroids)

# Merge centroids with Cluster Index
centroids['Cluster'] = centroids.index
print("\nInterpretation Hint: Centroid Overview")
try:
    from IPython.display import display
    display(centroids.sort_values(by='Spending Score (1-100)', ascending=False))
except:
    print(centroids.sort_values(by='Spending Score (1-100)', ascending=False))

# -----------------------------
# Statistical Summary per Cluster
# -----------------------------
print("\n Detailed Statistical Summary by Cluster:\n")

# Group by Cluster and describe
cluster_summary = dataset.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].agg(
    ['mean', 'median', 'min', 'max', 'std']
).round(2)

# Flatten multi-level column names
cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns]

print(cluster_summary)

try:
    display(cluster_summary)
except:
    pass

# -----------------------------
# Summary
# -----------------------------
print("""
 Project Summary:
- Used K-Means to segment customers based on income & spending behavior.
- Determined optimal k = 5 using Elbow Method.
- Visualized clusters and compared with Gender & Spending Category.
- Generated statistical insights and centroid interpretation.
- Great for marketing segmentation and personalized business strategies. 
""")
# -----------------------------
# Predict Cluster for New Customer Input
# -----------------------------
print("\n Predict Cluster for a New Customer:")

try:
    income = float(input("Enter Annual Income (k$): "))
    score = float(input("Enter Spending Score (1-100): "))

    new_data = np.array([[income, score]])
    predicted_cluster = kmeans.predict(new_data)

    print(f"\n The customer belongs to Cluster: {predicted_cluster[0]}")

    # Optional: show which type of cluster it is
    cluster_info = centroids.loc[predicted_cluster[0]]
    print("\n Cluster Info:")
    print(cluster_info)

except ValueError:
    print("Invalid input. Please enter numeric values.")
