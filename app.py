import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Hierarchical Clustering", layout="centered")
st.title("🌳 Hierarchical Clustering (Agglomerative)")

# -------------------------------
# Upload dataset
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (numeric features only)", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    # -------------------------------
    # Feature selection
    # -------------------------------
    st.subheader("⚙️ Feature Selection")
    features = st.multiselect(
        "Select features for clustering",
        df.columns.tolist(),
        default=df.columns.tolist()
    )

    if len(features) >= 2:
        X = df[features]

        # -------------------------------
        # Scaling
        # -------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -------------------------------
        # Dendrogram
        # -------------------------------
        st.subheader("🌳 Dendrogram")
        linkage_method = st.selectbox(
            "Select linkage method",
            ["ward", "complete", "average", "single"]
        )

        Z = linkage(X_scaled, method=linkage_method)

        fig, ax = plt.subplots()
        dendrogram(Z, ax=ax)
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        st.pyplot(fig)

        # -------------------------------
        # Clustering
        # -------------------------------
        st.subheader("🔢 Agglomerative Clustering")
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        clusters = model.fit_predict(X_scaled)

        df["Cluster"] = clusters

        st.subheader("📌 Clustered Data")
        st.dataframe(df.head())

        st.success("Hierarchical clustering completed successfully ✅")

    else:
        st.warning("Please select at least two features.")

else:
    st.info("Upload a CSV file to begin clustering.")
