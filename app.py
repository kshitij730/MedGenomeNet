import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
model = load_model("mutation_prediction_model.h5")

# Streamlit UI
st.title("🧬 Gene Mutation Prediction Dashboard")
st.write("Upload clinical/genomic data to predict mutation risk categories.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    df = pd.read_csv(uploaded_file)

    # Display uploaded data
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Feature Selection
    features = df[['TMB (nonsynonymous)', 'Fraction Genome Altered', 'Tumor Purity', 
                   'Cancer Type', 'Sex', 'Smoking History', 'Primary Tumor Site', 'Specimen Type']]

    # Handle missing values
    features.fillna(features.median(), inplace=True)
    features.fillna(features.mode().iloc[0], inplace=True)

    # One-hot encoding for categorical features
    features = pd.get_dummies(features, columns=['Cancer Type', 'Sex', 'Smoking History', 
                                                 'Primary Tumor Site', 'Specimen Type'])

    # Normalize numerical features
    scaler = StandardScaler()
    features[['TMB (nonsynonymous)', 'Fraction Genome Altered', 'Tumor Purity']] = scaler.fit_transform(
        features[['TMB (nonsynonymous)', 'Fraction Genome Altered', 'Tumor Purity']]
    )

    # Make predictions
    predictions = model.predict(features.values)
    pred_labels = np.argmax(predictions, axis=1)

    # Map predictions to class labels
    mutation_classes = ['Low', 'Medium', 'High']
    pred_labels = [mutation_classes[i] for i in pred_labels]

    # Add predictions to dataframe
    df['Predicted Mutation Type'] = pred_labels

    # Display results
    st.subheader("Predicted Mutation Types")
    st.write(df[['Cancer Type', 'Predicted Mutation Type']])

    # Visualization: Mutation distribution per cancer type
    st.subheader("📊 Mutation Type Distribution by Cancer Type")

    mutation_counts = df.groupby(['Cancer Type', 'Predicted Mutation Type']).size().unstack()

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    mutation_counts.plot(kind='bar', stacked=False, colormap="viridis", ax=ax)
    plt.title("Mutation Type Distribution per Cancer Type")
    plt.xlabel("Cancer Type")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.legend(title="Mutation Type")
    st.pyplot(fig)

    # Pie Chart for a specific cancer type
    st.subheader("🍩 Mutation Distribution for Selected Cancer Type")
    cancer_types = df['Cancer Type'].unique()
    selected_cancer = st.selectbox("Select a Cancer Type", cancer_types)

    if selected_cancer:
        cancer_data = df[df["Cancer Type"] == selected_cancer]["Predicted Mutation Type"].value_counts()

        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(cancer_data, labels=cancer_data.index, autopct="%1.1f%%", colors=["skyblue", "orange", "red"])
        plt.title(f"Mutation Distribution for {selected_cancer}")
        st.pyplot(fig_pie)
