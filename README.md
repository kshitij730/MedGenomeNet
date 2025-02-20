# GeneMutAI: Deep Learning for Gene Mutation Prediction

## 📌 Overview

GeneMutAI is a deep learning-powered tool designed to predict the impact of gene mutations on diseases, particularly in cancer research. By leveraging genomic and clinical data, the model provides insights into mutation types and their potential health implications.

## 🚀 Features

- **Mutation Prediction**: Predicts whether a mutation is Low, Medium, or High risk.
- **Deep Learning Model**: Uses a trained neural network to analyze genomic data.
- **Data Visualization**: Displays mutation distributions across different cancer types.
- **Streamlit Dashboard**: Interactive web application for data analysis.
- **CSV Support**: Upload and process clinical/genomic data.

## 📂 Dataset

- **Name**: MSK-IMPACT Clinical Sequencing Dataset
- **Source**: [CBioPortal](https://www.cbioportal.org/)
- **File Format**: CSV (converted from TSV)
- **Key Features**:
  - TMB (nonsynonymous)
  - Fraction Genome Altered
  - Tumor Purity
  - Cancer Type
  - Smoking History
  - Specimen Type

## 🔧 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo/genemutai.git
cd genemutai
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit Dashboard

```bash
streamlit run app.py
```

## 🏗️ Model Training

1. Preprocesses clinical & genomic data.
2. Encodes categorical variables & normalizes features.
3. Trains a deep learning model for mutation prediction.
4. Saves the trained model as `mutation_prediction_model.h5`.

## 📊 Streamlit Dashboard

- **Upload CSV** → Load genomic data for predictions.
- **Mutation Predictions** → Displays risk levels.
- **Visualizations** → Bar & pie charts of mutation distributions.

## 📌 Future Enhancements

- **Deploy on Hugging Face Spaces**
- **Integrate Explainable AI (XAI) for interpretability**
- **Expand dataset for broader generalization**

## 🛠️ Technologies Used

- Python (TensorFlow, Pandas, NumPy, Matplotlib, Seaborn)
- Deep Learning (Neural Networks, TensorFlow/Keras)
- Streamlit (Web UI for interactive visualization)
- Data Processing (Feature Engineering, One-hot Encoding, Scaling)

## 🤝 Contributing

Feel free to open an issue or submit a pull request to improve the project!

## 📜 License

This project is licensed under the MIT License.



