💳 Credit Risk Prediction Using Machine Learning
🧠 Overview

This project predicts whether a loan applicant is a Good or Bad credit risk using Machine Learning.
It leverages the German Credit Data dataset, applies data preprocessing, trains an Extra Trees Classifier, and provides a Streamlit web app for real-time predictions.

🚀 Project Structure
📁 Credit-Risk-Modeling/
│
├── 📄 train_credit_model.py       # Model training & encoding pipeline
├── 📄 app.py                      # Streamlit web app for predictions
├── 📄 german_credit_data.csv      # Dataset used for training
├── 📄 README.md                   # Project documentation
├── 📄 extra_trees_credit_model.pkl       # Saved trained model
├── 📄 target_encoder.pkl                 # Encoder for target column
├── 📄 Sex_encoder.pkl                   # Encoder for 'Sex'
├── 📄 Housing_encoder.pkl               # Encoder for 'Housing'
├── 📄 Saving accounts_encoder.pkl       # Encoder for 'Saving accounts'
├── 📄 Checking account_encoder.pkl      # Encoder for 'Checking account'
└── 📁 __pycache__/ (auto-generated)

⚙️ Features

✅ Data preprocessing with Label Encoding
✅ Model training using Extra Trees Classifier
✅ Model accuracy evaluation and export
✅ Interactive Streamlit web app for credit risk prediction
✅ Modular structure for scalability and deployment

🗂️ Dataset

The dataset used is the German Credit Data from UCI Machine Learning Repository.
It contains customer attributes such as:

Feature	Description
Age	Applicant's age
Sex	Male/Female
Job	Job type (0–3)
Housing	Own, Rent, or Free
Saving accounts	Level of savings
Checking account	Level of checking account balance
Credit amount	Loan amount requested
Duration	Duration of credit in months
Risk	Target variable — Good or Bad
🧩 Model Workflow

Data Loading → Read dataset using pandas

Feature Selection → Select relevant numerical & categorical columns

Label Encoding → Convert categorical data to numeric format

Train-Test Split → 80% training, 20% testing

Model Training → Extra Trees Classifier with balanced class weights

Evaluation → Print model accuracy

Save Artifacts → Export model and encoders using joblib

🖥️ Streamlit App Overview

The Streamlit app (app.py) allows users to:

Input applicant information (age, job, housing, etc.)

Automatically encode the inputs using saved encoders

Predict whether the credit risk is Good or Bad

📦 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<PRANAVDEVJ>/Credit-Risk-Modeling.git
cd Credit-Risk-Modeling

2️⃣ Create a Virtual Environment (recommended)
python -m venv venv
venv\Scripts\activate      # For Windows
# or
source venv/bin/activate   # For macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt


(If you haven’t created one yet, I can generate a perfect requirements.txt for you — just ask!)

4️⃣ Train the Model
python train_credit_model.py

5️⃣ Run the Streamlit App
streamlit run app.py

📊 Sample Output
✅ Data loaded successfully!
✅ Encoders saved successfully!
✅ Model trained successfully! Accuracy: 0.78
✅ Model saved as 'extra_trees_credit_model.pkl'


When the app runs:

A user-friendly form appears to enter applicant details

The app predicts Good or Bad credit risk instantly

🧰 Technologies Used
Category	Tools
Programming	Python
Libraries	pandas, scikit-learn, joblib, streamlit
Model	Extra Trees Classifier
Dataset	German Credit Data (UCI)
🧑‍💻 Author

Pranav Dev J
💼 Data Science & Machine Learning Enthusiast
📧 [jpranavdev@gmail.com
]
🌐 [https://www.linkedin.com/in/pranavdevj/]

⭐ Future Enhancements

Add feature importance visualization

Integrate SHAP for explainable AI insights

Deploy app on Streamlit Cloud / Render / HuggingFace Spaces

Support more datasets for better generalization
