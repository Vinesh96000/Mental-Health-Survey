***Predicting Mental Health Treatment-Seeking Behavior***
A deep learning project that predicts whether an individual is likely to seek treatment for a mental health condition based on data from a tech workplace survey. This project uses a custom-built neural network with PyTorch and is deployed as an interactive web application using Streamlit.


Features
Interactive UI: A user-friendly web interface built with Streamlit for easy data input.

Real-Time Predictions: Get instant predictions from the trained deep learning model.

Data-Driven Insights: The model is trained on the 2014 OSMI Mental Health in Tech Survey dataset.

End-to-End Pipeline: Covers the complete machine learning lifecycle from data cleaning and preprocessing to model training, evaluation, and deployment.

Tech Stack & Libraries
Language: Python

Model Development: PyTorch

Web Framework: Streamlit

Data Manipulation: Pandas

Machine Learning Utilities: Scikit-learn

Project Structure
├── app.py                  # The main script for the Streamlit web application.
├── project.py              # The script for data preprocessing and model training.
├── requirements.txt        # A list of Python libraries required to run the project.
├── .gitignore              # Specifies files for Git to ignore.
├── model.pth               # The saved weights of the trained PyTorch model.
├── scaler.pkl              # The saved Scikit-learn scaler for data normalization.
├── model_columns.pkl       # The saved list of feature columns for the model.
└── README.md               # You are here!
How to Run This Project Locally
Follow these steps to set up and run the project on your own machine.

Prerequisites
Git

Python 3.8+

A GitHub account

Installation & Setup
Clone the repository:

Bash

git clone https://github.com/Vinesh96000/Mental-Health-Survey.git
cd Mental-Health-Survey
Create and activate a virtual environment:

Bash

# Create the virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
Install the required dependencies:

Bash

pip install -r requirements.txt
Running the Application
Start the Streamlit app:

Bash

streamlit run app.py
Your web browser should automatically open to the application's local address.

Model Details
Model Type: Multi-Layer Perceptron (MLP) / Feedforward Neural Network.

Architecture: Input Layer -> Hidden Layer 1 (16 neurons, ReLU) -> Hidden Layer 2 (8 neurons, ReLU) -> Output Layer (1 neuron, Sigmoid).

Target Variable: treatment (predicting if a person has sought treatment).

Evaluation Metrics:

Accuracy: ~70.2%

Precision: ~68.8%

Recall: ~71.5%

F1-Score: ~70.1%

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
This project uses the Mental Health in Tech Survey dataset from the Open Sourcing Mental Illness (OSMI) organization.
