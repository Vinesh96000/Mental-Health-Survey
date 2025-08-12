import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import pickle

# --- LOAD THE SAVED MODEL, SCALER, AND COLUMNS ---

# Define the model architecture (must be same as in training)
class DepressionModel(nn.Module):
    def __init__(self, input_size):
        super(DepressionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# Load the saved model state
# First, get the input size from the saved columns
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)
input_size = len(model_columns)

model = DepressionModel(input_size)
model.load_state_dict(torch.load('model.pth'))
model.eval() # Set model to evaluation mode

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- BUILD THE STREAMLIT WEB APP INTERFACE ---

st.title("Mental Health Treatment Prediction 🧠")
st.write("This app predicts whether a person might seek treatment for a mental health condition based on survey data. Please answer the questions below.")

# Create input fields for the user
age = st.slider("What is your age?", 18, 100, 30)
gender = st.selectbox("What is your gender?", ['Male', 'Female', 'Other'])
family_history = st.selectbox("Do you have a family history of mental illness?", ['No', 'Yes'])
benefits = st.selectbox("Does your employer provide mental health benefits?", ['No', 'Yes', "Don't know"])
care_options = st.selectbox("Do you know the options for mental health care your employer provides?", ['No', 'Yes', 'Not sure'])
anonymity = st.selectbox("Is your anonymity protected if you use mental health benefits?", ['No', 'Yes', "Don't know"])
leave = st.selectbox("How easy is it for you to take medical leave for a mental health condition?", 
                     ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"])
work_interfere = st.selectbox("Does your mental health condition interfere with your work?", ['Never', 'Rarely', 'Sometimes', 'Often'])


# --- PREPROCESS USER INPUT AND MAKE PREDICTION ---

if st.button("Predict"):
    # 1. Create a dictionary from user inputs
    input_data = {
        'Age': age,
        'family_history': 1 if family_history == 'Yes' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Other': 1 if gender == 'Other' else 0,
        'benefits_No': 1 if benefits == 'No' else 0,
        'benefits_Yes': 1 if benefits == 'Yes' else 0,
        'care_options_Not sure': 1 if care_options == 'Not sure' else 0,
        'care_options_Yes': 1 if care_options == 'Yes' else 0,
        'anonymity_No': 1 if anonymity == 'No' else 0,
        'anonymity_Yes': 1 if anonymity == 'Yes' else 0,
        'leave_Somewhat difficult': 1 if leave == 'Somewhat difficult' else 0,
        'leave_Somewhat easy': 1 if leave == 'Somewhat easy' else 0,
        'leave_Very difficult': 1 if leave == 'Very difficult' else 0,
        'leave_Very easy': 1 if leave == 'Very easy' else 0,
        'work_interfere_Often': 1 if work_interfere == 'Often' else 0,
        'work_interfere_Rarely': 1 if work_interfere == 'Rarely' else 0,
        'work_interfere_Sometimes': 1 if work_interfere == 'Sometimes' else 0,
    }

    # 2. Convert to a DataFrame with the correct column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # 3. Scale the 'Age' column using the loaded scaler
    input_df['Age'] = scaler.transform(input_df[['Age']])

    # 4. Convert to a PyTorch Tensor
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

    # 5. Make a prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        probability = prediction.item()
    
    # 6. Display the result
    st.subheader("Prediction Result")
    if probability >= 0.5:
        st.success(f"There is a high likelihood that you would seek treatment. (Probability: {probability:.2f})")
        st.write("It's a sign of strength to seek help. Consider talking to a professional.")
    else:
        st.info(f"There is a lower likelihood that you would seek treatment. (Probability: {probability:.2f})")
        st.write("Remember to prioritize your mental well-being, regardless of the score.")