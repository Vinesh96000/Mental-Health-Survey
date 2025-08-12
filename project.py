import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- STEP 1 & 2: DATA LOADING AND PREPROCESSING ---
df = pd.read_csv('data/survey.csv')
columns_to_keep = [
    'Age', 'Gender', 'family_history', 'benefits', 
    'care_options', 'anonymity', 'leave', 'work_interfere', 'treatment'
]
df = df[columns_to_keep]

def clean_gender(gender):
    g = str(gender).lower()
    if 'female' in g or g == 'f' or 'woman' in g:
        return 'Female'
    elif 'male' in g or g == 'm' or 'man' in g:
        return 'Male'
    else:
        return 'Other'
df['Gender'] = df['Gender'].apply(clean_gender)

mode_work_interfere = df['work_interfere'].mode()[0]
df['work_interfere'] = df['work_interfere'].fillna(mode_work_interfere)

binary_cols = ['family_history', 'treatment']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

categorical_cols = ['Gender', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

# --- STEP 3: BUILD AND TRAIN THE DEEP LEARNING MODEL ---
X = df.drop('treatment', axis=1)
y = df['treatment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

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

input_size = X_train.shape[1]
model = DepressionModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- STARTING MODEL TRAINING ---")
epochs = 100
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
print("--- MODEL TRAINING COMPLETE ---")

# --- SAVE THE TRAINED MODEL AND THE SCALER ---
print("\n--- SAVING MODEL AND SCALER ---")
torch.save(model.state_dict(), 'model.pth')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Also save the columns to a file, so our app knows the exact order
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns, f)

print("Model, scaler, and columns saved successfully!")