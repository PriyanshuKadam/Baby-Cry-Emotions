import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('../data/cry.csv')  # Use forward slash (/) for path separator

# Split data into features (X) and labels (y)
X = df.drop(columns=['Reason'])  # Exclude Reason for now
y = df['Reason']

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model to a pickle file
with open('../data/baby_cry_model.pkl', 'wb') as f:  # Use forward slash (/) for path separator
    pickle.dump(model, f)