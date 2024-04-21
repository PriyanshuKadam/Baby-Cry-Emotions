import streamlit as st
import librosa
import numpy as np
import pickle

# Load the trained model from the pickle file
with open('../data/baby_cry_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define extract_features function
def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Additional features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Aggregate statistics for each feature
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_rolloff_mean = np.mean(spectral_rolloff)

    # Concatenate selected features
    features = np.concatenate([
        [zero_crossing_rate_mean,
         spectral_centroid_mean,
         spectral_rolloff_mean],
        mfccs_mean, mfccs_std
    ])

    return features

# Mapping dictionary for Reason
reason_mapping = {
    0: 'belly pain',
    1: 'hunger',
    2: 'fear',
    3: 'discomfort due to temperature',
    4: 'tiredness',
    5: 'general discomfort',
    6: 'loneliness',
    7: 'needs burping'
}

# Streamlit app
st.title('Baby Cry Emotions')

uploaded_file = st.file_uploader("Upload baby cry audio file", type=['wav'])

if uploaded_file is not None:
    # Extract features from the uploaded audio file
    features = extract_features(uploaded_file)

    # Get user input for baby's gender and age
    gender = st.selectbox('Select baby\'s gender', options=['male', 'female'])
    age = st.selectbox('Select baby\'s age', options=['0 to 4 weeks old', '4 to 8 weeks old',
                                                      '2 to 6 months old', '7 month to 2 years old',
                                                      'more than 2 years old'])

    # Map gender and age to numerical values
    gender_mapping = {'male': 0, 'female': 1}
    age_mapping = {'0 to 4 weeks old': 0, '4 to 8 weeks old': 1, '2 to 6 months old': 2,
                   '7 month to 2 years old': 3, 'more than 2 years old': 4}
    gender_value = gender_mapping[gender]
    age_value = age_mapping[age]

    # Concatenate features with gender and age
    features = np.append(features, [gender_value, age_value])

    # Reshape features for prediction
    features = features.reshape(1, -1)

    # Predict the reason for baby's cry
    prediction = model.predict(features)
    predicted_reason = reason_mapping[prediction[0]]

    # Personalized output based on gender
    if gender == 'male':
        st.write(f"He is crying because he is experiencing {predicted_reason}.")
    else:
        st.write(f"She is crying because she is experiencing {predicted_reason}.")
