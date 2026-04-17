
# Scenario 1: Data Validation 
def validate_data(data):
    invalid_entries = []
    for entry in data:
        if not isinstance(entry.get("age"), int):
            invalid_entries.append(entry)
    return invalid_entries



# Scenario 2: Logging Decorator 
import time

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

@log_execution_time
def calculate_sum(n):
    return sum(range(1, n + 1))
calculate_sum(100)



# Scenario 3: Missing Value Handling 

import pandas as pd
df = pd.DataFrame({"income": [50000, 60000, None, 75000, None, 1000000]})

skewness = df["income"].skew()
if abs(skewness) <= 0.5:
    df["income"] = df["income"].fillna(df["income"].median())
else:
    df["income"] = df["income"].fillna(df["income"].mode().iloc[0])
print(df)



# Scenario 4: Text Pre-processing 

import pandas as pd
import re

df = pd.DataFrame({"text": ["Hello! World@", "Python #is Great!", "AI & ML rocks!"]})
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()
df["cleaned"] = df["text"].apply(clean_text)
print(df)


# Scenario 5: Hyperparameter Tuning 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def tune_model(X, y):
    model = RandomForestClassifier()

    params = {
        "max_depth": [3, 5, 7],
        "n_estimators": [50, 100]
    }

    grid = GridSearchCV(model, params, cv=3)
    grid.fit(X, y)

    return grid.best_params_, grid.best_estimator_


#  Scenario 6: Custom Evaluation Metric 

def weighted_accuracy(y_true, y_pred):
    weights = {0: 1, 1: 2}
    weighted_correct = sum(weights[yt] for yt, yp in zip(y_true, y_pred) if yt == yp)
    total_weight = sum(weights[yt] for yt in y_true)
    return weighted_correct / total_weight

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
print(weighted_accuracy(y_true, y_pred))



#  Scenario 7: Image Augmentation 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_augmentor():
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )
    return datagen



#  Scenario 8: EarlyStopping Callback

from tensorflow.keras.callbacks import EarlyStopping

def get_early_stop():
    callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
    return callback




#  Scenario 9: Structured Response Generation (Gemini API) 

import json

def get_json_response(model, prompt):
    response = model.generate_content(prompt).text

    try:
        data = json.loads(response)
    except:
        # fallback if response is not valid JSON
        data = {
            "result": response,
            "note": "Response was not valid JSON"
        }

    return data



#  Scenario 10: Summarization with Constraints (Prompt Writing)

prompt = """You are a helpful assistant.
Provide a summary of the given news article in exactly 2 sentences.
Ensure the entire summary is within 50 words.
If it exceeds 50 words, refine it to meet the limit while still keeping exactly 2 complete sentences and capturing the essential information.
Return only the final summary.
"""
