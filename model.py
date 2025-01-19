import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle
from sklearn.utils import shuffle
from skimage.util import random_noise

# Parameters
IMAGE_SIZE = (128, 128)  # Resized image size
RANDOM_STATE = 42
IMAGES_DIR = "images"  # Path to the images folder
CALORIE_CSV_PATH = "calorie.csv"  # Path to calorie CSV file

# Data Preprocessing function
def load_images_and_labels(images_dir, calorie_df, img_size=IMAGE_SIZE):
    image_data = []
    labels = []
    calories = []

    # Loop through image files and load them
    for food_name in os.listdir(images_dir):
        food_path = os.path.join(images_dir, food_name)
        if food_name in calorie_df['Fruits'].values:
            for image_file in os.listdir(food_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    try:
                        img_path = os.path.join(food_path, image_file)
                        img = imread(img_path)
                        img_resized = resize(img, img_size, anti_aliasing=True)  # Resize to fit model input
                        img_resized = img_resized / 255.0  # Normalize pixel values
                        image_data.append(img_resized.flatten())  # Flatten for classifier input
                        labels.append(food_name)
                        calories.append(calorie_df.loc[calorie_df['Fruits'] == food_name, 'calorie'].values[0])
                    except Exception as e:
                        print(f"Error reading {image_file}: {e}")
        else:
            print(f"Skipping folder {food_name}, not in calorie CSV.")
    
    return np.array(image_data), np.array(labels), np.array(calories)

# Image Augmentation function to increase dataset size artificially
def augment_images(images, labels):
    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        # Apply random noise to simulate real-world variation
        noisy_image = random_noise(img, mode='s&p', amount=0.1)
        augmented_images.append(noisy_image)
        augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)

# Load data
calorie_df = pd.read_csv(CALORIE_CSV_PATH)
X_images, y_labels, y_calories = load_images_and_labels(IMAGES_DIR, calorie_df)

# Encode labels (fruit names) into numeric values
label_encoder = LabelEncoder()
y_labels_encoded = label_encoder.fit_transform(y_labels)

# Augment data by adding noisy variations
X_images_augmented, y_labels_augmented = augment_images(X_images, y_labels_encoded)

# Concatenate original and augmented images and labels to increase dataset size
X_images_extended = np.concatenate([X_images, X_images_augmented], axis=0)
y_labels_extended = np.concatenate([y_labels_encoded, y_labels_augmented], axis=0)
y_calories_extended = np.concatenate([y_calories, y_calories], axis=0)  # Duplicate calorie labels

# Shuffle the data to mix original and augmented images
X_images_extended, y_labels_extended, y_calories_extended = shuffle(X_images_extended, y_labels_extended, y_calories_extended, random_state=RANDOM_STATE)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, y_cal_train, y_cal_test = train_test_split(
    X_images_extended, y_labels_extended, y_calories_extended, test_size=0.2, random_state=RANDOM_STATE
)

# Compute class weights to handle imbalanced classes
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Create and train the Random Forest classifier
classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight=class_weight_dict,  # Apply class weights
    random_state=RANDOM_STATE
)
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred_labels = classifier.predict(X_test)
classification_accuracy = accuracy_score(y_test, y_pred_labels)

# Train Random Forest regressor for calorie prediction
regressor = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
regressor.fit(X_train, y_cal_train)

# Evaluate the regressor
y_pred_calories = regressor.predict(X_test)
calorie_mae = mean_absolute_error(y_cal_test, y_pred_calories)
calorie_r2 = r2_score(y_cal_test, y_pred_calories)

# Print evaluation metrics
print(f"Classification Accuracy: {classification_accuracy * 100:.2f}%")
print(f"Calorie Estimation MAE: {calorie_mae:.2f} calories")
print(f"Calorie Estimation R² Score: {calorie_r2:.2f}")

# Save the models and label encoder
with open("model.pkl", "wb") as f:
    pickle.dump((classifier, regressor, label_encoder), f)

print("Model saved to 'model.pkl'.")
