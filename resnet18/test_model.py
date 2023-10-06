import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Define input shape and number of classes
input_shape = (224, 224, 3)  # Modify to match the input size used during training
num_classes = 2  # Number of classes (e.g., wildfire and not wildfire)

# Load the trained model
model = load_model('wildfire_detection_resnet.h5')  # Replace with the path to your saved model

# Define the test data directory
test_data_dir = '/Users/atin/Documents/GitHub/jetson-drone-project/resized_dataset/validation'  # Replace with the path to your test dataset

# Create a data generator for testing
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=32,  # Adjust batch size as needed
    class_mode='binary',  # Use 'categorical' for binary classification
    shuffle=False  # Set to False to preserve order for evaluation
)

# Evaluate the model on the test set
results = model.evaluate(test_generator)

# Predict the labels for the test set
y_true = test_generator.classes
y_pred = model.predict(test_generator)

# Convert predicted probabilities to class labels (0 or 1)
y_pred_classes = (y_pred > 0.5).astype(int)

# Print evaluation metrics
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])

# Generate and print classification report
# class_names = list(test_generator.class_indices.keys())
# report = classification_report(y_true, y_pred_classes, target_names=class_names)
# print("\nClassification Report:\n", report)

# Generate and print confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:\n", conf_matrix)
