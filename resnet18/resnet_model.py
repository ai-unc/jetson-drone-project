import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def residual_block(x, filters, stride=1):
    identity = x

    # First convolution layer
    x = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolution layer
    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # If the stride is greater than 1, use a convolutional layer to adjust the identity size
    if stride != 1:
        identity = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same')(identity)
        identity = BatchNormalization()(identity)

    # Add the identity (skip connection) to the output
    x = Add()([x, identity])
    x = Activation('relu')(x)
    
    return x

def build_resnet(input_shape, num_classes, num_blocks=[2, 2, 2, 2]):
    # Input layer
    input_tensor = Input(shape=input_shape)
    
    # Initial convolution layer
    x = Conv2D(64, kernel_size=(7, 7), strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Max pooling
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Building the residual blocks
    for block_num, num_filters in enumerate([64, 128, 256, 512]):
        for _ in range(num_blocks[block_num]):
            stride = 1 if block_num == 0 else 2  # Downsample the first block of each stage
            x = residual_block(x, filters=num_filters, stride=stride)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer for classification
    x = Dense(num_classes, activation='sigmoid')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=x, name='resnet')
    
    return model


# Define input shape and number of classes (2 for binary classification: wildfire or not)
input_shape = (224, 224, 3)
num_classes = 1

# Build the ResNet model
model = build_resnet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use categorical cross-entropy for binary classification
              metrics=['accuracy'])

# Define data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,             # Normalize pixel values to [0, 1]
    rotation_range=20,            # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,        # Randomly shift images horizontally by up to 20% of the width
    height_shift_range=0.2,       # Randomly shift images vertically by up to 20% of the height
    horizontal_flip=True,         # Randomly flip images horizontally
    zoom_range=0.2                # Randomly zoom in on images by up to 20%
)

# Define data augmentation for validation (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define the directories for your dataset
train_data_dir = 'resized_dataset/train'
validation_data_dir = 'resized_dataset/validation'

# Create data generators for training and validation
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for binary classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for binary classification
)

# Train the model
epochs = 10
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Save the trained model
model.save('wildfire_detection_resnet.keras')

# Optionally, evaluate the model on a test set
# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode='categorical'  # Use 'categorical' for binary classification
# )
# test_loss, test_acc = model.evaluate(test_generator)
# print(f'Test accuracy: {test_acc}')
# In this code:

# We define the ResNet architecture and compile it.
# We use ImageDataGenerator for data augmentation during training.
# We organize your dataset into training and validation directories and create data generators for both.
# We train the model using the generators.
# Optionally, you can evaluate the model on a test set if available.
# Make sure to adjust the dataset paths, image dimensions, batch size, and training parameters as needed for your specific dataset and requirements.





