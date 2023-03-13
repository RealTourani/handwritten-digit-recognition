import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, precision_score

# Read CSV file containing paths and labels
df = pd.read_csv('labels.csv')

# Create lists to hold the image data and labels
images = []
labels = []

# Loop through each row in the CSV file
for index, row in df.iterrows():
    # Load the image using PIL
    image = Image.open(row['name'].replace("\\", "/"))
    # Resize the image to 64x64
    image = image.resize((64, 64))
    # Convert the image to a numpy array
    image = np.array(image)
    # Add the image and label to their respective lists
    images.append(image)
    labels.append(row['label'])

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Get the number of unique labels in the dataset
num_classes = len(np.unique(labels))
print('Number of unique labels:', num_classes)

# Check label values
if np.max(labels) > num_classes:
    print('Error: label values are not integers ranging from 1 to num_classes')
else:
    # Subtract 1 from the labels to convert them to the range 0 to num_classes-1
    labels -= 1

    # Convert the labels to one-hot encoding
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Convert predictions from one-hot encoding to integer labels
    y_pred_int = np.argmax(y_pred, axis=1)
    y_test_int = np.argmax(y_test, axis=1)

    # Calculate F1 score and precision
    f1 = f1_score(y_test_int, y_pred_int, average='macro')
    precision = precision_score(y_test_int, y_pred_int, average='macro')

    print('F1 score:', f1)
    print('Precision:', precision)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)

    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    model.save('trained_model.h5')
