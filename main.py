import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K

# Load the data required for detecting the license plates from cascade classifier.
plate_cascade = cv2.CascadeClassifier('data/indian_license_plate.xml')

def detect_plate(img, text=''):  # Detects and performs blurring on the number plate.
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2,
                                                minNeighbors=7)
    for (x, y, w, h) in plate_rect:
        roi_ = roi[y:y + h, x:x + w, :]  # Extract Region of Interest for blurring.
        plate = roi[y:y + h, x:x + w, :]
        cv2.rectangle(plate_img, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155),
                      3)  # Drawing rectangle around the detected contours.

    if text != '':
        plate_img = cv2.putText(plate_img, text, (x - w // 2, y - h // 2),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (51, 181, 155), 1, cv2.LINE_AA)

    return plate_img, plate  # Return processed image.

def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

# Load input image
img = cv2.imread('data/in.jpg')
display(img, 'Input Image')

# Detect license plate in image
output_img, plate = detect_plate(img)
display(output_img, 'Detected License Plate in Input Image')
display(plate, 'Extracted License Plate')

# Match contours to license plate or character template
def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(intX)

            char_copy = np.zeros((44, 24))
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(ii, cmap='gray')

            char = cv2.subtract(255, char)

            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)

    plt.show()

    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = [img_res[idx] for idx in indices]
    img_res = np.array(img_res_copy)

    return img_res

# Find characters in the resulting images
def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    dimensions = [LP_WIDTH / 6,
                  LP_WIDTH / 2,
                  LP_HEIGHT / 10,
                  2 * LP_HEIGHT / 3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg', img_binary_lp)

    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

char = segment_characters(plate)

# Display extracted characters
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')

# Fix model prediction method
def fix_dimension(img):
    return np.stack([img] * 3, axis=-1)

# Custom F1 Score for evaluation (Updated to avoid the TensorFlow issue)
def f1score(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

# Model Architecture
model = Sequential()
model.add(Conv2D(16, (22, 22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16, 16), activation='relu', padding='same'))
model.add(Conv2D(64, (8, 8), activation='relu', padding='same'))
model.add(Conv2D(64, (4, 4), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=[f1score])

# Dataset preparation
train_datagen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.1, height_shift_range=0.1)
path = 'data/data'
train_generator = train_datagen.flow_from_directory(path + '/train', target_size=(28, 28), batch_size=1, class_mode='sparse')
validation_generator = train_datagen.flow_from_directory(path + '/val', target_size=(28, 28), batch_size=1, class_mode='sparse')

# Callbacks for early stopping and model checkpointing
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
]

# Train model
model.fit(train_generator, validation_data=validation_generator, epochs=80, verbose=1, callbacks=callbacks)

# Predict and show results
def show_results():
    dic = {i: c for i, c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    output = []
    for ch in char:
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)
        y_ = np.argmax(model.predict(img), axis=-1)[0]
        character = dic[y_]
        output.append(character)
    return ''.join(output)

# Final display
plate_number = show_results()

# Print the predicted license plate number on the command prompt
print(f"Predicted License Plate: {plate_number}")

# Detect and display the output image with predicted plate number
output_img, plate = detect_plate(img, plate_number)
plt.imshow(output_img)
plt.title(f'Predicted License Plate: {plate_number}')
plt.axis('off')
plt.show()

# Save the output image with the predicted license plate
output_file_path = 'predicted_license_plate.jpg'  # Set the desired file name and path
cv2.imwrite(output_file_path, output_img)  # Save the image to the file

# Print the location where the output image is saved
print(f"Output image saved as: {output_file_path}")
