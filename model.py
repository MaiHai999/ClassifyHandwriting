import numpy as np
import csv
import random
from tensorflow import keras

num_class = 26
threshold = 0.8

def convertOnehotVetor(arr , N_Class):
    # Create a one-hot encoding matrix
    one_hot_matrix = np.zeros((len(arr), N_Class))

    # Fill in the one-hot encoding matrix
    for i in range(len(arr)):
        one_hot_matrix[i, arr[i]] = 1

    return one_hot_matrix

# read data
with open('/content/drive/MyDrive/dataset/A_Z Handwritten Data.csv', 'r') as csv_file:
    result = csv.reader(csv_file)
    rows = []

    for row in result:
        rows.append(row)

#creat train set
train_data = []
train_label = []

for row in rows:
    if all(row[1:]):
        x = np.array([int(j) for j in row[1:]])
        train_data.append(x)
        train_label.append(int(row[0]))

shuffle_order = list(range(len(train_data)))
random.shuffle(shuffle_order)

train_data = np.array(train_data)
train_label = np.array(train_label)

train_data = train_data[shuffle_order]
train_label = train_label[shuffle_order]

#validation
train_x = train_data[:int(len(train_data) * threshold)]
print(train_x.shape)
train_y = train_label[:int(len(train_data) * threshold)]
train_y = convertOnehotVetor(train_y , num_class)

val_x = train_data[int(len(train_data) * threshold):]
val_y = train_label[int(len(train_data) * threshold):]
val_y = convertOnehotVetor(val_y , num_class)


# Create a simple feedforward neural network
model = keras.Sequential([
    keras.layers.Dense(255, activation='relu', input_shape=(train_x.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_class, activation='sigmoid')

])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



# Train the model
model.fit(train_x, train_y, epochs=10, batch_size=32)  # X_train and y_train are your training data and labels

#save model
model.save("preTrain/my_model.h5")

# Evaluate the model on a test dataset
test_loss, test_accuracy = model.evaluate(val_x, val_y)  # X_test and y_test are your test data and labels

print(f'Test accuracy: {test_accuracy * 100:.2f}%')

