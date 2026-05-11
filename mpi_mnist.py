
!pip install mpi4py

import tensorflow as tf
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Reshape data
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Split dataset
n = len(x_train)
chunk = n // size

start = rank * chunk
end = (rank + 1) * chunk

x_chunk = x_train[start:end]
y_chunk = y_train[start:end]

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_chunk, y_chunk, epochs=1)

# Test accuracy
loss, acc = model.evaluate(x_test, y_test)

# Combine accuracy
final_acc = comm.allreduce(acc, op=MPI.SUM) / size

# Output
if rank == 0:
    print("Test Accuracy:", final_acc)
