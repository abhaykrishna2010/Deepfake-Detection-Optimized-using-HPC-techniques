import time
import tensorflow as tf
from keras.models import Sequential #type:ignore
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization #type:ignore
from keras.optimizers import Adam #type:ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau #type:ignore
from data_preparation import load_dataset

# Record start time
start_time = time.time()

# Load data
X_train, X_val, Y_train, Y_val = load_dataset('C:/Users/abhay/OneDrive/Documents/VIT/Sem 7/HPC/Project/CASIA2/Au/', 'C:/Users/abhay/OneDrive/Documents/VIT/Sem 7/HPC/Project/CASIA2/Tp/')

# Optimized data pipeline
def create_dataset(X, Y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = create_dataset(X_train, Y_train, batch_size=32)
val_dataset = create_dataset(X_val, Y_val, batch_size=32)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_uniform'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Optimizer with basic learning rate
optimizer = Adam(learning_rate=1e-4)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=5, 
    callbacks=[early_stopping, reduce_lr], 
    verbose=2
)

# Record end time
end_time = time.time()

# Print time taken
print(f"Non-HPC Training Time: {end_time - start_time:.2f} seconds")
