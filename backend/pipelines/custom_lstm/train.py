import os
import numpy as np
import tensorflow as tf
from pipelines.custom_lstm.model import CustomLSTM
from utils.preprocess import Preprocessor


DATA_DIR = "./data"
EPOCHS = 50
BATCH_SIZE = 32
hidden_sizes = [64, 128, 256, 128, 64]
output_size = 2
input_size = 100  
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Preprocess and get seq length
preprocessor = Preprocessor(DATA_DIR)
seq_length = preprocessor.median_frames

# Build model
trainer = CustomLSTM(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    output_size=output_size,
    seq_length=seq_length
)

keras_model = trainer.build_keras_model()
keras_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Data generators
train_gen = preprocessor.load_data(preprocessor.train_keys, batch_size=BATCH_SIZE)
val_gen = preprocessor.load_data(preprocessor.val_keys, batch_size=BATCH_SIZE)

# Training loop with history saving
history_records = []

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    # Train for one epoch
    history = keras_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=1,
        verbose=1
    )

    # Save training history for this epoch
    history_dict = {
        "epoch": epoch,
        "loss": history.history["loss"][-1],
        "val_loss": history.history["val_loss"][-1],
        "accuracy": history.history["accuracy"][-1],
        "val_accuracy": history.history["val_accuracy"][-1],
    }
    history_records.append(history_dict)

    # Save as numpy checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"history_epoch_{epoch}.npy")
    np.save(checkpoint_path, history_dict)
    print(f"Saved checkpoint: {checkpoint_path}")

# Save final Keras model as .h5
model_output_path = os.path.join(CHECKPOINT_DIR, "custom_lstm_model.h5")
keras_model.save(model_output_path)
print(f"Final model saved: {model_output_path}")