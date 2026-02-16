import os
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import set_global_policy
from utils.preprocess import Preprocessor
from pipelines.lstm.model import LSTMTrainer
import numpy as np
from IPython.display import display, Javascript
import glob



# Initialize model
preprocessor = Preprocessor("/content/drive/MyDrive/backend/data")
input_shape = (preprocessor.median_frames, 100)
trainer = LSTMTrainer(input_shape)

# Split data
train_keys_limited = preprocessor.train_keys[:1000] if len(preprocessor.train_keys) > 1000 else preprocessor.train_keys
val_keys_limited = preprocessor.val_keys[:300] if len(preprocessor.val_keys) > 300 else preprocessor.val_keys
test_keys_limited = preprocessor.test_keys[:300] if len(preprocessor.test_keys) > 300 else preprocessor.test_keys
print(f"Train keys: {len(train_keys_limited)}, Val keys: {len(val_keys_limited)}, Test keys: {len(test_keys_limited)}")

# Create TF datasets
def create_dataset(keys):
    def generator():
        for X_batch, y_batch in preprocessor.load_data(keys, batch_size=32):
            yield X_batch, y_batch
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.float32, tf.int8),
        output_shapes=([None, preprocessor.median_frames, 100], [None, preprocessor.median_frames, 2])
    ).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_keys_limited)
val_dataset = create_dataset(val_keys_limited)
test_dataset = create_dataset(test_keys_limited)

# Define checkpoint path
checkpoint_dir = '/content/drive/MyDrive/backend/pipelines/lstm/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch_{epoch:02d}.h5")
final_model_path = os.path.join(checkpoint_dir, "lstm_model.h5")

# Load latest checkpoint
checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.h5")))
latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None
if latest_checkpoint:
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    trainer.model = load_model(latest_checkpoint)
    epoch_num = int(latest_checkpoint.split('_epoch_')[1].split('.h5')[0])
    initial_epoch = epoch_num
else:
    print("Starting training from scratch")
    initial_epoch = 0

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    checkpoint_path,
    save_best_only=False,
    monitor='val_accuracy',
    mode='max',
    save_freq='epoch'
)

# Train
total_epochs = 50
print(f"Training for {total_epochs} epochs, starting from epoch {initial_epoch + 1}")
history = trainer.train(
    x=train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    initial_epoch=initial_epoch
)

# Save final model
save_model(trainer.model, final_model_path)
print("Final model saved to:", final_model_path)

# Combine history
history_dict = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
for epoch in range(1, total_epochs + 1):
    history_file = os.path.join(checkpoint_dir, f"history_epoch_{epoch}.npy")
    if os.path.exists(history_file):
        h = np.load(history_file, allow_pickle=True).item()
        for key in history_dict:
            if key in h:
                history_dict[key].append(h[key])

# Print history
print("Training History - Accuracy:", [f"{acc:.4f}" for acc in history_dict.get('accuracy', [])])
print("Validation History - Accuracy:", [f"{acc:.4f}" for acc in history_dict.get('val_accuracy', [])])
print("Training History - Loss:", [f"{loss:.4f}" for loss in history_dict.get('loss', [])])
print("Validation History - Loss:", [f"{loss:.4f}" for loss in history_dict.get('val_loss', [])])

# Keep Colab active
display(Javascript('setInterval(() => { console.log("Active"); }, 60000);'))