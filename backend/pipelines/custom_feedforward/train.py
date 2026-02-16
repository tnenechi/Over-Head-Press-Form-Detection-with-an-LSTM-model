import numpy as np
from utils.preprocess import Preprocessor
from pipelines.custom_feedforward.model import FeedforwardTrainer
import os
from IPython.display import display, Javascript
import glob

# Initialize preprocessor and trainer
preprocessor = Preprocessor("/content/drive/MyDrive/backend/data")
input_shape = (preprocessor.median_frames, 100)
trainer = FeedforwardTrainer(input_shape)

# Load and prepare data
def load_all_data(keys):
    X, y = [], []
    for X_batch, y_batch in preprocessor.load_data(keys, batch_size=32):
        X.append(X_batch)
        y.append(y_batch)
    return np.vstack(X), np.vstack(y)

train_X, train_y = load_all_data(preprocessor.train_keys)
val_X, val_y = load_all_data(preprocessor.val_keys)
test_X, test_y = load_all_data(preprocessor.test_keys)

# Define checkpoint path
checkpoint_dir = "/content/drive/MyDrive/backend/pipelines/custom_feedforward/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch_{epoch:02d}.npz")
final_model_path = os.path.join(checkpoint_dir, "custom_feedforward_model.npz")

# Load latest checkpoint
checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.npz")))
latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None
if latest_checkpoint:
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    trainer.load_weights(latest_checkpoint)
    initial_epoch = int(latest_checkpoint.split('_epoch_')[1].split('.npz')[0])
else:
    print("Starting training from scratch")
    initial_epoch = 0

# Checkpoint callback
class CheckpointCallback:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
    def on_epoch_end(self, epoch):
        np.savez(self.checkpoint_path.format(epoch=epoch + 1), weights=trainer.weights, biases=trainer.biases)

# Train
total_epochs = 30
checkpoint = CheckpointCallback(checkpoint_path)
print(f"Training for {total_epochs} epochs, starting from epoch {initial_epoch + 1}")
history = trainer.train(
    X=train_X,
    y=train_y,
    epochs=total_epochs - initial_epoch,
    batch_size=32,
    validation_data=(val_X, val_y),
    callbacks=[checkpoint]
)

# Save final model
np.savez(final_model_path, weights=trainer.weights, biases=trainer.biases)
print("Final model saved to:", final_model_path)

# Save history
history_path = os.path.join(checkpoint_dir, f"history_epoch_{total_epochs}.npy")
np.save(history_path, history)

# Combine history
history_dict = {}
for epoch in range(1, total_epochs + 1):
    history_file = os.path.join(checkpoint_dir, f"history_epoch_{epoch}.npy")
    if os.path.exists(history_file):
        h = np.load(history_file, allow_pickle=True).item()
        for key, value in h.items():
            history_dict.setdefault(key, []).extend(value)

# Print history
print("Training History - Accuracy:", [f"{acc:.4f}" for acc in history_dict.get('accuracy', [])])
print("Validation History - Accuracy:", [f"{acc:.4f}" for acc in history_dict.get('val_accuracy', [])])

# Keep Colab active
display(Javascript('setInterval(() => { console.log("Active"); }, 60000);'))