# import cv2
# import json
# import numpy as np
# import os
# import mediapipe as mp
# from scipy.interpolate import interp1d

# class Preprocessor:
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         with open(os.path.join(data_dir, 'labelled_dataset', 'labels', 'error_knees.json'), 'r') as f:
#             self.error_knees = json.load(f)
#         with open(os.path.join(data_dir, 'labelled_dataset', 'labels', 'error_elbows.json'), 'r') as f:
#             self.error_elbows = json.load(f)
#         with open(os.path.join(data_dir, 'labelled_dataset', 'splits', 'train_keys.json'), 'r') as f:
#             self.train_keys = json.load(f)
#         with open(os.path.join(data_dir, 'labelled_dataset', 'splits', 'val_keys.json'), 'r') as f:
#             self.val_keys = json.load(f)
#         with open(os.path.join(data_dir, 'labelled_dataset', 'splits', 'test_keys.json'), 'r') as f:
#             self.test_keys = json.load(f)
#         self.median_frames = 113  # Hardcode to match trained model

#     def calculate_median_frames(self):
#         # Kept for reference but overridden by hardcoded value
#         frame_counts = []
#         video_dir = os.path.join(self.data_dir, 'labelled_dataset', 'videos')
#         for key in self.train_keys + self.val_keys + self.test_keys:
#             video_path = os.path.join(video_dir, f'{key}.mp4')
#             if os.path.exists(video_path):
#                 cap = cv2.VideoCapture(video_path)
#                 frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
#                 cap.release()
#         return int(np.median(frame_counts))

#     def extract_frames(self, video_path, target_fps=30):
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video file: {video_path}")
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frames = []
#         frame_idx = 0
#         skip = max(1, int(fps / target_fps))
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if frame_idx % skip == 0:
#                 frame = cv2.resize(frame, (224, 224))
#                 frames.append(frame)
#             frame_idx += 1
#         cap.release()
#         return frames, fps

#     def detect_keypoints(self, frames):
#         mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
#         keypoints_list = []
#         for frame in frames:
#             results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             if results.pose_landmarks:
#                 landmarks = results.pose_landmarks.landmark
#                 keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks[:25]]).flatten()
#             else:
#                 keypoints = np.zeros(25 * 4)
#             keypoints_list.append(keypoints)
#         mp_pose.close()
#         return np.array(keypoints_list)

#     def interpolate_keypoints(self, keypoints, target_length):
#         if len(keypoints) == target_length:
#             return keypoints
#         x = np.linspace(0, 1, len(keypoints))
#         x_new = np.linspace(0, 1, target_length)
#         interp_func = interp1d(x, keypoints, axis=0, kind='linear')
#         return interp_func(x_new)

#     def interpolate_labels(self, labels, target_length):
#         if len(labels) == target_length:
#             return labels
#         x = np.linspace(0, 1, len(labels))
#         x_new = np.linspace(0, 1, target_length)
#         interp_func = interp1d(x, labels, axis=0, kind='nearest', fill_value='extrapolate')
#         return interp_func(x_new).astype(np.int8)

#     def get_frame_labels(self, time_stamps, error_intervals_knees, error_intervals_elbows):
#         labels_knees = np.zeros(len(time_stamps), dtype=np.int8)
#         labels_elbows = np.zeros(len(time_stamps), dtype=np.int8)
#         for start, end in error_intervals_knees:
#             labels_knees += np.logical_and(start <= time_stamps, time_stamps <= end).astype(np.int8)
#         for start, end in error_intervals_elbows:
#             labels_elbows += np.logical_and(start <= time_stamps, time_stamps <= end).astype(np.int8)
#         labels_knees = np.clip(labels_knees, 0, 1)
#         labels_elbows = np.clip(labels_elbows, 0, 1)
#         return np.stack([labels_knees, labels_elbows], axis=-1)  # Shape: (num_frames, 2)

#     def load_data(self, keys, batch_size=32):
#         X_list = []
#         y_list = []
#         for i in range(0, len(keys), batch_size):
#             batch_keys = keys[i:i + batch_size]
#             for key in batch_keys:
#                 video_path = os.path.join(self.data_dir, 'labelled_dataset', 'videos', f'{key}.mp4')
#                 if not os.path.exists(video_path):
#                     print(f"Warning: Video {video_path} not found, skipping.")
#                     continue
#                 try:
#                     frames, fps = self.extract_frames(video_path)
#                     if not frames:
#                         print(f"Warning: No frames extracted from {video_path}, skipping.")
#                         continue

#                     skip = max(1, int(fps / 30))
#                     num_frames = len(frames)
#                     time_stamps = np.array([i * skip / fps for i in range(num_frames)])

#                     keypoints = self.detect_keypoints(frames)
#                     keypoints = self.interpolate_keypoints(keypoints, self.median_frames)
#                     diff = np.diff(keypoints, axis=0, prepend=keypoints[0:1])

#                     error_intervals_knees = self.error_knees.get(key, [])
#                     error_intervals_elbows = self.error_elbows.get(key, [])
#                     labels = self.get_frame_labels(time_stamps, error_intervals_knees, error_intervals_elbows)
#                     labels = self.interpolate_labels(labels, self.median_frames)  # Interpolate labels to median_frames
                    
#                     X_list.append(diff)
#                     y_list.append(labels)
#                 except Exception as e:
#                     print(f"Error processing {video_path}: {e}")
#                     continue
#             if X_list:
#                 yield (np.array(X_list), np.array(y_list))
#                 X_list.clear()
#                 y_list.clear()




import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d

class Preprocessor:
    def __init__(self, data_dir=None, load_labels=False):
        """
        If load_labels=True, will attempt to load training labels (not recommended in production).
        Inference only mode: load_labels=False
        """
        self.data_dir = data_dir
        self.median_frames = 113  # Hardcoded to match trained model

        if load_labels and data_dir:
            import json
            import os
            with open(os.path.join(data_dir, 'labelled_dataset', 'labels', 'error_knees.json'), 'r') as f:
                self.error_knees = json.load(f)
            with open(os.path.join(data_dir, 'labelled_dataset', 'labels', 'error_elbows.json'), 'r') as f:
                self.error_elbows = json.load(f)
            with open(os.path.join(data_dir, 'labelled_dataset', 'splits', 'train_keys.json'), 'r') as f:
                self.train_keys = json.load(f)
            with open(os.path.join(data_dir, 'labelled_dataset', 'splits', 'val_keys.json'), 'r') as f:
                self.val_keys = json.load(f)
            with open(os.path.join(data_dir, 'labelled_dataset', 'splits', 'test_keys.json'), 'r') as f:
                self.test_keys = json.load(f)
        else:
            self.error_knees = {}
            self.error_elbows = {}
            self.train_keys = []
            self.val_keys = []
            self.test_keys = []

    def extract_frames(self, video_path, target_fps=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_idx = 0
        skip = max(1, int(fps / target_fps))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip == 0:
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            frame_idx += 1
        cap.release()
        return frames, fps

    def detect_keypoints(self, frames):
        mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        keypoints_list = []
        for frame in frames:
            results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks[:25]]).flatten()
            else:
                keypoints = np.zeros(25 * 4)
            keypoints_list.append(keypoints)
        mp_pose.close()
        return np.array(keypoints_list)

    def interpolate_keypoints(self, keypoints, target_length):
        if len(keypoints) == target_length:
            return keypoints
        x = np.linspace(0, 1, len(keypoints))
        x_new = np.linspace(0, 1, target_length)
        interp_func = interp1d(x, keypoints, axis=0, kind='linear')
        return interp_func(x_new)


