import os
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split


class AudioClipDataset:
    """
    Dataset for loading audio clips with labels (ad or broadcast)
    """
    
    def __init__(self, data_dir, duration=3.0, sample_rate=22050, overlap=0.0, cache_audio=True):
        self.data_dir = Path(data_dir)
        self.duration = duration
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.clip_length = int(duration * sample_rate)
        self.hop_size = int(self.clip_length * (1 - overlap))
        self.cache_audio = cache_audio
        self.audio_cache = {}  # Cache for loaded audio files
        
        self.label_map = {'ad': 0, 'broadcast': 1}
        self.samples = []
        self._build_dataset()
        
    def _build_dataset(self):
        for sport_dir in self.data_dir.iterdir():
            if not sport_dir.is_dir():
                continue
                
            for label_dir in sport_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                
                if label_dir.name == 'misc':
                    continue
                    
                if label_dir.name not in self.label_map:
                    continue
                
                label = self.label_map[label_dir.name]
                
                for audio_file in label_dir.glob('*.wav'):
                    audio, sr = librosa.load(str(audio_file), sr=self.sample_rate, mono=True)
                    num_clips = max(1, (len(audio) - self.clip_length) // self.hop_size + 1)
                    
                    for clip_idx in range(num_clips):
                        self.samples.append({
                            'file_path': str(audio_file),
                            'label': label,
                            'label_name': label_dir.name,
                            'sport': sport_dir.name,
                            'clip_index': clip_idx,
                            'filename': audio_file.name
                        })
        
        print(f"Dataset built: {len(self.samples)} clips from {len(set([s['file_path'] for s in self.samples]))} files")
        print(f"Labels distribution: ad={sum(1 for s in self.samples if s['label']==0)}, broadcast={sum(1 for s in self.samples if s['label']==1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def load_clip(self, idx):
        """Load a single audio clip and return as numpy array."""
        sample_info = self.samples[idx]
        file_path = sample_info['file_path']
        
        if self.cache_audio:
            if file_path not in self.audio_cache:
                audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                self.audio_cache[file_path] = audio
            else:
                audio = self.audio_cache[file_path]
        else:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        start_idx = sample_info['clip_index'] * self.hop_size
        end_idx = start_idx + self.clip_length
        
        if end_idx > len(audio):
            clip = np.zeros(self.clip_length, dtype=np.float32)
            available_samples = len(audio) - start_idx
            clip[:available_samples] = audio[start_idx:]
        else:
            clip = audio[start_idx:end_idx]
        
        return clip.astype(np.float32), sample_info['label']
    
    def get_all_data(self):
        """Load all clips into memory. Use for small datasets."""
        print(f"Loading {len(self.samples)} clips into memory...")
        X = []
        y = []
        total = len(self.samples)
        for i in range(total):
            if i % 500 == 0:
                print(f"  Progress: {i}/{total} clips loaded ({i/total*100:.1f}%)")
            clip, label = self.load_clip(i)
            X.append(clip)
            y.append(label)
        print(f"  Progress: {total}/{total} clips loaded (100.0%)")
        return np.array(X), np.array(y)


def create_tf_dataset(data_dir, duration=3.0, sample_rate=22050, overlap=0.0, 
                      batch_size=32, train_split=0.8, shuffle_buffer=1000, seed=42):

    dataset = AudioClipDataset(data_dir, duration=duration, sample_rate=sample_rate, overlap=overlap)
    X, y = dataset.get_all_data()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=(1-train_split), random_state=seed, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Train labels: ad={np.sum(y_train==0)}, broadcast={np.sum(y_train==1)}")
    print(f"Val labels: ad={np.sum(y_val==0)}, broadcast={np.sum(y_val==1)}")
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(shuffle_buffer, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    dataset_info = {
        'num_samples': len(dataset),
        'num_train': len(X_train),
        'num_val': len(X_val),
        'clip_length': dataset.clip_length,
        'sample_rate': sample_rate,
        'duration': duration,
        'num_classes': 2
    }
    
    return train_dataset, val_dataset, dataset_info


def create_generator_dataset(data_dir, duration=3.0, sample_rate=22050, overlap=0.0,
                             batch_size=32, train_split=0.8, seed=42):

    dataset = AudioClipDataset(data_dir, duration=duration, sample_rate=sample_rate, overlap=overlap)
    
    indices = np.arange(len(dataset))
    labels = np.array([s['label'] for s in dataset.samples])
    
    train_indices, val_indices = train_test_split(
        indices, test_size=(1-train_split), random_state=seed, stratify=labels
    )
    
    print(f"\nTrain set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    
    def train_generator():
        for idx in train_indices:
            clip, label = dataset.load_clip(idx)
            yield clip, label
    
    def val_generator():
        for idx in val_indices:
            clip, label = dataset.load_clip(idx)
            yield clip, label
    
    output_signature = (
        tf.TensorSpec(shape=(dataset.clip_length,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    
    train_dataset = tf.data.Dataset.from_generator(
        train_generator, 
        output_signature=output_signature
    )
    train_dataset = train_dataset.shuffle(1000, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=output_signature
    )
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    dataset_info = {
        'num_samples': len(dataset),
        'num_train': len(train_indices),
        'num_val': len(val_indices),
        'clip_length': dataset.clip_length,
        'sample_rate': sample_rate,
        'duration': duration,
        'num_classes': 2
    }
    
    return train_dataset, val_dataset, dataset_info


if __name__ == "__main__":
    data_dir = '/Users/lense/Documents/projects/entertainment/data'
    
    print("creating datasets...")
    train_ds, val_ds, info = create_tf_dataset(
        data_dir,
        duration=3.0,
        sample_rate=22050,
        overlap=0.5,
        batch_size=8,
        train_split=0.8
    )
    
    print(f"\nDataset info: {info}")
    
    # Test batch
    print("\nTesting batch loading...")
    for audio_batch, label_batch in train_ds.take(1):
        print(f"Audio batch shape: {audio_batch.shape}")
        print(f"Label batch shape: {label_batch.shape}")
        print(f"Labels: {label_batch.numpy()}")
        print(f"Audio dtype: {audio_batch.dtype}")
    
    print("\n" + "="*50)
    print("\nTesting generator-based dataset (memory efficient)...")
    train_gen_ds, val_gen_ds, gen_info = create_generator_dataset(
        data_dir,
        duration=3.0,
        sample_rate=22050,
        overlap=0.0,
        batch_size=8,
        train_split=0.8
    )
    
    for audio_batch, label_batch in train_gen_ds.take(1):
        print(f"Audio batch shape: {audio_batch.shape}")
        print(f"Labels: {label_batch.numpy()}")

