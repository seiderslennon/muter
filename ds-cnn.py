import tensorflow as tf
from tensorflow import keras
from audio_dataset import create_tf_dataset
layers = keras.layers

def tiny_dscnn_1d(input_len=16000, num_classes=2, base=16):
    inp = keras.Input(shape=(input_len, 1), name="wav")
    x = layers.Conv1D(base, 9, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    for out_ch in [32, 64, 64]:
        x = layers.DepthwiseConv1D(9, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.)(x)
        x = layers.Conv1D(out_ch, 1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.)(x)

    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

if __name__ == "__main__":
    DATA_DIR = '/Users/lense/Documents/projects/entertainment/data'
    DURATION = 1.0  # seconds
    SAMPLE_RATE = 16000  # Hz
    BATCH_SIZE = 32
    EPOCHS = 50
    
    print("Creating datasets...")
    train_ds, val_ds, info = create_tf_dataset(
        DATA_DIR,
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        overlap=0.5,
        batch_size=BATCH_SIZE,
        train_split=0.8
    )
    
    train_ds = train_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))
    val_ds = val_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))
    
    input_len = info['clip_length']
    model = tiny_dscnn_1d(input_len=input_len, num_classes=2, base=16)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
    )
    
    model.save('ad_detector_model.keras')
    print("\nModel saved to 'ad_detector_model.keras'")
    
    print("\nFinal evaluation on validation set:")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
