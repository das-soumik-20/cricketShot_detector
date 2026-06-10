import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from preprocess import preprocess_data

# --- Clean Build Function ---
def build_model(num_classes=4):
    # 1. Keep augmentation completely local to the model architecture space
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    # 2. Keep the base model securely encapsulated
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Start with it frozen for Feature Extraction
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)         
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  
    x = base_model(x, training=False)     # Keeps Batch Normalization layers stable
    x = layers.GlobalAveragePooling2D()(x) 
    x = layers.Dropout(0.3)(x)             

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Return BOTH the main wrapper and the inner base model tracking reference cleanly
    return tf.keras.Model(inputs, outputs), base_model


if __name__ == "__main__":
    # Load your isolated datasets
    train_ds, val_ds, test_ds = preprocess_data()
    
    # Build the model and unpack both necessary tracking pointers
    model, base_model = build_model(num_classes=4)

    # --- Phase 1: Train Top Layers ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("\n Kicking off Phase 1 Training...")
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    # --- Phase 2: Safe Fine-Tuning ---
    # Now mutating base_model works perfectly because it is a direct pointer to the nested child graph!
    base_model.trainable = True

    print(f"Total base layers: {len(base_model.layers)}")
    fine_tune_at = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Recompile to push the unfreezing changes down into the engine
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Your second summary will now accurately show a drop in Non-trainable params!
    model.summary()

    print("\n Kicking off Phase 2 Fine-Tuning...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,        
        initial_epoch=10
    )

    model.save("Cricket_Shot_Detection_model.keras")
    print("[SUCCESS] Model training complete and asset saved!")