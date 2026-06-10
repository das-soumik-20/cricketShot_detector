import tensorflow as tf

def preprocess_data():
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    data_dir = "data"

    # 1. Training Pool: Scrambling allowed, completely isolated root pool
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True 
    )

    # 2. Evaluation Pool: Set shuffle=False so the file order stays identical every epoch
    val_and_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False # Lock the file order in place!
    )

    # 3. Carve the static evaluation pool into two permanent halves
    val_batches = len(val_and_test_ds) // 2

    val_ds = val_and_test_ds.take(val_batches)
    test_ds = val_and_test_ds.skip(val_batches)

    # 4. Performance Optimization (Optional but highly recommended)
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds