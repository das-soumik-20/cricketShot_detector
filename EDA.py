import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_data


def eda():
    train_ds, val_ds, test_ds = preprocess_data()
    
    # Class counts
    print("Class names:", train_ds.class_names)
    for images, labels in train_ds.unbatch().batch(10000).take(1):
        unique, counts = np.unique(labels.numpy(), return_counts=True)
        for u, c in zip(unique, counts):
            print(f"{train_ds.class_names[u]}: {c} images")

    # Visual sample — 3 images from each class
    plt.figure(figsize=(12, 10))
    sample_images = {name: [] for name in train_ds.class_names}

    for images, labels in train_ds.unbatch():
        label_name = train_ds.class_names[labels.numpy()]
        if len(sample_images[label_name]) < 3:
            sample_images[label_name].append(images.numpy().astype("uint8"))
        if all(len(v) == 3 for v in sample_images.values()):
            break

    for i, (class_name, imgs) in enumerate(sample_images.items()):
        for j, img in enumerate(imgs):
            plt.subplot(4, 3, i*3 + j + 1)
            plt.imshow(img)
            plt.title(class_name)
            plt.axis("off")

    plt.tight_layout()
    plt.show()