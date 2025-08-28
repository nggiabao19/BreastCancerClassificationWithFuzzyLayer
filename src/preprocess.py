import numpy as np
import tensorflow as tf
from collections import Counter

def advanced_augment_data(images, labels):
    print("Balancing and augmenting data...")
    counter = Counter(labels)
    max_count = max(counter.values())
    balanced_images = []
    balanced_labels = []
    for cls_id, count in counter.items():
        idxs = np.where(labels == cls_id)[0]
        if count < max_count:
            chosen = np.random.choice(idxs, size=max_count, replace=True)
        else:
            chosen = idxs
        balanced_images.append(images[chosen])
        balanced_labels.append(labels[chosen])
    images = np.vstack(balanced_images)
    labels = np.hstack(balanced_labels)
    perm = np.random.permutation(len(labels))
    images = images[perm]
    labels = labels[perm]
    print(f"After balancing: {len(images)} images, labels: {Counter(labels)}")

    aug_images = []
    aug_labels = []
    for img, label in zip(images, labels):
        aug_images.append(img)
        aug_labels.append(label)
        rotated = tf.image.rot90(img, k=np.random.randint(0, 4)).numpy()
        aug_images.append(rotated)
        aug_labels.append(label)
    images = np.array(aug_images)
    labels = np.array(aug_labels)
    print(f"After augmentation: {len(images)} images, labels: {Counter(labels)}")
    return images, labels