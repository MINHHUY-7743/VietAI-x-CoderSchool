import numpy as np
from matplotlib import pyplot as plt

# Bước 1:
file_path = "full_numpy_bitmap_bicycle.npy"
images = np.load(file_path).astype(np.float32)
print(f"Dataset shape: {images.shape}")

train_images = images[:-10]
test_images = images[-10:]

# Bước 2:
avg_image = np.mean(train_images, axis=0).reshape(28, 28)

# Bước 3:
plt.imshow(avg_image, cmap="gray")
plt.title("Average Image")
plt.show()

# Bước 4:
index = 4
test_image = test_images[index].reshape(-1)
avg_image_flat = avg_image.reshape(-1)
score = np.dot(test_image, avg_image_flat)
print(f"Dot product score for index {index}: {score}")

# Bước 5:
categories = ["alarm clock", "apple", "arm", "bear", "bee", "bicycle", "bird", "book", "brain", "The Eiffel Tower"] # Tên categories
avg_images = []

for category in categories:
    file_path = f"./full_numpy_bitmap_{category}.npy"
    images = np.load(file_path).astype(np.float32)
    avg_images.append(np.mean(images, axis=0).reshape(28, 28))

scores = [np.dot(test_image, avg.reshape(-1)) for avg in avg_images]
print("Scores for all categories:", scores)
print(f"Highest score is for category: {categories[np.argmax(scores)]}")

# Bước 6:
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(avg_images[i], cmap="gray")
    ax.set_title(categories[i])
    ax.axis("off")
plt.tight_layout()
plt.show()

# Dataset shape: (126527, 784)
# Dot product score for index 4: 2979815.5

# Scores for all categories: [np.float32(3110129.5), np.float32(2140212.0),
# np.float32(2082595.6), np.float32(2783016.8), np.float32(3627086.5),
# np.float32(2979891.2), np.float32(2442470.8), np.float32(3293694.8),
# np.float32(3829713.5), np.float32(2394869.5)]

# Highest score is for category: brain