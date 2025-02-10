from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
# dataset = CIFAR10(root="my-cifar", download=True)

# image, label = dataset[100]
# print(image.show())
# print(dataset.classes)
# print(label)

cifar10 = datasets.CIFAR10(root="my-cifar", train=True,  transform=transform)

images = np.array([np.array(img[0]) for img in cifar10])
# print(images.shape[0])
images = images.reshape(images.shape[0], -1)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(images)

tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=300)
tsne_result = tsne.fit_transform(images[:2000])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], s=5, alpha=0.7)
ax.set_title('CIFAR-10 Visualization with t-SNE')
plt.show()