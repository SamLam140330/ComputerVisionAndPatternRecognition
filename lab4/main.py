import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern

image = cv2.imread('./lab4data/face.jpg')

print("Showing local binary pattern map")
radius = 2
n_points = 8
grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lbp = local_binary_pattern(grayImg, n_points, radius, "default")
plt.imshow(lbp, 'gray')
plt.show()

print("Showing histogram of local binary pattern")
n_bins = int(lbp.max() + 1)
hist, bins = np.histogram(lbp, n_bins, (0, n_bins), density=True)
fig, ax = plt.subplots()
ax.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")
plt.show()

print("Showing Principal Components Analysis")
mean = [20, 20]
cov = [[5, 0], [25, 25]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.style.use('ggplot')
plt.plot(x, y, 'o', 1)
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.show()

print("Showing Eigenvectors and eigenvalues of data")
X = np.vstack((x, y)).T
mean, eigenvectors, eigenvalues = cv2.PCACompute2(X, np.array([]))
plt.plot(x, y, 'o', zorder=1)
plt.quiver([mean[0, 0], mean[0, 0]], [mean[0, 1], mean[0, 1]], eigenvectors[:, 0], eigenvectors[:, 1], zorder=3,
           scale=0.2, units='xy')
plt.text(mean[0, 0] + 5 * eigenvectors[0, 0], mean[0, 1] + 5 * eigenvectors[0, 1], str(int(eigenvalues[0].item())),
         zorder=5, fontsize=16, bbox=dict(facecolor='white', alpha=0.6))
plt.text(mean[0, 0] + 7 * eigenvectors[1, 0], mean[0, 1] + 4 * eigenvectors[1, 1], str(int(eigenvalues[1].item())),
         zorder=5, fontsize=16, bbox=dict(facecolor='white', alpha=0.6))
plt.axis([0, 40, 0, 40])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()

print("Showing high dimensional data to low dimensional data")
np.random.seed(5)
Dim = 20
mean = np.random.rand(1, Dim).squeeze()
cov = np.random.rand(Dim, Dim)
cov = cov + cov.T
original_data = np.random.multivariate_normal(mean, cov, 1000)
mean, eigenvectors, eigenvalues = cv2.PCACompute2(original_data, np.array([]))
lowDim_data = cv2.PCAProject(original_data, mean, eigenvectors)
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.plot(lowDim_data[:, 0], lowDim_data[:, 1], 'o')
plt.xlabel('first principal component')
plt.ylabel('second principal component')
plt.axis([-20, 20, -10, 10])
plt.show()
