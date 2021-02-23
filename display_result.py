import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

## celeba: Black_Hair | Blond_Hair | Brown_Hair | Male | Young

dir_result = './results/stargan/celeba/images'
# dir_result = './results/stargan/rafd/images'
lst_result = os.listdir(dir_result)

nx = 128
ny = 128
nch = 3

n = 8
m = 1 + 5
# m = 1 + 8

n_id = np.arange(len(lst_result)//m)
np.random.shuffle(n_id)
img = torch.zeros((n*m, ny, nx, nch))

for i in range(m):
    for j in range(n):
        p = i + m*n_id[j]
        q = n * i + j

        img[q, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_result[p]))[:, :, :nch])

img = img.permute((0, 3, 1, 2))

plt.figure(figsize=(n, m))
plt.axis("off")
# plt.title("Generated Images")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1, 2, 0)))

plt.show()

## RaFD: angry | contemptuous | disgusted | fearful | happy | neutral | sad | surprised

# dir_result = './results/stargan/celeba/images'
dir_result = './results/stargan/rafd/images'
lst_result = os.listdir(dir_result)

nx = 128
ny = 128
nch = 3

n = 8
# m = 1 + 5
m = 1 + 8

n_id = np.arange(len(lst_result)//m)
np.random.shuffle(n_id)
img = torch.zeros((n*m, ny, nx, nch))

for i in range(m):
    for j in range(n):
        p = i + m*n_id[j]
        q = n * i + j

        img[q, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_result[p]))[:, :, :nch])

img = img.permute((0, 3, 1, 2))

plt.figure(figsize=(n, m))
plt.axis("off")
# plt.title("Generated Images")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1, 2, 0)))

plt.show()

