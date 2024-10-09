import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import radon, iradon
from skimage.filters import threshold_otsu

image_name = 'full'
image_path = f'./images/{image_name}.bmp'

image = imread(image_path, as_gray=True)
thresh = threshold_otsu(image)
binary_image = (image > thresh).astype(np.float64)

original_sum = np.sum(binary_image)

output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))

angles = [180, 160, 140, 120, 100, 80, 60, 40, 30, 20, 10, 5]
rmes = []

for angle in angles:
    print(f"{angle} vetület feldolgozása...")

    theta = np.linspace(0., 180., angle, endpoint=False)

    sinogram = radon(binary_image, theta=theta, circle=True)

    reconstructed_image = iradon(sinogram, theta=theta, circle=True)
    reconstructed_image_binary = (reconstructed_image > 0.5).astype(np.float64)

    output_path = os.path.join(output_dir, f'{image_name}_{angle}.png')
    imsave(output_path, (reconstructed_image_binary * 255).astype(np.uint8))

    rme = np.sum(np.abs(binary_image - reconstructed_image_binary)) / original_sum
    rmes.append(rme)

    print(f"{angle} vetület feldolgozása befejezve.")

plt.plot(angles, rmes, marker='o')
plt.xlabel('Vetületek száma')
plt.ylabel('RME hiba')
plt.title('Rekonstrukciós hiba (RME) a vetületek számának függvényében')
plt.grid(True)

plt.savefig(os.path.join(output_dir, f'{image_name}_rme_plot.png'))
plt.close()
