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

output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    os.remove(file_path)

angles = [180, 160, 140, 120, 100, 80, 60, 40, 30, 20, 10, 5]
rmes = []

original_sum = np.sum(binary_image)

for angle in angles:
    print(f"{angle} vetület feldolgozása...")
    theta = np.linspace(0., 180., angle, endpoint=False)

    # Radon-transzformáció
    sinogram = radon(binary_image, theta=theta, circle=True)

    # Rekonstrukció (inverz Radon)
    reconstructed_image = iradon(sinogram, theta=theta, circle=True)

    # Binarizálás a rekonstrukcióhoz
    reconstructed_image_binary = (reconstructed_image > 0.5).astype(np.float64)

    # Rekonstrukciós kép mentése (konvertálás 0-255 közötti értékekre)
    output_path = os.path.join(output_dir, f'{image_name}_{angle}.png')
    imsave(output_path, (reconstructed_image_binary * 255).astype(np.uint8))  # Konvertálás és mentés
    print(f"{angle} vetület feldolgozása befejezve, kép mentve: {output_path}")

    # RME hiba kiszámítása
    rme = np.sum(np.abs(binary_image - reconstructed_image_binary)) / original_sum
    rmes.append(rme)

plt.plot(angles, rmes, marker='o')
plt.xlabel('Vetületek száma')
plt.ylabel('RME hiba')
plt.title('Rekonstrukciós hiba (RME) a vetületek számának függvényében')
plt.grid(True)

plt.savefig(os.path.join(output_dir, f'{image_name}_rme_plot.png'))
plt.close()
