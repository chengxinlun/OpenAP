import matplotlib.pyplot as plt
import numpy as np
from OpenAP.frequency import msLinearTransform
from skimage import data, color


# Generate image
img = color.rgb2lab(data.astronaut(), illuminant='D50')
lum = img[:, :, 0] / 100.0
# Add noise
noise = np.random.normal(0.0, 0.01, lum.shape)
lum_noisy = lum + noise
# Denoise
lum_mask = np.ones(lum_noisy.shape, dtype=np.float64)
lum_denoise = msLinearTransform(lum_noisy, lum_mask, 5,
                                np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
                                [5, 5, 5, 5, 5], [1.0, 1.0, 1.0, 1.0, 1.0])
# Plot
fig = plt.figure(figsize=(18, 6))
ax1 = plt.subplot(131)
ax1.imshow(lum)
ax2 = plt.subplot(132)
ax2.imshow(lum_noisy)
ax3 = plt.subplot(133)
ax3.imshow(lum_denoise)
plt.show()
