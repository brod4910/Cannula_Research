from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)

img_1 = Image.open("./psf_1_10.png")
img_2 = Image.open("./psf_1_1.png")

img_1_arr = np.asarray(img_1)
img_2_arr = np.asarray(img_2)

print(img_1_arr)
print(img_2_arr)