import cv2
from cv2 import dnn_superres
from os import listdir
from os.path import isfile, join

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

image_path = "./images/"
upscale_path = "./upscaled/"
model_path = "EDSR_x3.pb"
model_name = "edsr"
model_times = 3

files = [f for f in listdir(image_path) if isfile(join(image_path, f))]

for x in files:
    print(join(image_path, x))
    image = cv2.imread(join(image_path, x))

    sr.readModel(model_path)
    sr.setModel(model_name, model_times)
    result = sr.upsample(image)
    cv2.imwrite(join(upscale_path, x), result)
