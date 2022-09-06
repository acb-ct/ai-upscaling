import cv2
from cv2 import dnn_superres
from os import listdir
from os.path import isfile, join
import time

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Define folders and models
image_path = "./images/"
upscale_path = "./upscaled/"
model_path = "EDSR_x3.pb"
model_name = "edsr"
model_times = 3
gpu = True


files = [f for f in listdir(image_path) if isfile(join(image_path, f))]

for x in files:

    print(join(image_path, x))
    image = cv2.imread(join(image_path, x))
    sr.readModel(model_path)

    # Set CUDA backend and target to enable GPU inference
    if gpu:
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        use = "gpu"
    else:
        use = "cpu"

    # Set a start time to calculate the scaling time
    start = time.time()

    sr.setModel(model_name, model_times)
    result = sr.upsample(image)
    cv2.imwrite(join(upscale_path, x), result)

    # Set a ending time to calculate the scaling time
    end = time.time()

    print(f'{x} upscaled {model_times} times with {model_name} in {end - start}, using the {use}.')
