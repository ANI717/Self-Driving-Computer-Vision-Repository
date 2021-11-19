import onnxruntime as ort
import numpy as np
import cv2


def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img




IFILE = '../dataset/images/03_12_2020_0/output_0001/i0000049_s09_m07.jpg'
SHAPE = (1, 3, 75, 75)


frame = cv2.imread(IFILE)
frame = cv2.resize(frame, dsize=SHAPE[-2:])
frame = normalize(frame, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0)
frame = cv2.dnn.blobFromImage(frame)


ort_session = ort.InferenceSession("checkpoint/epoch_128.onnx")
outputs = ort_session.run(None, {"input.1": frame.astype(np.float32)},)
print(outputs[0][0][0])