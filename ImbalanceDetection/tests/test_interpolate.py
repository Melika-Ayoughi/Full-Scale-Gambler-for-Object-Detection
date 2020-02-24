
from detectron2.data import detection_utils as utils
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

filename = "/home/ayoughi/detectron2/datasets/coco/train2017/000000458139.jpg"
filename1 = "/home/ayoughi/detectron2/datasets/coco/train2017/000000207349.jpg"
image1 = utils.read_image(filename, format='BGR')
image1 = image1.transpose(2, 0, 1)
image1 = np.expand_dims(image1, axis=0)
input_images = torch.as_tensor(image1.astype("float32"))

image2 = Image.open(filename1)
# image2 = image2.convert("BGR")
image2 = image2.resize((640, 640))
image2 = np.asarray(image2)
image2 = image2.transpose(2, 0, 1)
image2 = np.expand_dims(image2, axis=0)
input_images2 = torch.as_tensor(image2.astype("float32"))

input_images = torch.cat((input_images, input_images2), dim=0)


print(input_images.shape)
input_images = F.interpolate(input_images, scale_factor=1 /16, mode='bilinear') # todo: stride depends on feature map layer
c= input_images.numpy()[0, :,:, :].transpose(1,2,0)
c=c.astype(np.int)
print(np.max(c), np.min(c))
# plt.imshow(image1)
plt.imshow(np.clip(c, 0, 255))
plt.show()
