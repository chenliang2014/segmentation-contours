import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2

def decode_segmaps(image,label_colors,nc=21):
	
	r = np.zeros_like(image).astype(np.uint8)
	g = np.zeros_like(image).astype(np.uint8)
	b = np.zeros_like(image).astype(np.uint8)
	
	for i in range(nc):
		pos = image==i
		r[pos]=label_colors[i,0]
		g[pos]=label_colors[i,1]
		b[pos]=label_colors[i,2]
		
	rgbimg = np.stack((r,g,b),axis=-1)
	
	return rgbimg
	
label_colors = np.array([(0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),(0,128,128),(128,128,128),(64,0,0),
						(192,0,0),(64,128,0),(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128),(0,64,0),(128,64,0),\
						(0,192,0),(128,192,0),(0,64,128)])


print(len(label_colors))

model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)

model.eval()

img_file = sys.argv[1]

img = Image.open(img_file)
img_draw = cv2.imread(img_file)
trans = T.Compose([T.ToTensor(),T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
img_t = trans(img).unsqueeze(0)
output = model(img_t)

print("output shape:", output['out'].shape)

outputarg = torch.argmax(output['out'].squeeze(),dim=0).numpy().astype(np.uint8)
img_bin = outputarg.copy()
img_bin[img_bin>0]=128
contours,_ = cv2.findContours(img_bin,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)



seg_img = decode_segmaps(outputarg,label_colors)
#print(seg_img.shape)
Image.fromarray(seg_img).save("seg.jpg")

cv2.drawContours(seg_img,contours,-1,(255,0,255),3)
cv2.imwrite("cont.jpg",seg_img)