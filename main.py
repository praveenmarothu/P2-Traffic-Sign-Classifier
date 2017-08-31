import glob
import os
import cv2
from model import Model


imgs=[]
labels=[]

lbls={"1x.png":11,"2x.png":1,"3x.png":12,"5x.png":38,"6x.png":34,"8x.png":18,"9x.png":25,"10x.png":3 }

for i, fpath in enumerate(glob.glob('./predict_images/*x.png')):
    fname=os.path.basename(fpath)
    imgs.append(cv2.imread(fpath))
    labels.append(lbls[fname])

print(labels)

model = Model()
model.predict(imgs,labels)
