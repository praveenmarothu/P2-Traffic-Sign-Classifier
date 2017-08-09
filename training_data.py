import pickle
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform,AffineTransform
import random
import cv2
import os
from sklearn.utils import shuffle

class TrainingData(object):

    data=None

    def __init__(self):
        self.x_train , self.y_train = None,None
        self.x_test , self.y_test = None,None
        self.x_valid , self.y_valid = None,None

        self.load()
        self.image_grayscale_normalize()
        self.plot_histogram("output_images/loaded_train_histogram.png")
        self.augment_data()
        self.plot_histogram("output_images/augmented_train_histogram.png")
        self.one_hot_encode_labels()
        self.reshape_features()
        self.split_train_valid()


    def load(self):
        with open('data/train.p', mode='rb') as f:
            data=pickle.load(f)
            self.x_train,self.y_train = data["features"],data["labels"]

        with open('data/test.p','rb') as f:
            data=pickle.load(f)
            self.x_test,self.y_test = data["features"],data["labels"]


    def split_train_valid(self):
        self.x_train,self.x_valid,self.y_train,self.y_valid=train_test_split(self.x_train,self.y_train,test_size=0.05,random_state=87878,stratify=self.y_train)
        self.x_train,self.y_train=shuffle(self.x_train,self.y_train)

    def print_counts(self):
        pass

    def plot_histogram(self,f_name):
        values, counts = np.unique(self.y_train, return_counts=True)
        plt.figure()
        plt.bar(values, counts)
        plt.ylabel('Num of Examples')
        plt.xlabel('Class')
        plt.title('Num Per Class')
        plt.savefig(f_name)
        plt.close()

    def augment_data(self):
        data = {}
        for cls in np.unique(self.y_train): data[cls] = []
        for img,cls in zip(self.x_train,self.y_train): data[cls].append(img)

        _x_train=[]
        _y_train=[]

        for cls,imgs in data.items():
            c_length=len(imgs)
            _x_train.extend(imgs)
            _y_train.extend([cls]*c_length)

            idx,counter,target=0,c_length,1200

            while counter<=target:
                _x_train.append(self.image_random_transform(imgs[idx]))
                _y_train.append(cls)

                counter+=1
                idx+=1 if(idx<c_length-1) else 0

        self.x_train=np.array(_x_train)
        self.y_train=np.array(_y_train)

    def image_grayscale_normalize(self):

        _x_train=[]
        _x_test=[]

        for img in self.x_train:
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img= (np.float32(img)-np.min(img))/( np.max(img) - np.min(img) )
            _x_train.append(img)

        for img in self.x_test:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img= (np.float32(img)-np.min(img))/( np.max(img) - np.min(img) )
            _x_test.append(img)

        self.x_train=np.array(_x_train)
        self.x_test=np.array(_x_test)


    def image_random_transform(self,img):
        image_size = img.shape[0]
        d = image_size * 0.2
        tl_top,tl_left,bl_bottom,bl_left,tr_top,tr_right,br_bottom,br_right = np.random.uniform(-d, d,size=8)   # Bottom right corner, right margin
        aft=  AffineTransform(scale=(1, 1/1.2))
        img= warp(img, aft,output_shape=(image_size, image_size), order = 1, mode = 'edge')
        transform = ProjectiveTransform()
        transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))

        img = warp(img, transform, output_shape=(image_size, image_size), order = 1, mode = 'edge')
        return img

    def one_hot_encode_labels(self):
        n_classes=len(np.unique(self.y_train))
        self.y_train=np.eye(n_classes)[self.y_train]
        n_classes=len(np.unique(self.y_test))
        self.y_test=np.eye(n_classes)[self.y_test]


    def reshape_features(self):
        self.x_train=np.reshape(self.x_train,(-1,32,32,1) )
        self.x_test=np.reshape(self.x_test,(-1,32,32,1) )

    @classmethod
    def get_data(cls):
        if cls.data is not None:
            return cls.data
        elif os.path.isfile("pickled/training_data.p"):
            with open('pickled/training_data.p', 'rb') as f:
                cls.data = pickle.load(f)
        else:
            cls.data=TrainingData()
            with open('pickled/training_data.p', 'wb') as f:
                pickle.dump(cls.data,f)

        return cls.data

if __name__ == "__main__":
    td = TrainingData.get_data()
    print(td.x_train.shape)
    print(td.y_train.shape)



