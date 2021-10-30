import os

import h5py
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.preprocessing import image
from numpy import linalg as LA

'''
模型训练:

https://juejin.cn/post/6844903933266100237

'''
class VGGNet:
    def __init__(self):
        self.input_shap = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model_vgg = VGG16(weights=self.weight,
                               input_shape=(self.input_shap[0], self.input_shap[1], self.input_shap[2]),
                               pooling=self.pooling, include_top=False)
        self.model_vgg.predict(np.zeros((1, 224, 224, 3)))

    # 提取vgg16最后一层卷积特征
    def vgg_extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shap[0], self.input_shap[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_vgg(img)
        feat = self.model_vgg.predict(img)

        norm_deat = feat[0] / LA.norm(feat[0])
        return norm_deat


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]


if __name__ == "__main__":
    database = 'database'
    index = 'models/vgg_featureCNN.h5'
    img_list = get_imlist(database)

    print("--------------------------------------------------------------")
    print("                 feature extraction starts                    ")
    print("--------------------------------------------------------------")

    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        # 修改此处改变提取特征的网络
        norm_feat = model.vgg_extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image NO. %d , %d images in total" % ((i + 1), len(img_list)))

    feats = np.array(feats)
    output = index
    print("--------------------------------------------------------------")
    print("         writing feature extraction results ...               ")
    print("--------------------------------------------------------------")

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()
