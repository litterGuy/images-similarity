import h5py
import matplotlib.image as mping
import matplotlib.pyplot as plt
import numpy as np

from vgg.vgg_train import VGGNet

'''
图片相似度搜索
'''
query = "test.png"
index = "models/vgg_featureCNN.h5"
result = "database"

h5f = h5py.File(index, 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

print("-----------------------------------------------------")
print("                 searching starts                    ")
print("-----------------------------------------------------")

# read and show query image
queryImg = mping.imread(query)
plt.title("Query image")
plt.imshow(queryImg)
plt.show()

# init VGGNet16 model
model = VGGNet()

# 修改此处改变提取特征的网络
queryVec = model.vgg_extract_feat(query)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

# number of top retrieved images to show
maxres = 3
imlist = []
for i, index in enumerate(rank_ID[0:maxres]):
    imlist.append(imgNames[index])
    print("image names : " + str(imgNames[index]) + " scores：%f " % rank_score[i])
print("top %d images in order are: " % maxres, imlist)

for i, im in enumerate(imlist):
    image = mping.imread(result + "/" + str(im, 'utf-8'))
    plt.title("search output %d" % (i + 1))
    plt.imshow(image)
    plt.show()
