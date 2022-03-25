import os
import datetime
import numpy as np
from sklearn.cluster import KMeans
import cv2
from imutils import build_montages
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(resnet50.conv1,
                                    resnet50.bn1,
                                    resnet50.relu,
                                    resnet50.maxpool,
                                    resnet50.layer1,
                                    resnet50.layer2,
                                    resnet50.layer3,
                                    resnet50.layer4)

    def forward(self, x):
        x = self.resnet(x)
        return x


print("AOI Defect 图像聚类分析神经网络初始化中... in %s" % datetime.datetime.now())
net = Net().eval()
print("神经网络初始化完成 in %s" % datetime.datetime.now())

image_path = []
all_images = []
images = os.listdir('./image')
print("不良图像获取完成 in %s" % (datetime.datetime.now()))

for image_name in images:
    print(".", end="")
    image_path.append('./image/' + image_name)
print("\n不良图像Resize[224, 224]中... in %s" % datetime.datetime.now())
for path in image_path:
    print(".", end="")
    image = Image.open(path).convert('RGB')
    image = transforms.Resize([224, 244])(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = net(image)
    image = image.reshape(-1, )
    all_images.append(image.detach().numpy())
print("\n不良图像Resize完成 in %s" % datetime.datetime.now())

print("神经网络加载中... in %s" % datetime.datetime.now())
clt = KMeans(n_clusters=2)
print("神经网络加载完成 in %s" % datetime.datetime.now())
print("不良图像自学习中... in %s" % datetime.datetime.now())
clt.fit(all_images)
print("不良图像学习完成，模型生成完成 in %s" % datetime.datetime.now())
labelIDs = np.unique(clt.labels_)

print("分析结果准备中... in %s" % datetime.datetime.now())
for labelID in labelIDs:
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(81, len(idxs)),
                            replace=False)
    show_box = []
    for i in idxs:
        print(".", end="")
        image = cv2.imread(image_path[i])
        image = cv2.resize(image, (96, 96))
        show_box.append(image)
    montage = build_montages(show_box, (96, 96), (9, 9))[0]

    title = "Type {}".format(labelID)
    print("\nType[%s]分析完成 in %s" % (labelID, datetime.datetime.now()))
    cv2.imshow(title, montage)
    cv2.waitKey(0)
