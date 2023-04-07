import numpy as np
import os
from lr_utils import load_dataset
from PIL import Image
import cv2

if not os.path.isdir('./4/datasets'):
    print("Dataset not found!")
    exit(1)

# 读入数据
trainSet_x_orig, trainSet_y, testSet_x_orig, testSet_y, classes = load_dataset()

# 训练集正/反例个数：
trainNum_Neg = 0
trainNum_Pos = 0

# 训练集路径创建：
if not os.path.isdir('./4/datasets/TrainImages'):
    os.mkdir('./4/datasets/TrainImages')

# 将trainSet_x_orig中的数据转换为jpg图片保存在训练集目录下：
for i in range(trainSet_y.shape[1]):
    if trainSet_y[0][i] == 0:
        trainNum_Neg += 1
        im = Image.fromarray(np.uint8(trainSet_x_orig[i]))
        im.save('./4/datasets/TrainImages/neg-%d.jpg' %trainNum_Neg)
    else:
        trainNum_Pos += 1
        im = Image.fromarray(np.uint8(trainSet_x_orig[i]))
        im.save('./4/datasets/TrainImages/Pos-%d.jpg' %trainNum_Pos)

# 测试集同上：
testNum_Neg = 0
testNum_Pos = 0

if not os.path.isdir('./4/datasets/TestImages'):
    os.mkdir('./4/datasets/TestImages')

for i in range(testSet_y.shape[1]):
    if testSet_y[0][i] == 0:
        testNum_Neg += 1
        im = Image.fromarray(np.uint8(testSet_x_orig[i]))
        im.save('./4/datasets/TestImages/neg-%d.jpg' %testNum_Neg)
    else:
        testNum_Pos += 1
        im = Image.fromarray(np.uint8(testSet_x_orig[i]))
        im.save('./4/datasets/TestImages/Pos-%d.jpg' %testNum_Pos)


# 补全训练集：
if trainNum_Neg != trainNum_Pos:
    diff = trainNum_Neg - trainNum_Pos
    for i in range(diff):
        im = cv2.imread('./4/datasets/TrainImages/Pos-%d.jpg' % (i+1))
        im_f = cv2.flip(im, 1)
        trainNum_Pos += 1
        cv2.imwrite('./4/datasets/TrainImages/Pos-%d.jpg' %trainNum_Pos, im_f)