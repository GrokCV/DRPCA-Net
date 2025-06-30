import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import scipy.io as scio
import os
import time

from models import get_model


dataset_name = 'NUDT-SIRST' # NUDT-SIRST IRSTD-1k sirst_aug SIRSTv1

net = get_model('Drpcanet')

file_path =  fr'./datasets/{dataset_name}'

if dataset_name == 'NUDT-SIRST':
    pkl_file = './checkpoints/NUDTIRSTD_mIoU_94.16.pkl'
elif dataset_name == 'IRSTD-1k':
    pkl_file = './checkpoints/IRSTD1K_mIoU_64.14.pkl'
elif dataset_name =='sirst_aug':
    pkl_file = './checkpoints/SIRSTAUG_mIoU_76.50.pkl'
elif dataset_name == 'SIRSTv1':
    pkl_file = './checkpoints/SIRSTv1_mIoU_75.52.pkl'
else:
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")

checkpoint = torch.load(pkl_file, map_location=torch.device('cuda:0'))
net.load_state_dict(checkpoint)
net.eval()

imgDir = f"./result/{dataset_name}/img1/"
if not os.path.exists(imgDir):
    os.makedirs(imgDir)
matDir = f"./result/{dataset_name}/mat1/"
if not os.path.exists(matDir):
    os.makedirs(matDir)
number = 0

if dataset_name == 'SIRSTv1':
    txt_name = "test_v1.txt"
    txt_path = fr'E:\datasets\SIRSTdevkit-master\Splits\{txt_name}'
    file_path_base = fr'E:\datasets\SIRSTdevkit-master\PNGImages'
    with open(os.path.join(txt_path), "r") as f:
        file_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
else:
    file_list = os.listdir(file_path)

for filename in file_list:
    number += 1
    if dataset_name == 'SIRSTv1':
        filename = filename + '.png'
        img_gray = cv2.imread(file_path_base + '/' + filename, 0)
    else:
        img_gray = cv2.imread(file_path + '/' + filename, 0)
    img_gray = cv2.resize(img_gray, [256, 256],interpolation=cv2.INTER_LINEAR)
    img = img_gray.reshape(1, 1, 256, 256) / 255.
    img = torch.from_numpy(img).type(torch.FloatTensor)
    name = os.path.splitext(filename)[0]
    matname = name+'.mat'

    with torch.no_grad():
        start = time.time()
        D, T = net(img)
        end = time.time()
        total = end - start
        T = F.sigmoid(T)
        T = T.detach().numpy().squeeze()
        T[T < 0] = 0
        Tout = T * 255
        print(number)

    cv2.imwrite(imgDir + filename, Tout)
    scio.savemat(matDir + matname, {'T': T})