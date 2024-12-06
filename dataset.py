from __future__ import print_function
import copy
import os
import time
import math
import random
import torch
import numpy as np
from colorama import Fore, Style
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader

from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import scipy.io as sio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    print(f"Folder '{folder_path}' already create !!!")


def scale_to_01_range(x):
    a, b = x.shape
    x = x.reshape(a * b, -1)
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    result = starts_from_zero / value_range
    result = result.reshape(a, b)
    return result


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def add(X, windowSize):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    spectralData = np.zeros((X.shape[0] * X.shape[1], 1, 1, X.shape[2]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            spectral = zeroPaddedX[r, c]
            spectralData[patchIndex, :, :, :] = spectral
            patchIndex = patchIndex + 1
    return patchesData, spectralData


class DPNCSSTD_dataset(torch.utils.data.Dataset):
    def __init__(self, data1, data2, labels, run):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels
        self.run = run

    def __getitem__(self, index):
        if self.run == 'train':
            img1 = torch.Tensor(self.data1[index]).unsqueeze(0)
            img2 = torch.Tensor(self.data2[index]).unsqueeze(0)
            labels = self.labels[index]
            return img1, img2, labels
        if self.run == 'test':
            img1 = torch.Tensor(self.data1[index]).unsqueeze(0)
            img2 = torch.Tensor(self.data2[index]).unsqueeze(0)
            return img1, img2

    def __len__(self):
        return len(self.data1)


# ——————————————  Data Augmentation  ——————————————

def augment_area(x):
    aug = x.copy()
    np.random.seed(int(time.time()))
    noise = np.ones((aug.shape[0], aug.shape[1]))
    ran1 = np.random.randint(0, aug.shape[0] - 1)
    ran2 = np.random.randint(ran1 + 1, aug.shape[0])
    ran3 = np.random.randint(0, aug.shape[0] - 1)
    ran4 = np.random.randint(ran3 + 1, aug.shape[0])
    noise[ran1:ran2, ran3:ran4] = 0
    noise[aug.shape[0] // 2, aug.shape[1] // 2] = 1  # Ensure that the noise is not in the center position
    noise = noise[:, :, np.newaxis]
    noise = np.concatenate([noise] * aug.shape[2], 2)
    aug = aug * noise
    # aug=aug.transpose(2,0,1)
    Aug = torch.from_numpy(aug.copy())

    return Aug


def augment_point(x):
    aug = x.copy()

    np.random.seed(int(time.time()))
    noise = np.random.randint(0, 2, (aug.shape[0], aug.shape[1]))
    noise[aug.shape[0] // 2, aug.shape[1] // 2] = 1   # Ensure that the noise is not in the center position
    noise = noise[:, :, np.newaxis]
    noise = np.concatenate([noise] * aug.shape[2], 2)
    aug = aug * noise
    Aug = torch.from_numpy(aug.copy())
    return Aug


def augment_unite_randomflip(x):
    num = random.randint(0, 1)

    if num == 0:
        aug1 = np.fliplr(x)
        Aug1 = augment_point(aug1)
        aug2 = np.flipud(x)
        Aug2 = augment_area(aug2)

    elif num == 1:
        aug1 = np.flipud(x)
        Aug1 = augment_point(aug1)
        aug2 = np.fliplr(x)
        Aug2 = augment_area(aug2)

    return Aug1, Aug2


def augment_spectral(y):
    num = random.randint(0, 1)
    if num == 0:
        aug1 = y
        aug2 = y

    elif num == 1:
        aug1 = y
        aug2 = y

    return aug1, aug2


# ——————————————  Dataset  ——————————————
class HSIDataset_train_unite(Dataset):
    def __init__(self, X, Y, aug1, aug2):
        self.X_w = copy.deepcopy(X)
        self.X_s = copy.deepcopy(X)
        self.Y_w = Y
        self.Y_s = Y
        for i in range(self.X_w.shape[0]):
            self.X_w[i], self.X_s[i] = aug1(self.X_w[i])
        self.Y_w, self.Y_s = aug2(self.Y_w)

    def __len__(self):
        return self.X_w.shape[0]

    def __getitem__(self, index):
        X = torch.Tensor(self.X_w[index]).unsqueeze(dim=0), torch.Tensor(self.X_s[index]).unsqueeze(dim=0)
        Y = torch.Tensor(self.Y_w[index]).unsqueeze(dim=0), torch.Tensor(self.Y_s[index]).unsqueeze(dim=0)

        return X, Y


def loaddata(image_file, label_file, process):
    print("===> Getting dataset......")
    if process == 'DPNCSSTD_pretrain':
        img = sio.loadmat('./data/' + image_file + '.mat')['data'].astype(np.float32)
        sp = 1
        label = 1
    else:
        image_mat = sio.loadmat('./data/' + image_file + '.mat')
        label_mat = sio.loadmat('./data/' + label_file + '.mat')

        try:
            img = image_mat['data'].astype(np.float32)
            sp = image_mat['TargetPrioriSpectra'].T.astype(np.float32)
            label = label_mat['map']
        except Exception as err:
            img = image_mat['data'].astype(np.float32)
            try:
                sp = image_mat['TargetPrioriSpectra'].T.astype(np.float32)
                label = 1
            except Exception as err:
                sp = 1
                label = label_mat['map']
    print("===> Already loading {} dataset......".format(str(process)))

    return img, label, sp


# ——————————————  Get training samples ——————————————
def get_patch_data(finetune_data):
    windowSize = 5
    x, y = add(finetune_data, windowSize)
    print('valid data size : ', x.shape)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

def Hyperpixel_segmentation(img, number, show_figure):
    h, w, band = img.shape
    image = img.reshape(h * w, band)
    image = scale_to_01_range(image)
    image = image.reshape(h, w, band)
    rgb = image[:, :, 0:3]
    rgb[:, :, 0] = image[:, :, 44]
    rgb[:, :, 1] = image[:, :, 32]
    rgb[:, :, 2] = image[:, :, 15]
    # 44 32 15
    rgb = img_as_float(rgb)
    segments = slic(rgb, n_segments=int(number), sigma=2)
    if show_figure == 1:
        fig = plt.figure("Superpixels -- %d segments" % 100)
        plt.subplot(131)
        plt.title('image')
        plt.imshow(rgb)
        plt.subplot(132)
        plt.title('segments')
        plt.imshow(segments)
        plt.subplot(133)
        plt.title('image and segments')
        plt.imshow(mark_boundaries(rgb, segments))
        plt.show()
        plt.imshow(mark_boundaries(rgb, segments))
        plt.axis('off'), plt.show()
        plt.imshow(segments)
        plt.axis('off'), plt.show()

    return segments


def get_center_geolocation(polygon):
    area = 0.0
    center_x, center_y = 0.0, 0.0

    a = len(polygon)
    for i in range(a):
        x = polygon[i, 0]
        y = polygon[i, 1]

        if i == 0:
            x1 = polygon[-1][0]
            y1 = polygon[-1][1]

        else:
            x1 = polygon[i - 1][0]
            y1 = polygon[i - 1][1]

        fg = np.abs((x * y1 - y * x1) / 2.0)

        area += fg
        center_x += fg * (x + x1) / 3.0
        center_y += fg * (y + y1) / 3.0

    xy = np.zeros((1, 2)) / 1.0
    xy[0, 0] = int(center_x / area)
    xy[0, 1] = int(center_y / area)

    return xy


def get_center_pixel(img, number, show_figure):
    min_diff = 10
    for input_value in range(15, 25):
        segments = Hyperpixel_segmentation(img, input_value, show_figure)
        class_num = len(np.unique(segments))
        diff = abs(class_num - number)
        if diff < min_diff:
            min_diff = diff
            closest_input = input_value
    final_segments = Hyperpixel_segmentation(img, closest_input, show_figure)
    final_class_num = len(np.unique(final_segments))

    center_s = np.zeros((1, 2))
    center_s = center_s.astype(np.float32)
    center_xy = np.zeros((1, 2))
    for c in range(final_class_num):
        s_xy = np.zeros((1, 2))
        a = 0
        for i in range(final_segments.shape[0]):
            for j in range(final_segments.shape[1]):
                if final_segments[i, j] == c + 1:
                    center_s[a, 0] = i
                    center_s[a, 1] = j
                    s_xy = np.concatenate((s_xy, center_s))
        xy = get_center_geolocation(s_xy[1:, :])
        center_xy = np.concatenate((center_xy, xy))
    center_xy = center_xy[1:, :]
    return center_xy


def sam(data1, data2):
    radian = math.acos(np.dot(data1, data2) / (np.linalg.norm(data1) * np.linalg.norm(data2)))
    radian = radian * 180 / math.pi
    return radian


def delect_similar_pixel(img, sp, number, show_figure, method):
    if method == 'Hyperpixel_segmentation':
        xy = get_center_pixel(img, number, show_figure)

    new_xy = np.zeros((1, 2))
    ab = np.zeros((1, 2))
    for i in range(xy.shape[0]):
        spectral = img[int(xy[i, 0]), int(xy[i, 1]), :]
        a = 0
        for j in range(sp.shape[0]):
            radian = sam(spectral, sp[j, :])
            if 350 > radian > 5:
                a = a + 1
        if a == sp.shape[0]:
            ab[0, 0] = xy[i, 0]
            ab[0, 1] = xy[i, 1]
            new_xy = np.concatenate((new_xy, ab))
    new_xy = new_xy[1:, :]
    return new_xy


def follow_pixel_get_patch(img, position, windowSize):
    h, w, band = img.shape
    margin = int((windowSize - 1) / 2)

    def padWithZeros(X, margin):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    new_img = padWithZeros(img, margin)
    patchIndex = 0
    patchesData = np.zeros((h * w, windowSize, windowSize, band))
    spectralData = np.zeros((h * w, 1, 1, band))
    for i in range(position.shape[0]):
        x = int(position[i, 0] + margin)
        y = int(position[i, 1] + margin)
        patch = new_img[x - margin:x + margin + 1, y - margin:y + margin + 1, :]
        spectral = new_img[x, y, :]
        patchesData[patchIndex, :, :, :] = patch
        spectralData[patchIndex, :, :, :] = spectral
        patchIndex = patchIndex + 1

    patchesData = patchesData[0:patchIndex, :, :, :]
    spectralData = spectralData[0:patchIndex, :, :, :]

    return patchesData, spectralData


def get_target_samples(img, sp, windowSize, number):
    [h, w, band] = img.shape
    margin = int((windowSize - 1) / 2)
    origin_data = img.reshape(h * w, band)
    detect_data = CEM(img, sp)
    M2 = np.hstack((origin_data, detect_data))

    # background spectra
    s = M2[np.lexsort(-M2.T)]
    cem_value = s[:, band]
    t_threshold = cem_value[number]

    cem_result = detect_data.reshape(h, w)

    def padWithZeros(X):
        margin = 1
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    new_img = padWithZeros(img)
    patchIndex = 0
    patchesData = np.zeros((h * w, windowSize, windowSize, band))
    spectralData = np.zeros((h * w, 1, 1, band))
    for i in range(margin, new_img.shape[0] - margin):
        for j in range(margin, new_img.shape[1] - margin):
            if cem_result[i - margin, j - margin] > t_threshold:
                t_patch = new_img[i - margin:i + margin + 1, j - margin:j + margin + 1, :]
                patchesData[patchIndex, :, :, :] = t_patch
                spectral = new_img[i, j]
                spectralData[patchIndex, :, :, :] = spectral
                patchIndex = patchIndex + 1

    patchesData = patchesData[0:patchIndex, :, :, :]
    spectralData = spectralData[0:patchIndex, :, :, :]

    return patchesData, spectralData


def get_samples_all(img, sp, windowSize, method, threshold, show_figure, number):
    position = delect_similar_pixel(img, sp, number, show_figure, method)
    b_patch, b_spectral = follow_pixel_get_patch(img, position, windowSize)  # background
    print('b', b_patch.shape, b_spectral.shape)
    t_patch, t_spectral = get_target_samples(img, sp, windowSize, threshold)  # target

    print('t', t_patch.shape, t_spectral.shape)

    patch = np.concatenate((b_patch, t_patch))
    spectral = np.concatenate((b_spectral, t_spectral))
    labels = np.concatenate((np.zeros(b_patch.shape[0]), np.ones(t_patch.shape[0])))
    labels = np.expand_dims(labels, 1)
    return patch, spectral, labels


def CEM(img, sp):
    [h, w, band] = img.shape
    N = h * w
    M = img.reshape(N, band).T

    M = M.astype('float32')

    target = np.mean(sp.T, axis=1)

    R = np.dot(M, M.T) / N
    R_v = np.linalg.inv(R)
    tmp = np.dot(np.dot(target.T, R_v), target)

    cem = np.zeros((N, 1))
    for k in range(0, N):
        cem[k, :] = np.dot(np.dot(target.T, R_v), M[:, k]) / tmp

    return cem


# ——————————————  ROC curve  ——————————————
def plot_roc_curve(prediction, label, if_show):
    fpr, tpr, threshold = roc_curve(label.ravel(), prediction.ravel())
    AUC = auc(fpr, tpr)

    if if_show == 0:
        a = 1
    if if_show == 1:
        plt.subplots(num='ROC curve')
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % AUC)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()
    fpr = fpr[1:]
    tpr = tpr[1:]
    threshold = threshold[1:]
    auc1 = round(metrics.auc(fpr, tpr), 4)
    auc2 = round(metrics.auc(threshold, fpr), 4)
    auc3 = round(metrics.auc(threshold, tpr), 4)

    auc4 = round(auc1 + auc3 - auc2, 4)
    auc5 = round(auc1 + auc3, 4)
    auc6 = round(auc1 - auc2, 4)
    auc7 = round(auc3 - auc2, 4)
    auc8 = round(auc3 - auc2 + 1, 4)
    auc9 = round(auc3 / auc2, 4)
    print('————————————————————————————————————————————————————')
    print(
        f'[AUC ]:{Fore.RED}{auc1:.4f}{Style.RESET_ALL}  [F  ]:{Fore.RED}{auc2:.4f}{Style.RESET_ALL}  [D   ]:{Fore.RED}{auc3:.4f}{Style.RESET_ALL}')
    print(
        f'[OD  ]:{Fore.RED}{auc4:.4f}{Style.RESET_ALL}  [TD ]:{Fore.RED}{auc5:.4f}{Style.RESET_ALL}  [BS  ]:{Fore.RED}{auc6:.4f}{Style.RESET_ALL}')
    print(
        f'[TDBS]:{Fore.RED}{auc7:.4f}{Style.RESET_ALL}  [ODP]:{Fore.RED}{auc8:.4f}{Style.RESET_ALL}  [SNPR]:{Fore.RED}{auc9:.4f}{Style.RESET_ALL}')

    print('————————————————————————————————————————————————————')
    return auc1, auc2, auc3
