import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image

#这是一个数据集类，用于加载数据
class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,  # weak augmentation policy
        ops_strong=None,    # strong augmentation policy
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        #如果提供了弱增强操作，就需要同时提供强增强操作
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy" 

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            # # 计算要选择的数据数量（10%）
            # num_samples = len(self.sample_list)
            # num_samples_to_select = int(1 * num_samples)
            # # 随机选择数据
            # selected_samples = random.sample(self.sample_list, num_samples_to_select)
            # self.sample_list = selected_samples

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":   #如果提供了数据数量，就只加载指定数量的数据
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)    #返回数据集的长度

    def __getitem__(self, idx):   #根据索引获取数据
        case = self.sample_list[idx]    #获取指定索引的数据
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")    #训练集在data/slices文件夹下
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")   #验证集在data文件夹下
        image = h5f["image"][:]   #获取图像数据
        label = h5f["label"][:]  #获取标签数据
        sample = {"image": image, "label": label}
        if self.split == "train":   #如果是训练集
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class BaseDataSets_Synapse(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            # # 计算要选择的数据数量（10%）
            # num_samples = len(self.sample_list)
            # num_samples_to_select = int(1 * num_samples)
            # # 随机选择数据
            # selected_samples = random.sample(self.sample_list, num_samples_to_select)
            # self.sample_list = selected_samples

        elif self.split == "val":
            with open(self._base_dir + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = np.load(self._base_dir+"/Synapse" + "/train_npz/{}.npz".format(case))    #训练集在train_npz文件夹下
        else:
            if self.split == "val":
                h5f = h5py.File(self._base_dir +"/Synapse" + "/test_vol_h5/{}.npy.h5".format(case))
            else:
                h5f = h5py.File(self._base_dir +"/Synapse" + "/test_vol_h5/{}.npy.h5".format(case))
                
        image = np.array(h5f["image"])
        label = np.array(h5f["label"])
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

#数据增强，包括随机旋转和翻转(这是一个弱增强操作，用于增强图像的旋转和翻转)
def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k) #k 次旋转将会产生四种可能的方向之一（0 度、90 度、180 度、270 度）
    axis = np.random.randint(0, 2)  # 0: vertical flip, 1: horizontal flip
    image = np.flip(image, axis=axis).copy()    #沿着指定轴翻转数组，确保翻转后的图像是独立于原始图像的副本
    if label is not None:   #如果有标签，也需要进行相同的操作
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

#数据增强，包括随机旋转(这是一个强增强操作，用于增强图像的旋转)
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)  # rotate between -20 and 20 degrees
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

#数据增强，包括颜色抖动(这是一个强增强操作，用于增强图像的颜色)
def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)    #返回颜色抖动后的图像

#这是一个转换器，用于将图像和标签转换为张量
class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):   #对图像和标签进行增强
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions 
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):  
        if ops is None:
            return pil_img
        for op, args in ops:    
            pil_img = OPS[op].f(pil_img, *args) 
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

#这是一个随机生成器，用于生成随机的旋转和翻转 
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

#这是一个弱强增强器，用于生成弱增强和强增强的图像
class WeakStrongAugment(object):
    """returns weakly and strongly augmented images 

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label_aug = random_rot_flip(image, label)
        # strong augmentation is color jitter   
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        label_aug = torch.from_numpy(label_aug.astype(np.uint8))
        # print(label.shape,image_weak.shape, image_strong.shape, label_aug.shape)

        sample = {
            "image": image, #原始图像
            "label": label, #原始标签
            "image_weak": image_weak,   #弱增强图像
            "image_strong": image_strong,   #强增强图像
            "label_aug": label_aug,  #增强后的标签
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

#同时处理标记数据和未标记数据
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    #输入参数包括主要索引、次要索引、批量大小和次要批量大小
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        #确保主要和次要索引集的长度分别大于等于各自的批次大小，且批次大小大于0
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):     #仅迭代一次的随机排列序列
    return np.random.permutation(iterable)


def iterate_eternally(indices):    #无限迭代的随机排列序列
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

#接收一个可迭代对象和一个正整数 n，返回一个生成器，每次产生一个包含 n 个元素的元组
def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
