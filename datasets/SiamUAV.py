

from __future__ import absolute_import, print_function

import sys
import os
import numpy as np
from torch.utils.data import Dataset
import glob
import json
from PIL import Image
import cv2
from .Augmentation import RandomCrop,RandomRotate,EdgePadding,RandomResize,RotateAndCrop
from torchvision import transforms
from .random_erasing import RandomErasing
import math
class SiamUAV_test(Dataset):
    def __init__(self, root_dir, opt,mode="1"):
        '''
        :param root_dir: root of SiamUAV
        :param transform: a dict, format as {"UAV":Compose(),"Satellite":Compose()}
        '''
        super(SiamUAV_test, self).__init__()
        self.root_dir = root_dir
        self.opt = opt
        self.transform = self.get_transformer()
        self.root_dir_train = os.path.join(self.root_dir,mode)
        self.seq = glob.glob(os.path.join(self.root_dir_train,"*"))
        self.list_all_info = self.get_total_info()

    def get_total_info(self):
        list_all_info = []
        for seq in self.seq:

            UAV_list = glob.glob(os.path.join(seq, "UAV/*.[JjPp]*[GgNn]"))  # 匹配.jpg/.JPG/.png/.PNG等
            if not UAV_list:  # 跳过无无人机图的seq
                print(f"警告：序列 {seq} 的UAV目录下无图像，已跳过")
                continue
            # 2. 加载当前seq下所有卫星图像（保持原逻辑，但确保非空）
            Satellite_list = glob.glob(os.path.join(seq, "Satellite/*"))
            if not Satellite_list:  # 跳过无卫星图的seq
                print(f"警告：序列 {seq} 的Satellite目录下无图像，已跳过")
                continue
            # 3. 加载标签文件（确保每个卫星图的位置标签存在）
            label_path = os.path.join(seq, "labels.json")
            if not os.path.exists(label_path):
                print(f"警告：序列 {seq} 缺少labels.json，已跳过")
                continue
            with open(label_path, 'r', encoding='utf8') as fp:
                json_context = json.load(fp)
            # 4. 多对多配对：每张无人机图 ↔ 每张卫星图（生成所有组合的样本）
            for uav_path in UAV_list:
                for sat_path in Satellite_list:
                    sat_filename = os.path.basename(sat_path)  # 卫星图文件名（匹配标签用）
                    # 检查当前卫星图是否有对应的标签
                    if sat_filename not in json_context:
                        print(f"警告：卫星图 {sat_filename} 在 {label_path} 中无标签，已跳过")
                        continue
                    # 存储单一样本信息（无人机图+卫星图+对应标签）
                    single_dict = {
                        "UAV": uav_path,
                        "Satellite": sat_path,
                        "position": json_context[sat_filename]  # 卫星图对应的位置标签
                    }
                    list_all_info.append(single_dict)
        return list_all_info
##第一种图像处理方法
    def gamma(self, image):
        fgamma = 1
        image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
        cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(image_gamma, image_gamma)
        return image_gamma
    def gamma1(self,image):
        fgamma = 1
        image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
        cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(image_gamma, image_gamma)
        return image_gamma
##第一种图像处理方法





    def get_transformer(self):
        transform_uav_list = [
            transforms.Resize(self.opt.UAVhw, interpolation=3),
            transforms.ToTensor()
        ]

        transform_satellite_list = [
            transforms.Resize(self.opt.Satellitehw, interpolation=3),
            transforms.ToTensor()
        ]

        data_transforms = {
            'UAV': transforms.Compose(transform_uav_list),
            'satellite': transforms.Compose(transform_satellite_list)
        }

        return data_transforms

    def __len__(self):
        return len(self.list_all_info)

    def __getitem__(self, index):
        single_info = self.list_all_info[index]
        UAV_image_path = single_info["UAV"]

       # ## UAV_image = Image.open(UAV_image_path)  原来的方法暂时注释
        image = cv2.imread(UAV_image_path)
        image_gamma = self.gamma(image)
        image_gamma = cv2.cvtColor(image_gamma, cv2.COLOR_BGR2RGB)
        UAV_image = Image.fromarray(image_gamma)
        UAV_image = self.transform["UAV"](UAV_image)




        Satellite_image_path = single_info["Satellite"]
        #Satellite_image_ = Image.open(Satellite_image_path)
        image = cv2.imread(Satellite_image_path)
        image_gamma = self.gamma1(image)
        image_gamma = cv2.cvtColor(image_gamma, cv2.COLOR_BGR2RGB)
        Satellite_image_ = Image.fromarray(image_gamma)
        Satellite_image = self.transform["satellite"](Satellite_image_)
        X,Y = single_info["position"]
        X = int(X/Satellite_image_.height*self.opt.Satellitehw[0])
        Y = int(Y/Satellite_image_.width*self.opt.Satellitehw[1])
        return [UAV_image,Satellite_image,X,Y,UAV_image_path,Satellite_image_path]


class SiamUAVCenter(Dataset):
    def __init__(self, root_dir, opt):
        '''
        :param root_dir: root of SiamUAV
        :param transform: a dict, format as {"UAV":Compose(),"Satellite":Compose()}
        '''
        super(SiamUAVCenter, self).__init__()
        self.opt = opt
        self.root_dir = root_dir
        self.transform = self.get_transformer()
        self.root_dir_train = os.path.join(self.root_dir,"dataset_train3")
        self.seq = glob.glob(os.path.join(self.root_dir_train,"*"))
        self.list_all_info = self.get_total_info()
        self.SatelliteAugmentation = RandomCrop(cover_rate=0.7,map_size=(512,1200))

    def get_total_info(self):
        list_all_info = []
        for seq in self.seq:
            # 1. 加载当前seq下所有无人机图像（不再固定单张）
            UAV_list = glob.glob(os.path.join(seq, "UAV/*.[JjPp]*[GgNn]"))  # 匹配常见图像格式
            if not UAV_list:
                print(f"警告：序列 {seq} 的UAV目录下无图像，已跳过")
                continue
            # 2. 加载当前seq下所有卫星图像（不再固定0.tif）
            Satellite_list = glob.glob(os.path.join(seq, "Satellite/*"))  # 匹配所有卫星图
            if not Satellite_list:
                print(f"警告：序列 {seq} 的Satellite目录下无图像，已跳过")
                continue
            # 3. 多对多配对：每张无人机图 ↔ 每张卫星图
            for uav_path in UAV_list:
                for sat_path in Satellite_list:
                    single_dict = {
                        "Satellite": sat_path,  # 存储当前卫星图路径
                        "UAV": uav_path          # 存储当前无人机图路径
                    }
                    list_all_info.append(single_dict)
        return list_all_info

#第一种图像处理方法
    def gamma(self,image):
        fgamma = 1
        image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
        cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(image_gamma, image_gamma)
        return image_gamma

    def gamma1(self,image):
        fgamma = 1
        image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
        cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(image_gamma, image_gamma)
        return image_gamma


    def get_transformer(self):
        transform_uav_list = [
            RandomResize(self.opt.UAVhw),
            # transforms.RandomRotation((-180,180)),
            # transforms.RandomApply([transforms.GaussianBlur(21, 10)], p=0.5),
            # RandomRotate(),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.3),
        ]

        transform_satellite_list = [
            RandomResize(self.opt.Satellitehw),
            transforms.ToTensor(),
        ]

        if self.opt.padding:
            transform_uav_list = [EdgePadding(self.opt.padding)] + transform_uav_list

        data_transforms = {
            'UAV': transforms.Compose(transform_uav_list),
            'Satellite': transforms.Compose(transform_satellite_list)
        }

        return data_transforms

    def __len__(self):
        return len(self.list_all_info)

    def __getitem__(self, index):
        # load the json context
        single_info = self.list_all_info[index]
        UAV_image_path = single_info["UAV"]

      ##  UAV_image = Image.open(UAV_image_path)  原来的读取方法暂时注释
        # ###第一种图像处理方法
        image = cv2.imread(UAV_image_path)
        image_gamma = self.gamma(image)
        image_gamma = cv2.cvtColor(image_gamma, cv2.COLOR_BGR2RGB)
        UAV_image = Image.fromarray(image_gamma)
        #UAV_image.show()
        UAV_image = self.transform["UAV"](UAV_image)


        Satellite_image_path = single_info["Satellite"]
      #  Satellite_image = Image.open(Satellite_image_path)
        image = cv2.imread(Satellite_image_path)
        image_gamma = self.gamma1(image)
        image_gamma = cv2.cvtColor(image_gamma, cv2.COLOR_BGR2RGB)
        Satellite_image = Image.fromarray(image_gamma)
        #Satellite_image.show()
        Satellite_image,[ratex,ratey] = self.SatelliteAugmentation(Satellite_image)
        Satellite_image = self.transform["Satellite"](Satellite_image)

        return [UAV_image,Satellite_image,ratex,ratey]# x y 为裁剪后的图像中心和原图中心的相对位置，如果重合则为0.5
