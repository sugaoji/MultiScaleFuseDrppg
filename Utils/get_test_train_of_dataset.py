import glob
import os

import torch
import torchvision.transforms as T
import config

from Dataset_loaders.dataset import DataLoader
from Train import SplitforSubject


def get_test_train (Map_path,dataset_type,fold_num = 1):
    Maps_path_per_video = glob.glob(os.path.join(Map_path, '*'))

    trainset_dic = {}
    testset_dic = {}

    # UBFC
    if dataset_type == "UBFC":
        testset_dic["UBFC"] = [f'{Map_path}/subject{x}' for x in range(38, 50)]
        trainset_dic["UBFC"] = [x for x in Maps_path_per_video if x not in testset_dic["UBFC"]]

    # PURE
    if dataset_type == "PURE":
        testset_dic["PURE"] = []
        subject = ['01', '02', '04', '03']
        for sj in subject:
            datas = glob.glob(f'{Map_path}/{sj}*')
            for data in datas:
                testset_dic["PURE"].append(data)
        trainset_dic["PURE"] = [x for x in Maps_path_per_video if x not in testset_dic["PURE"]]

    # VIPL
    if dataset_type == "VIPL":
        traindata, testdata = SplitforSubject(Maps_path_per_video, name="VIPL" ,fold_num=fold_num)
        trainset_dic["VIPL"] = traindata
        testset_dic["VIPL"] = testdata

    train = trainset_dic[dataset_type]
    test = testset_dic[dataset_type]

    # MSTmap for UBFC success1
    ubfc_transform_train = T.Compose([
        T.ToTensor(),
        T.Normalize([0.49826374650001526, 0.4897593557834625, 0.48688215017318726, 0.4903276562690735, 0.4862031638622284,
                     0.5129157900810242],
                    [0.2629663944244385, 0.25689175724983215, 0.25882282853126526, 0.2566501200199127, 0.2559047341346741,
                     0.26456427574157715])])
    ubfc_transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize([0.49826374650001526, 0.4897593557834625, 0.48688215017318726, 0.4903276562690735, 0.4862031638622284,
                     0.5129157900810242],
                    [0.2629663944244385, 0.25689175724983215, 0.25882282853126526, 0.2566501200199127, 0.2559047341346741,
                     0.26456427574157715])])

    # MSTmap for PURE_MST
    pure_transform_train = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5100011229515076, 0.5100011229515076, 0.5043121576309204, 0.501944899559021, 0.49301525950431824,
                     0.5126286745071411],
                    [0.2933797240257263, 0.27947697043418884, 0.27273887395858765, 0.2857646644115448, 0.2774929702281952,
                     0.27686169743537903])])

    pure_transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5100011229515076, 0.5100011229515076, 0.5043121576309204, 0.501944899559021, 0.49301525950431824,
                     0.5126286745071411],
                    [0.2933797240257263, 0.27947697043418884, 0.27273887395858765, 0.2857646644115448, 0.2774929702281952,
                     0.27686169743537903])])

    # MSTmap for VIPL_MST
    vipl_transform_train = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5075124502182007, 0.49850431084632874, 0.49572888016700745, 0.5010648965835571, 0.49240589141845703,
                     0.5182220339775085],
                    [0.26640602946281433, 0.2659032940864563, 0.2612386643886566, 0.26867133378982544, 0.2477424591779709,
                     0.2556385099887848])])
    vipl_transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5075124502182007, 0.49850431084632874, 0.49572888016700745, 0.5010648965835571, 0.49240589141845703,
                     0.5182220339775085],
                    [0.26640602946281433, 0.2659032940864563, 0.2612386643886566, 0.26867133378982544, 0.2477424591779709,
                     0.2556385099887848])])

    if (dataset_type == "UBFC"):
        transform_train = ubfc_transform_train
        transform_test = ubfc_transform_test
    if (dataset_type == "PURE"):
        transform_train = pure_transform_train
        transform_test = pure_transform_test
    if (dataset_type == "VIPL"):
        transform_train = vipl_transform_train
        transform_test = vipl_transform_test

    dataset_train = DataLoader(train, transforms=transform_train)
    dataset_test = DataLoader(test, transforms=transform_test)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config.Train_batchsize,
        # num_workers=config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=config.Test_batchsize,
        # num_workers=config.NUM_WORKERS,
        shuffle=False,
        pin_memory=True
    )

    return train_loader,test_loader,test