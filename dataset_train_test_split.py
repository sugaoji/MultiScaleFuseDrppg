import glob
import os

import scipy
from sklearn import model_selection
import torchvision.transforms as T

import config


def SplitforSubject(file_list, name,fold_num, n_splits=5):
    """
    :param file_list: 每条video的dir
    :param n_splits:
    :return: 按subject划分的train和test

    Args:
    """
    kf = model_selection.KFold(n_splits=n_splits, shuffle=True)
    file_list = [x for x in file_list if os.listdir(x)]
    train_ls = []
    test_ls = []
    if "VIPL" == name:
        fold_data_dict = {}
        fold_files = glob.glob(f"./Utils/fold_split/fold{fold_num}.mat")
        for fold in fold_files:
            name = fold.split('/')[-1].split('.')[0]
            fold_data = scipy.io.loadmat(fold)
            fold_data_dict[name] = fold_data[name]
        subjects = [x + 1 for x in range(107)]
        for idx, fold in enumerate(fold_data_dict.keys()):
            test_subject = [f'p{x}_' for x in fold_data_dict[fold].squeeze(0)]
            # test_subject = list(map(str, test_subject))
            train_subject = [f'p{x}_' for x in subjects if f'p{x}_' not in test_subject]
            # train_subject = list(map(str, train_subject))
            train_video = [x for x in file_list if f'{x.split("/")[-1].split("_")[0]}_' in train_subject]
            test_video = [x for x in file_list if f'{x.split("/")[-1].split("_")[0]}_' in test_subject]
            train_ls.append(train_video)
            test_ls.append(test_video)
    else:
        print("not find such dataset!")

    return train_ls, test_ls

def train_test_split(dataset_type,Map_path ):
    Maps_path_per_video = glob.glob(os.path.join(Map_path, '*'))
    trainset_dic = {}
    testset_dic = {}

    # UBFC
    testset_dic["UBFC"] = [f'{Map_path}/subject{x}' for x in range(38, 50)]
    trainset_dic["UBFC"] = [x for x in Maps_path_per_video if x not in testset_dic["UBFC"]]
    # trainset_dic["UBFC"] = [f'{Map_path}/subject{x}' for x in range(1, 2)]
    # testset_dic["UBFC"] = [f'{Map_path}/subject{x}' for x in range(3, 4)]

    # MMPD
    trainset_dic["MMPD"] = []
    for i in range(1, 24):
        prefix = f'p{i}'
        folders = [f for f in os.listdir(Map_path) if
                   os.path.isdir(os.path.join(Map_path, f)) and f.startswith(prefix)]
        for folder in folders:
            trainset_dic["MMPD"].append(os.path.join(Map_path, folder))
    testset_dic["MMPD"] = [x for x in Maps_path_per_video if x not in trainset_dic["MMPD"]]

    testset_dic["PURE"] = []
    subject = ['01', '02', '04', '03']
    for sj in subject:
        datas = glob.glob(f'{Map_path}/{sj}*')
        for data in datas:
            testset_dic["PURE"].append(data)
    trainset_dic["PURE"] = [x for x in Maps_path_per_video if x not in testset_dic["PURE"]]

    # VIPL
    traindata, testdata = SplitforSubject(Maps_path_per_video, name="VIPL", fold_num=config.VIPL_Fold)
    trainset_dic["VIPL"] = traindata[0]
    testset_dic["VIPL"] = testdata[0]

    train_list = trainset_dic[dataset_type]
    test_list = testset_dic[dataset_type]



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

    mmpd_transform_train = T.Compose([
        T.ToTensor(),
        T.Normalize(
            [0.5075124502182007, 0.49850431084632874, 0.49572888016700745, 0.5010648965835571, 0.49240589141845703,
             0.5182220339775085],
            [0.26640602946281433, 0.2659032940864563, 0.2612386643886566, 0.26867133378982544, 0.2477424591779709,
             0.2556385099887848])])
    mmpd_transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(
            [0.5075124502182007, 0.49850431084632874, 0.49572888016700745, 0.5010648965835571, 0.49240589141845703,
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
    if (dataset_type == "MMPD"):
        transform_train = mmpd_transform_train
        transform_test = mmpd_transform_test

    return train_list, test_list, transform_train, transform_test