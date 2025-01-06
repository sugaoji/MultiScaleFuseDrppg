import torch
import numpy as np
import glob
import os
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as T
import config
import engine
from Dataset_loaders.dataset import DataLoader
from Utils.utils import compute_criteria
from Loss.loss_r import Neg_Pearson
from Utils.model_utils import save_model_checkpoint, setup_seed
from Models import diff_model
from sklearn import model_selection
import scipy.io
import sys

from dataset_train_test_split import train_test_split

os.chdir(sys.path[0])



def run_training(bs, dim, depth, heads, mlpdim, num_clusters):

    if torch.cuda.is_available():
        print("GPU available... Using GPU")
    else:
        print("GPU not available, using CPU")

    Map_path = config.MST_MAP_PATH
    dataset_type = Map_path.split('/')[-2].split('_')[0]

    TRAIN_CHK_PATH = f"./Checkpoint/{dataset_type}/train"
    BEST_CHK_PATH = f"./Checkpoint/{dataset_type}/best"
    os.makedirs(TRAIN_CHK_PATH,exist_ok=True)
    os.makedirs(BEST_CHK_PATH,exist_ok=True)

    # file_name = f'(MultiFreqFuseDrppg)_{dataset_type}_Epoch{config.EPOCHS}_LRate{config.L_rate}_Dim{dim}_Depth{depth}_TotalSteps{config.Total_steps}_Ktimes{config.K}'
    file_name = f'(MultiScaleFuseDrppg)_{dataset_type}_Epoch{config.EPOCHS}_LRate{config.L_rate}_Dim{dim}_Depth{depth}_TotalSteps{config.Total_steps}_Ktimes{config.K}'
    if dataset_type == "VIPL":
        file_name = f"FOLD_{config.VIPL_Fold}_" + file_name

    os.makedirs(f'./logs/{dataset_type}',exist_ok=True)
    log_file = f"./logs/{dataset_type}/{file_name}.txt"
    print(log_file)

    with open(log_file, 'w') as f:
        f.writelines([f"model=d_rppg_{dataset_type}, Train_bs={config.Train_batchsize}, Test_bs={config.Test_batchsize}learning_rate={config.L_rate},k_times={config.K},total_steps={config.Total_steps}\n",
                      "==============================\n"])
        f.flush()

    train, test, transform_train, transform_test = train_test_split(dataset_type,Map_path)

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

    model = diff_model.ViT(
        image_height=63,
        image_width=300,
        num_classes=300,
        num_clusters=num_clusters,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlpdim,
        dropout=0.1,
        emb_dropout=0.1,
        is_test = False
    )

    if (dataset_type == "VIPL"):
        print("load the pretrained_model from UBFC")
        pretrained_model_ubfc = 'RMSE(0.408)_18_of_(MultiScaleFuseDrppg)_UBFC_Epoch50_LRate0.001_Dim256_Depth4_TotalSteps1000_Ktimes5.pt'
        print("The pretrained model name: ",pretrained_model_ubfc)
        model.load_state_dict(torch.load(f"./Checkpoint/UBFC/best/{pretrained_model_ubfc}")['model_state_dict'])

    model.cuda()
#     if torch.cuda.device_count()>1:
    model=torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.L_rate)
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)

    lossfunc_mse = torch.nn.MSELoss()
    lossfunc_pearson = Neg_Pearson(downsample_mode=0)
    # lossfunc_ecg = torch.nn.MSELoss()
    min_mae = 1e3
    min_rmse = 1e3
    min_loss = 1e3

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    val_metrics_per_epoch = []
    test_list = [x.split('/')[-1] for x in test]

    for epoch in range(config.EPOCHS):

        _, _, _, train_loss_per_batch = engine.train_fn(model, train_loader, optimizer, lossfunc_pearson, lossfunc_mse)
        epoch_model_file_name = f"{epoch}_of_{file_name}.pt"
        last_epoch_model_file_name = f"{epoch-1}_of_{file_name}.pt"
        print(os.path.join(TRAIN_CHK_PATH,last_epoch_model_file_name),"this is last epoch model")
        if os.path.exists(os.path.join(TRAIN_CHK_PATH,last_epoch_model_file_name)):
            # 如果文件存在，则删除文件
            os.remove(os.path.join(TRAIN_CHK_PATH,last_epoch_model_file_name))
        save_model_checkpoint(model, epoch_model_file_name,checkpoint_path=TRAIN_CHK_PATH)
        
        train_loss_per_epoch.append(np.mean(train_loss_per_batch))

        with open(log_file, "a") as f:
            f.writelines([f"\nTraining! [Epoch: {epoch + 1}/{config.EPOCHS}]",
                            "\nTraining Loss: {:.3f} |".format(train_loss_per_epoch[-1]),
                            ])
            f.flush()

        if True:
            model_eval_temp = diff_model.ViT(
                image_height=63,
                image_width=300,
                num_classes=300,
                num_clusters=num_clusters,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlpdim,
                dropout=0.1,
                emb_dropout=0.1,
                is_test=True
            )
            model_eval_temp.load_state_dict(torch.load(os.path.join(TRAIN_CHK_PATH , epoch_model_file_name))['model_state_dict'])
            model_eval_temp.cuda()
            model_eval_temp = torch.nn.DataParallel(model_eval_temp)
            val_target_hr_per_batch, val_predicted_hr_per_batch, val_target_bvp_per_batch, val_predicted_bvp_per_batch, val_loss_per_batch, path_per_batch = engine.eval_fn(
                model_eval_temp, test_loader, lossfunc_pearson, lossfunc_mse)
            val_loss_per_epoch.append(np.mean(val_loss_per_batch))
            val_target_hr_per_batch = np.array(val_target_hr_per_batch)
            val_predicted_hr_per_batch = np.array(val_predicted_hr_per_batch)

            target_hr_per_video = []
            predicted_hr_per_video = []

            for x in test_list:
                sum_target_hr_per_map = 0
                sum_predicted_hr_per_map = 0
                count = 0
                for i, y in enumerate(path_per_batch):
                    if x in y:
                        count += 1
                        sum_target_hr_per_map += val_target_hr_per_batch[i]
                        sum_predicted_hr_per_map += val_predicted_hr_per_batch[i]
                target_hr_per_video.append(round(sum_target_hr_per_map / count))
                predicted_hr_per_video.append(round(sum_predicted_hr_per_map / count))

            val_metrics = compute_criteria(np.array(target_hr_per_video), np.array(predicted_hr_per_video))
            val_metrics_per_epoch.append(val_metrics)
#             scheduler.step(val_metrics["RMSE"])

            with open(log_file, "a") as f:
                f.writelines([f"Testing! [Epoch: {epoch + 1}/{config.EPOCHS}]",
                                "\nTesting Loss: {:.3f} |".format(val_loss_per_epoch[-1]),
                                "HR_MAE : {:.3f} |".format(val_metrics["MAE"]),
                                "HR_RMSE : {:.3f} |".format(val_metrics["RMSE"]),
                                "HR_MER: {:.3f} |".format(val_metrics["MER"]),
                                "HR_r: {:.3f} |".format(val_metrics["r"]),
                                "HR_std: {:.3f} |\n".format(val_metrics["std"])
                                ])
                f.flush()

            if len(val_loss_per_epoch) > 0 and min_rmse >= val_metrics["RMSE"]:
                RMSE_metric = val_metrics["RMSE"]
                best_model_file_name = f"RMSE({RMSE_metric:.3f})_{epoch}_of_{file_name}.pt"
                best_model_files = os.listdir(BEST_CHK_PATH)
                # 遍历目录中的所有文件
                for file in best_model_files:
                    # 检查文件名是否包含指定的子字符串
                    if file_name in file:
                        # 构建完整的文件路径
                        file_path = os.path.join(BEST_CHK_PATH, file)
                        # 删除文件
                        os.remove(file_path)
                save_model_checkpoint(model, best_model_file_name, checkpoint_path=BEST_CHK_PATH)
#                     min_loss = val_loss_per_epoch[-1]
                min_rmse = val_metrics['RMSE']
                min_mae = val_metrics['MAE']

        scheduler.step()

    with open(log_file, "a") as f:
        f.writelines([f"MIN MAE OF {config.EPOCHS} epochs: {min_mae}  ",
                        f"MIN RMSE OF {config.EPOCHS} epochs: {min_rmse}\n"])
        f.flush()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

    BS = [16,]
    Dim = [256,]
    Depth = [config.MIXste_depth,]
    Heads = [4,]
    Mlpdim = [256,]
    Num_clusters = [63,]

    for bs in BS:
        for dim in Dim:
            for depth in Depth:
                for heads in Heads:
                    for mlpdim in Mlpdim:
                        for num_clusters in Num_clusters:
                            run_training(bs, dim, depth, heads, mlpdim, num_clusters)


