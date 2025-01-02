import os
import numpy as np
import matplotlib.pyplot as plt
import config

def plots(train, val, title, save_path=config.FIG_PATH):
    """
    :param train: list
    :param val: list 长度小于等于train
    :param title: str
    :return:
    """
    x_t = np.arange(1, len(train)+1).astype(dtype=str)
    x_v = np.arange(config.EPOCH_PER_TEST, len(train)+1, config.EPOCH_PER_TEST).astype(dtype=str)
    assert len(x_v) == len(val), "val length not equal to x_v"
    assert type(title) == str, "title type error"
    plt.plot(x_t, train)
    plt.plot(x_v, val)
    plt.title(title)
    plt.legend(["train", "val"])
    if save_path:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path+f"/{title}.png")
    plt.show()

if __name__ == "__main__":
    train=[]
    val=[]
    for x in range(20):
        train.append(x+np.random.random())
        if (x+1)%config.EPOCH_PER_TEST==0:
            val.append(x-np.random.random())
    plots(train,val,"test", save_path=None)
