import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam

from model.deep_model import IvSmoother
from model.prior_model import ATMInterpolator
from dataset.iv_data import IvDataset

def filter(data_dict,ttm,ttm_key="ttm_fit:0"):
    index_s = []
    for i in range(len(data_dict[ttm_key])):
        # print(data_dict[ttm_key][i].item()-ttm)
        if abs(data_dict[ttm_key][i].item()-ttm)<1e-4:
            # print("!!!!!!")
            index_s.append(i)
    return torch.tensor(index_s,dtype=torch.int64)





if __name__ == "__main__":
    df = pd.read_csv("./data/IO_data.csv")
    date_train = "2024-06-05"
    date_test = "2024-06-06"
    df = df[df["quote_date"]==date_test]
    df_atm = df.loc[df.groupby('ttm')['logm'].apply(lambda x: abs(x).idxmin())]
    print(df_atm)
    w_atm_fun = ATMInterpolator(df_atm).interpolate()
    iv_data = IvDataset(df)
    data_di = iv_data.get_ivsmoother_dict()["di"]
    model = IvSmoother([40,40,40,40],prior="svi",w_atm_fun=w_atm_fun)
    device = "cuda:0"
    model.to(device)
    for key in data_di.keys():
        data_di[key] = data_di[key].to(device)
    model.load_state_dict(torch.load(f"./checkpoint/model_{date_train}.pth"))
    model.eval()
    # 绘制模型在训练集上的结果
    # 根据ttm的不同进行分类
    ttm_set = data_di["ttm_fit:0"].unique()
    print(len(ttm_set))
    data_di["iv_hat:0"] = model.compute_iv_hat(data_di["ttm_fit:0"],data_di["logm_fit:0"])
    for i in range(len(ttm_set)):
        index_s = filter(data_di,ttm_set[i])
        # print(index_s)
        data_logm = data_di["logm_fit:0"][index_s]
        data_iv_hat = data_di["iv_hat:0"][index_s]
        data_iv_true = data_di["iv:0"][index_s]
        plt.figure()
        plt.plot(data_logm.detach().cpu(),data_iv_hat.detach().cpu(),color="red",marker=".",label="pred")
        plt.plot(data_logm.detach().cpu(),data_iv_true.detach().cpu(),color="blue",marker=".",label="true")
        plt.legend()
        plt.title(f"ttm={ttm_set[i]}")
        # pic:6.5训练，6.6测试
        # pic_2:6.5训练，6.5测试
        plt.show()
        # plt.savefig(f"./pic/ttm_{i}.png",dpi=400)
