import pandas as pd
import matplotlib.pyplot as plot

import torch
from torch.optim import Adam

from model.deep_model import IvSmoother
from model.prior_model import ATMInterpolator
from dataset.iv_data import IvDataset

if __name__ == "__main__":
    df = pd.read_csv("./data/IO_data.csv")
    date = "2024-06-05"
    df = df[df["quote_date"]==date]
    df_atm = df.loc[df.groupby('ttm')['logm'].apply(lambda x: abs(x).idxmin())]
    w_atm_fun = ATMInterpolator(df_atm).interpolate()
    iv_data = IvDataset(df)
    data_di = iv_data.get_ivsmoother_dict()["di"]
    model = IvSmoother([40,40,40,40],prior="svi",w_atm_fun=w_atm_fun)
    device = "cuda:0"
    model.to(device)
    for key in data_di.keys():
        data_di[key] = data_di[key].to(device)
    model.train()
    optim = Adam(model.parameters(),lr=3e-4)
    for epoch in range(10000):
        optim.zero_grad()
        loss = model.get_all_loss(data_di)
        if (epoch+1) % 100 == 0:
            print(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optim.step()
    torch.save(model.state_dict(),f"./checkpoint/new_model_{date}.pth")
