import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from scipy.interpolate import interp1d,CubicSpline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class ATMInterpolator:
    def __init__(self, df_atm,ttm_name="ttm",w_name="w"):
        self.ttm_atm = df_atm[ttm_name].values
        self.w_atm = df_atm[w_name].values
        self.min_ttm = 0
        self.max_ttm = max(self.ttm_atm)
        self.ttm_grid = np.arange(self.min_ttm, self.max_ttm, 0.01)
        self.fit_y = self._fit()

    def _fit(self):
        if np.all(np.diff(self.w_atm) >= 0):
            ttm_atm = np.concatenate([[0], self.ttm_atm])
            w_atm = np.concatenate([[0], self.w_atm])
            fit = CubicSpline(ttm_atm, w_atm)
            return fit(self.ttm_grid)
        else:
            poly = PolynomialFeatures(degree=10)
            ttm_atm_poly = poly.fit_transform(self.ttm_atm.reshape(-1, 1))
            model = LinearRegression().fit(ttm_atm_poly, np.log(self.w_atm))
            ttm_grid_poly = poly.transform(self.ttm_grid.reshape(-1, 1))
            return np.exp(model.predict(ttm_grid_poly))

    def interpolate(self):
        # interp_func = interp1d(self.ttm_grid, self.fit_y, fill_value="extrapolate", bounds_error=False)
        # return interp_func(ttm)
        return interp1d(self.ttm_grid, self.fit_y, fill_value="extrapolate", bounds_error=False)

class PriorModel(nn.Module):
    def __init__(self, prior="svi", phi_fun="power_law"):
        super(PriorModel, self).__init__()
        self.prior = prior
        self.phi_fun = phi_fun

        if prior == "svi":
            self.rho_trans = nn.Parameter(torch.zeros(1))

            if phi_fun == "heston_like":
                self.lambda_trans = nn.Parameter(torch.zeros(1))
            elif phi_fun == "power_law":
                self.eta_trans = nn.Parameter(torch.zeros(1))
                self.gamma_trans = nn.Parameter(torch.zeros(1))
            else:
                raise ValueError("Incorrect function for phi")
        else:
            raise NotImplementedError("Only 'svi' prior is implemented")

    def forward(self, logm, w_atm):
        rho = torch.tanh(self.rho_trans)

        if self.prior == "svi":
            if self.phi_fun == "heston_like":
                lambda_ = torch.exp(self.lambda_trans)
                phi = 1 / (lambda_ * w_atm) * (1 - (1 - torch.exp(-lambda_ * w_atm)) / (lambda_ * w_atm))
            elif self.phi_fun == "power_law":
                eta = torch.exp(self.eta_trans)
                gamma = torch.sigmoid(self.gamma_trans)
                phi = eta / (torch.pow(w_atm, gamma) * torch.pow(1 + w_atm, 1 - gamma))
            else:
                raise ValueError("Incorrect function for phi")

            w_prior = w_atm / 2 * (1 + rho * phi * logm + torch.sqrt(torch.square(phi * logm + rho) + 1 - torch.square(rho)))
            return w_prior
        else:
            raise NotImplementedError("Only 'svi' prior is implemented")

# 示例用法
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("./data/IO_data.csv")
    df = df[df["quote_date"]=="2024-06-07"]
    df_atm = df.loc[df.groupby('ttm')['logm'].apply(lambda x: abs(x).idxmin())]
    w_atm_fun = ATMInterpolator(df_atm).interpolate()
    logm = torch.tensor([0.1, 0.2, 0.3])
    ttm = torch.tensor([0.1, 0.2, 0.3])

    # 初始化 PriorModel
    prior_model = PriorModel(prior="svi", phi_fun="power_law")

    # 创建 ttm_logm 数据
    ttm_logm = {'ttm': ttm, 'logm': logm}

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(prior_model.parameters(), lr=0.01)

    # 示例目标
    y = torch.tensor([0.2, 0.25, 0.3])
    print(list(prior_model.parameters()))
    # 训练步骤
    for epoch in range(100):
        optimizer.zero_grad()
        w_atm = torch.from_numpy(w_atm_fun(ttm)).to(torch.float32)
        w_prior = prior_model(logm, w_atm)
        print(w_prior.shape)
        loss = criterion(w_prior, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print(list(prior_model.parameters()))
