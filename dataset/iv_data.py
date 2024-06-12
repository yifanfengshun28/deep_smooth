# 数据集要包括fit/c4c5/c6/atm
# 模型的训练过程没有使用batch size
import numpy as np
import pandas as pd
from itertools import product
import torch

class IvDataset:
    def __init__(self, df, types=["fit", "c4c5", "c6", "atm"], iv_spread=False, **kwargs):
        self.df = df
        self.types = types
        self.iv_spread = iv_spread
        self.args = kwargs
        self.ttm = kwargs.get('ttm', df['ttm'])
        self.logm = kwargs.get('logm', df['logm'])

    def get_name(self, base, type_):
        return f"{base}_{type_}"

    def get_logspace_ttm(self, ttm_max):
        return np.exp(np.linspace(np.log(1/365), np.log(ttm_max), num=100))

    def get_powerspace_logm(self, logm_min, logm_max):

        return np.linspace(-(-logm_min * 2)**(1/3), (logm_max * 2)**(1/3), num=100)**3

    def get_ttm_logm(self, type_, ttm, logm, args):
        ttm_type = self.get_name("ttm", type_)
        expand = True
        if ttm_type in args:
            ttm = args[ttm_type]
        else:
            if type_ == "fit":
                ttm = ttm
                expand = False
            elif type_ == "c4c5":
                ttm = self.get_logspace_ttm(max(ttm) + 1)
            elif type_ in ["c6", "atm"]:
                ttm = np.unique(ttm)

        logm_type = self.get_name("logm", type_)
        if logm_type in args:
            logm = args[logm_type]
        else:
            if type_ == "fit":
                logm = logm
            elif type_ == "c4c5":
                logm = self.get_powerspace_logm(min(logm), max(logm))
            elif type_ == "c6":
                logm = np.array([6, 4, 4, 6]) * np.repeat([min(logm), max(logm)], 2)
            elif type_ == "atm":
                logm = np.array([0])

        if expand:
            ttm_logm = pd.DataFrame(product(ttm, logm), columns=['ttm', 'logm'])
        else:
            ttm_logm = pd.DataFrame({'ttm': ttm, 'logm': logm})
        
        return ttm_logm

    def get_ivsmoother_dict(self):
        ttm_logm = pd.DataFrame({
            'type': self.types,
            'ttm_logm': [self.get_ttm_logm(type_, self.ttm, self.logm, self.args) for type_ in self.types]
        })

        di = {}
        for type_, df in zip(self.types, ttm_logm['ttm_logm']):
            ttm_type = f"{self.get_name('ttm', type_)}:0"
            logm_type = f"{self.get_name('logm', type_)}:0"
            df = df.rename(columns={'ttm': ttm_type, 'logm': logm_type})
            di[ttm_type] = torch.from_numpy(df[ttm_type].values).to(torch.float32)
            di[logm_type] = torch.from_numpy(df[logm_type].values).to(torch.float32)
            # di[logm_type] = torch.from_numpy(df[logm_type].values)
            # di.update(df.to_dict(orient='array'))
        di["iv:0"] = torch.from_numpy(self.df["iv"].values).to(torch.float32)
        di["w:0"] = torch.from_numpy(self.df["w"].values).to(torch.float32)
        di["ttm_c4c5:0"].requires_grad_()
        di["logm_c4c5:0"].requires_grad_()
        di["ttm_c6:0"].requires_grad_()
        di["logm_c6:0"].requires_grad_()
        if self.iv_spread:
            di["iv_spread:0"] = torch.from_numpy(self.df["iv_spread"].values).to(torch.float32)

        return {'ttm_logm': ttm_logm, 'di': di}
