import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.prior_model import PriorModel

class IvSmoother(nn.Module):
    def __init__(self, neurons_vec, activation='relu', penalty = dict(fit = 1, c4 = 10, c5 = 10, c6 = 10, atm = 0.1),prior=None, phi_fun=None, w_atm_fun=None, spread=False):
        super(IvSmoother, self).__init__()
        # 构建神经网络模型
        self.neurons_vec = neurons_vec
        self.activation = activation
        self.penalty = penalty
        self.prior = prior
        self.phi_fun = phi_fun
        self.w_atm_fun = w_atm_fun
        self.spread = spread
        # Define the activation function
        if activation == 'relu':
            self.afun = F.relu
        elif activation == 'softplus':
            self.afun = F.softplus
        else:
            raise ValueError("Unsupported activation function")
        # Define layers
        self.layers = nn.ModuleList()
        n_input = 2
        for i, n_neurons in enumerate(neurons_vec):
            self.layers.append(nn.Linear(n_input, n_neurons))
            n_input = n_neurons
        self.output_layer = nn.Linear(n_input, 1)
        self.softplus = nn.Softplus()
        self.alpha = nn.Parameter(torch.tensor([torch.log(torch.exp(torch.tensor(1.0)) - 1)]))
        # 加入先验模型
        if self.prior is not None:
            self.prior_model = PriorModel(prior)
        self.w_atm_fun = w_atm_fun
    def get_ann_out(self,ttm,logm):
        x = torch.stack((ttm, logm), dim=-1)
        for i, layer in enumerate(self.layers):
            if self.activation == 'relu':
                x = F.relu(layer(x))
            elif self.activation == 'tanh':
                x = torch.tanh(layer(x))
            else:
                raise ValueError("Unsupported activation function")
        y = self.output_layer(x)
        ann_out = torch.exp(self.alpha)*(1+torch.tanh(y))
        return ann_out.flatten()

    def forward(self, ttm, logm):
        w_nn = self.get_ann_out(ttm,logm)
        if self.prior is not None:
            w_atm = torch.from_numpy(self.w_atm_fun(ttm.detach().cpu().numpy())).to(torch.float32)
            w_atm = w_atm.to(ttm.device)
            w_prior = self.prior_model(logm, w_atm)
            w_hat = w_nn*w_prior
        else:
            w_hat = w_nn
        return w_hat
    
    def compute_w_hat(self, ttm, logm):
        return self.forward(ttm, logm)
    
    def compute_iv_hat(self, ttm, logm):
        w_hat = self.compute_w_hat(ttm, logm)
        """if (w_hat<0).any():
            print("!!!!!!!!!!!!!!!!!!!!!!!!!")"""
        iv_hat = torch.sqrt(w_hat / ttm)
        return iv_hat
    
    def get_loss_fit(self, w, w_hat, iv, iv_hat, iv_spread=None,prior=None):
        # l_fit_w_rmse = torch.mean((1e-6 + (w - w_hat) ** 2) ** 0.5)
        l_fit_w_rmse = torch.sqrt(torch.mean((1e-6 + (w - w_hat) ** 2)))
        l_fit_w_mape = torch.mean(torch.abs((w_hat - w) / (w + 1e-6)))
        l_fit_w = (l_fit_w_rmse + l_fit_w_mape)
        
        l_fit_iv_mape = torch.mean(torch.abs((iv_hat - iv) / (iv + 1e-6)))

        if iv_spread is None:
            # l_fit_iv_rmse = torch.mean((1e-6 + (iv - iv_hat) ** 2) ** 0.5)
            l_fit_iv_rmse = torch.sqrt(torch.mean((1e-6 + (iv - iv_hat) ** 2)))
        else:
            l_fit_iv_rmse = torch.mean(1e-6 + torch.abs(iv - iv_hat) / (1 + iv_spread))
        
        l_fit_iv = (l_fit_iv_rmse + l_fit_iv_mape)

        return {
            "l_fit_w_rmse": l_fit_w_rmse,
            "l_fit_w_mape": l_fit_w_mape,
            "l_fit_w": l_fit_w,
            "l_fit_iv_rmse": l_fit_iv_rmse,
            "l_fit_iv_mape": l_fit_iv_mape,
            "l_fit_iv": l_fit_iv
        }

    def get_loss_arb(self, w, ttm, logm):
        dvdt = torch.autograd.grad(w, ttm, grad_outputs=torch.ones_like(w),create_graph=True)[0]
        dvdm = torch.autograd.grad(w, logm, grad_outputs=torch.ones_like(w),create_graph=True)[0]
        d2vdm2 = torch.autograd.grad(dvdm, logm, grad_outputs=torch.ones_like(dvdm),create_graph=True)[0]

        l_c4 = torch.mean(F.relu(-dvdt))
        # 论文中的表达式和官方代码中的表达式不同，这里以代码中的表达式为准
        g_k = ((1-((logm*dvdm)/(2*w)))**2 -
               ((dvdm**2/4)*(1/w + 1/4)) + (d2vdm2 / 2))
        l_c5 = torch.mean(F.relu(-g_k))
        l_c6 = torch.mean(torch.abs(d2vdm2))

        return {
            "l_c4": l_c4,
            "l_c5": l_c5,
            "l_c6": l_c6,
            "g_k": g_k,
            "dvdt": dvdt,
            "dvdm": dvdm,
            "d2vdm2": d2vdm2
        }
    

    def get_loss_atm(self, ann_output):
        if self.prior is None:
            l_atm = torch.tensor(0.0)
        elif self.prior in ["svi", "bs"]:
            l_atm = torch.mean((1e-6 + (ann_output - 1.0) ** 2) ** 0.5)
        else:
            raise ValueError("Unknown prior")
        return {"l_atm": l_atm}
    
    def get_all_loss(self,data_dict,pred_type="iv"):
        iv_hat_fit = self.compute_iv_hat(data_dict["ttm_fit:0"],data_dict["logm_fit:0"])
        w_hat_fit = self.compute_w_hat(data_dict["ttm_fit:0"],data_dict["logm_fit:0"])
        w_fit = data_dict["w:0"]
        iv_fit = data_dict["iv:0"]
        if pred_type == "iv":
            loss_fit = self.get_loss_fit(w_fit,w_hat_fit,iv_fit,iv_hat_fit)["l_fit_iv"]
        elif pred_type == "w":
            loss_fit = self.get_loss_fit(w_fit,w_hat_fit,iv_fit,iv_hat_fit)["l_fit_w"]
        w_hat_c4c5 = self.compute_w_hat(data_dict["ttm_c4c5:0"],data_dict["logm_c4c5:0"])
        c4c5 = self.get_loss_arb(w = w_hat_c4c5, ttm = data_dict["ttm_c4c5:0"], logm = data_dict["logm_c4c5:0"])
        w_hat_c6 = self.compute_w_hat(data_dict["ttm_c6:0"],data_dict["logm_c6:0"])
        c6 = self.get_loss_arb(w=w_hat_c6,ttm=data_dict["ttm_c6:0"],logm=data_dict["logm_c6:0"])
        logm_atm = torch.zeros_like(data_dict["ttm_c4c5:0"])
        ann_output = self.get_ann_out(data_dict["ttm_c4c5:0"],logm_atm)
        loss_c4 = c4c5["l_c4"]+c6["l_c4"]
        loss_c5 = c4c5["l_c5"]+c6["l_c5"]
        loss_c6 = c6["l_c6"]
        loss_atm = self.get_loss_atm(ann_output)["l_atm"]
        return self.penalty["fit"]*loss_fit+self.penalty["c4"]*loss_c4+self.penalty["c5"]*loss_c5+self.penalty["c6"]*loss_c6+self.penalty["atm"]*loss_atm
