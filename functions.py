import numpy as np
import torch


#%% Normalization
class Normalization:
    def __init__(self, mean_val=None,std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):
        return (x-self.mean_val)/self.std_val

    def unnormalize(self, x):
        return x*self.std_val + self.mean_val


#%% Metrics
def metrics(y,x):
    #x: reference signal
    #y: estimated signal
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    #corrlation
    x_mean = np.mean(x, axis=-1, keepdims=True)
    y_mean = np.mean(y, axis=-1, keepdims=True)
    x_std = np.std(x, axis=-1, keepdims=True)
    y_std = np.std(y, axis=-1, keepdims=True)
    corr = np.mean((x-x_mean)*(y-y_mean), axis=-1,keepdims=True)/(x_std*y_std)

    #coefficeint of determination (r2)
    S_tot = np.sum((x-x_mean)**2, axis=-1, keepdims=True)
    S_res = np.sum((x - y)**2, axis=-1, keepdims=True)

    r2 = (1-S_res/S_tot)

    return torch.tensor(corr), torch.tensor(r2)



def display_results(loss, property_corr, property_r2, args, header):
    property_corr = torch.mean(torch.cat(property_corr), dim=0).squeeze()
    property_r2 = torch.mean(torch.cat(property_r2), dim=0).squeeze()
    loss = torch.mean(torch.tensor(loss))
    corr_text = " | ".join([u"{:d}\xb0: {:.4f}".format(args.incident_angles[i], property_corr[i].squeeze()) for i in range(len(args.incident_angles))])
    r2_text =   " | ".join([u"{:d}\xb0: {:.4f}".format(args.incident_angles[i], property_r2[i].squeeze()) for i in range(len(args.incident_angles))])
    print("loss: {:.4f}\nCorrelation: {:s}\nr2 Coeff.  : {:s}".format(loss,corr_text,r2_text))
