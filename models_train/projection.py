from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch import Tensor
import torch
import numpy as np
import copy

class PLSRegressionCUDA:
    def __init__(self, n_components, cuda=True):
        self.n_components = n_components
        self.cuda = cuda
        self.cudaf = lambda x: x.cuda() if self.cuda else x

    def cuda(self):
        self.cuda = True
        self.P, self.W, self.Q, self.B = map(lambda x: x.cuda(), [self.P, self.W, self.Q, self.B])
        self.x_mean, self.y_mean = map(lambda x: x.cuda(), [self.x_mean, self.y_mean])

    def cpu(self):
        self.cuda = False
        self.P, self.W, self.Q, self.B = map(lambda x: x.cpu(), [self.P, self.W, self.Q, self.B])
        self.x_mean, self.y_mean = map(lambda x: x.cpu(), [self.x_mean, self.y_mean])

    def fit(self, X, Y):
        E = self.cudaf(X - X.mean(dim=0))
        F = self.cudaf(Y - Y.mean(dim=0))
        # X = TP^t + E
        T = self.cudaf(torch.zeros((X.size(0), self.n_components)))
        P = self.cudaf(torch.zeros((X.size(1), self.n_components)))
        W = self.cudaf(torch.zeros((X.size(1), self.n_components)))
        # Y = UQ^t + F
        U = self.cudaf(torch.zeros((Y.size(0), self.n_components)))
        Q = self.cudaf(torch.zeros((Y.size(1), self.n_components)))
        # U = BT
        B = self.cudaf(torch.zeros((self.n_components)))
        normed_matmul = lambda x, y: x@y/(y.T@y)
        for i in range(self.n_components):
            U[:, [i]] = F[:, [0]]
            W[:, [i]] = normed_matmul(E.T, U[:, [i]])
            W[:, [i]] /= W[:, [i]].norm()
            T[:, [i]] = normed_matmul(E,   W[:, [i]])
            Q[:, [i]] = normed_matmul(F.T, T[:, [i]])
            Q[:, [i]] /= Q[:, [i]].norm()
            U[:, [i]] = normed_matmul(F,   Q[:, [i]])
            P[:, [i]] = normed_matmul(E.T, T[:, [i]])
            # orthogonalization of t
            T[:, [i]] *= P[:, [i]].norm()
            W[:, [i]] *= P[:, [i]].norm()
            P[:, [i]] /= P[:, [i]].norm()
            # deflation
            B[[i]] = normed_matmul(U[:, [i]].T, T[:, [i]])
            F -= (B[i]*T[:, [i]])@Q[:, [i]].T # Y
            E -= T[:, [i]]@P[:, [i]].T # X
        self.P, self.W = P, W
        self.Q, self.B = Q, B
        self.y_mean = self.cudaf(Y.mean(dim=0))
        self.x_mean = self.cudaf(X.mean(dim=0))

        # self.x_scores_ = T
        # self.y_loading_ = Q
        # self.x_weights_ = W
        self.T, self.U = T, U
        return T.cpu(), U.cpu()

    def transform(self, X):
        E = self.cudaf(X) - self.x_mean
        T = self.cudaf(torch.zeros((X.size(0), self.n_components)))
        for i in range(self.n_components):
            T[:, [i]] = E@self.W[:, [i]]
            E -= T[:, [i]]@self.P[:, [i]].T
        return T.cpu()

    def predict(self, X):
        E = self.cudaf((X - X.mean(dim=0)))
        T = self.cudaf(torch.zeros((X.size(0), self.n_components)))
        y = self.cudaf(torch.zeros((X.size(0), 1)))
        for i in range(self.n_components):
            T[:, [i]] = E@self.W[:, [i]]
            E -= T[:, [i]]@self.P[:, [i]].T
            y[:, [0]] += self.B[i]*T[:, [i]]*self.Q[:, [i]].T
        return T.cpu(), (y + self.y_mean).cpu()

def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D)
    return D

def kernelmat(X, sigma, k_type="gaussian"):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    H = torch.eye(m) - (1./m) * torch.ones([m,m])

    if k_type == "gaussian":
        Dxx = distmat(X)
        
        if sigma:
            variance = 2.*sigma*sigma*X.size()[1]            
            Kx = torch.exp( -Dxx / variance).type(torch.FloatTensor)   # kernel matrices        
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X,X)
                Kx = torch.exp( -Dxx / (2.*sx*sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))

    ## Adding linear kernel
    elif k_type == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    Kxc = torch.mm(Kx,H)

    return Kxc

def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp( -X / (2.*sigma*sigma))
    return torch.mean(X)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    m = int(x.size()[0])
    H = torch.eye(m) - (1./m) * torch.ones([m,m])
    # H = Variable(H)
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x,y]))
    Dxy = Dxy[:x.size()[0], x.size()[0]:]
    Kxy = torch.exp( -Dxy / (1.*sxy*sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2*torch.mean(Kxy)

    return mmdval

def mmd_pxpy_pxy(x,y,sigma=None,use_cuda=True, to_numpy=False):
    """
    """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    m = int(x.size()[0])

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    A = torch.mean(Kx*Ky)
    B = torch.mean(torch.mean(Kx,dim=0)*torch.mean(Ky, dim=0))
    C = torch.mean(Kx)*torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2*B + C 
    return mmd_pxpy_pxy_val

def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """
    """
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy

def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """
    """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy/(Px*Py)
    return thehsic

def hsic_normalized_cca(x, y, sigma, use_cuda=True, to_numpy=True, k_type_y='gaussian'):
    """
    """
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma, k_type=k_type_y)

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon*m*K_I)
    Kyc_i = torch.inverse(Kyc + epsilon*m*K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy

def hsic(x, y):
    return hsic_normalized_cca(x, y, sigma=5.0, k_type_y='gaussian')

def hsic_objective(hidden, h_target, h_data, sigma, k_type_y='gaussian'):
    hsic_hy_val = hsic_normalized_cca(hidden, h_target, sigma=sigma, k_type_y=k_type_y)
    hsic_hx_val = hsic_normalized_cca(hidden, h_data,   sigma=sigma)
    return hsic_hx_val, hsic_hy_val



class PwoA(nn.Module):
    def __init__(self, prune_features, device='cuda'):
        super().__init__()
        self.lambda_x = 1
        self.lambda_y = 4
        self.device = device

        self.convs = prune_features

        self.distillation_criterion = nn.KLDivLoss()
        self.distillation_temp = 1.
        self.alpha = 0.3


    def loss_H(self, data, target, hidden):
        h_target = target.view(-1,1)
        h_data = data.view(-1, np.prod(data.size()[1:]))

        hx_l_list = []
        hy_l_list = []
        hsic_loss = 0

        for i in range(len(hidden)):
            W = hidden[i]
            if len(W) > 2:
                W = hidden[i].view(-1, np.prod(hidden[i].shape[1:]))
            hy_l = hsic(W, # output after i-th layer
                    h_target.float())
            hx_l = hsic(W, h_data)
            hx_l_list.append(hx_l)
            hy_l_list.append(hy_l)

            temp_hsic = self.lambda_x * hx_l - self.lambda_y * hy_l
            hsic_loss += temp_hsic.to(self.device)
        return hsic_loss
    
    def loss_D(self, output, output_pretrained):
        dis_loss = self.distillation_temp * \
            self.distillation_temp * \
            self.distillation_criterion((output / self.distillation_temp), 
                                        (output_pretrained / self.distillation_temp))
        return  dis_loss

    def forward(self, data, target, output, output_pre, hidden):
        L_h = self.loss_H(data, target, hidden)
        L_d = self.loss_D(output, output_pre)
        return self.alpha * L_d + L_h
        

class ADMM(nn.Module):
    def __init__(self, prune_features, prune_ratio, admm_epochs, rho=0.001, device='cuda'):
        super().__init__()
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}
        self.prune_ratio = prune_ratio
        self.prune_ratios = {}
        self.device = device
        # self.model = model
        self.convs = prune_features
        
        self.sparcity_type = "column"
        self.epochs = admm_epochs

        self.admm_loss = {}
        self.idxs = []

        self.init()
        self.admm_initialization()

    def init(self):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        # setup pruning ratio
        self.prune_ratios = {}
        for idx, layer in enumerate(self.convs):
            name = None
            W = layer.weight
            # for name, weight in layer.named_parameters():
            # print(name, W)
            if (len(W.size()) == 4):
                self.prune_ratios[idx] = self.prune_ratio
        
        # setup rho
        for k, v in self.prune_ratios.items():
            self.rhos[k] = self.rho
        
        # print(self.prune_ratios) 
        
        # initialize aux and dual params
        for idx, layer in enumerate(self.convs):
            name = None
            W = layer.weight
            # print(name, W.shape)
            if idx not in self.prune_ratios:
                continue
            self.ADMM_U[idx] = torch.zeros(W.shape).to(self.device)  # add U
            self.ADMM_Z[idx] = torch.Tensor(W.shape).to(self.device)  # add Z

    def admm_initialization(self):
        cross_x, cross_f = 4, 1
        sparsity_type = self.sparcity_type
        for idx, layer in enumerate(self.convs):
            name = None
            W = layer.weight
            if idx in self.prune_ratios:
                _, updated_Z = self.weight_pruning(
                    W, name, idx, self.prune_ratios[idx], sparsity_type, cross_x,
                    cross_f)  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
                self.ADMM_Z[idx] = updated_Z

    def admm_multi_rho_scheduler(self, name):
        """
        It works better to make rho monotonically increasing
        rho: using 1.1: 
            0.01   ->  50epochs -> 1
            0.0001 -> 100epochs -> 1
            using 1.2:
            0.01   -> 25epochs -> 1
            0.0001 -> 50epochs -> 1
            using 1.3:
            0.001   -> 25epochs -> 1
            using 1.6:
            0.001   -> 16epochs -> 1
        """
        current_rho = self.rhos[name]
        self.rhos[name] = min(1, 1.35*current_rho)  # choose whatever you like

    def z_u_update(self,
                    epoch,
                    batch_idx):
        cross_x, cross_f = 4, 1
        admm_epochs, sparsity_type = self.epochs, self.sparcity_type
        if epoch != 1 and (epoch - 1) % admm_epochs == 0 and batch_idx == 0:
            for idx, layer in enumerate(self.convs):
                name = None
                W = layer.weight
                if idx not in self.prune_ratios:
                    continue
                self.admm_multi_rho_scheduler(idx) # call multi rho scheduler every admm update
                self.ADMM_Z[idx] = W.detach() + self.ADMM_U[idx].detach()  # Z(k+1) = W(k+1)+U[k]
                _, updated_Z = self.weight_pruning(self.ADMM_Z[idx], name, idx, self.prune_ratios[idx], sparsity_type, 
                    cross_x, cross_f)  # equivalent to Euclidean Projection
                self.ADMM_Z[idx] = updated_Z
                self.ADMM_U[idx] = W.detach() - self.ADMM_Z[idx].detach() + self.ADMM_U[idx].detach()  # U(k+1) = W(k+1) - Z(k+1) +U(k)

    def calc_admm_loss(self):
        '''
        append admm loss to cross_entropy loss
        Args:
            args: configuration parameters
            model: instance to the model class
            ce_loss: the cross entropy loss
        Returns:
            ce_loss(tensor scalar): original cross enropy loss
            admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
            ret_loss(scalar): the mixed overall loss
        '''
        # for layer in self.convs:
        #     for i, (name, W) in enumerate(layer.named_parameters()):  ## initialize Z (for both weights and bias)
        for idx, layer in enumerate(self.convs):
            W = layer.weight
            if idx not in self.prune_ratios:
                continue
            self.admm_loss[idx] = 0.5 * self.rhos[idx] * (torch.norm(W - self.ADMM_Z[idx] + self.ADMM_U[idx], p=2)**2)
        mixed_loss = 0
        # mixed_loss += ce_loss
        for _, v in self.admm_loss.items():
            mixed_loss += v
        #print('admm_loss: ', mixed_loss)
        return mixed_loss

    def weight_pruning(self, weight, name, idx, prune_ratio, sparsity_type, cross_x=4, cross_f=1):
        """
        weight pruning [irregular,column,filter]
        Args:
            weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
            prune_ratio (float between 0-1): target sparsity of weights

        Returns:
            mask for nonzero weights used for retraining
            a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

        """
        device = weight.device
        weight = weight.cpu().detach().numpy()
        
        percent = prune_ratio * 100
        # print(percent)
        if (sparsity_type == "irregular"):
            weight_temp = np.abs(
                weight)  # a buffer that holds weights with absolute values
            percentile = np.percentile(weight_temp,
                                    percent)  # get a value for this percentitle
            under_threshold = weight_temp < percentile
            above_threshold = weight_temp > percentile
            above_threshold = above_threshold.astype(
                np.float32
            )  # has to convert bool to float32 for numpy-tensor conversion
            weight[under_threshold] = 0
            return torch.from_numpy(above_threshold).to(device), torch.from_numpy(weight).to(device)

        ####################################

        elif (sparsity_type == "column"):
            # print(weight.shape)
            shape = weight.shape
            weight2d = weight.reshape(shape[0], -1)
            shape2d = weight2d.shape
            # print(shape2d)
            column_l2_norm = np.linalg.norm(weight2d, 2, axis=0)
            percentile = np.percentile(column_l2_norm, percent)
            under_threshold = column_l2_norm < percentile
            above_threshold = column_l2_norm > percentile
            if len(self.idxs) <= idx:
                self.idxs.append((idx, under_threshold))
            else:
                self.idxs[idx] = (idx, under_threshold)
            weight2d[:, under_threshold] = 0
            above_threshold = above_threshold.astype(np.float32)
            expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
            for i in range(shape2d[1]):
                expand_above_threshold[:, i] = above_threshold[i]
            expand_above_threshold = expand_above_threshold.reshape(shape)
            weight = weight.reshape(shape)
            return torch.from_numpy(
                expand_above_threshold).to(device), torch.from_numpy(weight).to(device)

        elif (sparsity_type == "filter"):
            shape = weight.shape
            weight2d = weight.reshape(shape[0], -1)
            shape2d = weight2d.shape
            row_l2_norm = np.linalg.norm(weight2d, 2, axis=1)
            percentile = np.percentile(row_l2_norm, percent)
            under_threshold = row_l2_norm <= percentile
            above_threshold = row_l2_norm > percentile
            weight2d[under_threshold, :] = 0
            # weight2d[weight2d < 1e-40] = 0
            above_threshold = above_threshold.astype(np.float32)
            expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
            for i in range(shape2d[0]):
                expand_above_threshold[i, :] = above_threshold[i]
            weight = weight.reshape(shape)
            expand_above_threshold = expand_above_threshold.reshape(shape)
            return torch.from_numpy(
                expand_above_threshold).to(device), torch.from_numpy(weight).to(device)
        elif (sparsity_type == "bn_filter"):
            ## bn pruning is very similar to bias pruning
            weight_temp = np.abs(weight)
            percentile = np.percentile(weight_temp, percent)
            under_threshold = weight_temp < percentile
            above_threshold = weight_temp > percentile
            above_threshold = above_threshold.astype(
                np.float32
            )  # has to convert bool to float32 for numpy-tensor conversion
            weight[under_threshold] = 0
            return torch.from_numpy(above_threshold).to(device), torch.from_numpy(weight).to(device)
        else:
            raise SyntaxError("Unknown sparsity type")

    def forward(self, epoch, batch_idx):
        self.z_u_update(epoch, batch_idx)
        return self.calc_admm_loss()


if __name__ == "__main__":
    x = torch.randn(size=(2, 5))
    print(x)
    kx_l =  kernelmat(x, sigma=None, k_type='linear')
    kx_g =  kernelmat(x, sigma=None, k_type='gaussian')
    print(kx_l)
    print(kx_g)
