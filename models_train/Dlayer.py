from icecream import ic
from torch import nn
# from torch import FloatTensor, rand, pow
import torch
import numpy as np

class D1Layer(nn.Module):
    def __init__(self, D_in, H, D_out, Activ, num_embeddings, embedding_dim):
        super(D1Layer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.d_in = nn.Sequential(nn.Linear(D_in, H), Activ())
        self.d_h = nn.Sequential(nn.Linear(H, H), Activ())
        self.d_out = nn.Sequential(nn.Linear(H, D_out), Activ())
        
        # uniform_distribution = np.random.uniform(size=(self.num_embeddings, self.embedding_dim))
        # self.embeddings.weight.data = torch.FloatTensor(uniform_distribution)

    def forward(self, x):
        vector = x

        flat_batch = vector.flatten()
        x1 = flat_batch
        ic(x1.shape)
        expend_features = [x1]

        for i in range(2, self.embedding_dim+1):
            exp_val = torch.pow(vector, i).flatten()
            # ic(exp_val.shape)
            # ic(exp_val.flatten().shape)
            expend_features.append(exp_val)
        x_res = torch.stack(tuple(expend_features), 1)
        # ic(x_res)
        # ic(x_res.shape)
        ind = self.get_ind(x_res)
        # ic(ind.shape)
        # ic(ind)
        q = ind.view_as(x.T).T
        q = q.float()
        # ic(q)
        # ic(q.shape)
        emb_val = self.embeddings(ind)
        # ic(emb_val.shape)
        # ic(x_res.shape)
        q_latent_loss = 0.
        if self.training:
            q_latent_loss = torch.nn.functional.mse_loss(emb_val, x_res)
            e_latent_loss = torch.nn.functional.mse_loss(x, q.detach())
            q_latent_loss = q_latent_loss + 0.25 * e_latent_loss
        q = x + (q-x).detach().contiguous()

        # e, eq_loss = self.d_layer(f)
        h1 = self.d_in(q)
        h2 = self.d_h(h1)
        h3 = self.d_h(h2)
        h4 = self.d_h(h3)
        h5 = self.d_h(h4)
        f = self.d_out(h5)

        return f, q_latent_loss
    
    def get_ind(self, flat_x):
        sm = torch.sum(flat_x, dim=1).unsqueeze(1)
        # ic(sm.unsqueeze(1).shape)
        # ic(sm.unsqueeze(0).shape)
        # ic(sm.shape)
        emb = torch.sum(self.embeddings.weight**2, dim=1).unsqueeze(0)
        # ic(self.embeddings.weight.shape)
        # ic(emb.shape)
        dist = sm + emb - 2. * torch.matmul(flat_x, self.embeddings.weight.t())
        # ic(dist.shape)
        encoding_ind = torch.argmin(dist, dim=1)
        return encoding_ind


class D2Layer(nn.Module):
    def __init__(self, D_in, H, D_out, Activ, levels=25):
        super(D2Layer, self).__init__()
        self.Activ = Activ
        self.levels = levels
        self.l_in = nn.Sequential(nn.Linear(D_in, H), self.Activ())
        self.l_h = nn.Sequential(nn.Linear(H, H), self.Activ())
        self.l_out = nn.Sequential(nn.Linear(H, D_out), self.Activ())

    def get_ew_bins(self, x, levels=5):
        b_min, b_max = x.min(axis=0)[0], x.max(axis=0)[0]
        ew_bins = {}
        columns = ['column' + str(i) for i in range(x.size()[1])]
        for i, col in enumerate(columns):
            ew_bins[col] = torch.tensor(np.linspace(b_min[i].cpu().detach().numpy(), b_max[i].cpu().detach().numpy(), num=levels + 1)[1:-1].tolist()).cuda()
        return ew_bins

    def get_ef_bins(self, x, levels = 5):
        ef_bins = {}
        columns = ['column'+ str(i) for i in range(x.size()[1])]
        percentile_centroids = torch.quantile(x, torch.linspace(0, 100, levels + 1).cuda()[1:-1]/100, dim=0).T
        # ic(percentile_centroids)
        for i, col in enumerate(columns):
            ef_bins[col] = percentile_centroids[i]
        # ic(ef_bins)
        return ef_bins

    def discretize_by_bins(self, x, all_bins: dict):
        X_discre = []
        for i, (col, bins) in enumerate(all_bins.items()):
            _x = x[:, i]
            ic(_x)
            ic(bins)
            _x_discre = torch.bucketize(_x, bins)
            X_discre.append(_x_discre.reshape(-1, 1))
        X_discre = torch.hstack(X_discre)
        return torch.tensor(X_discre, dtype=torch.float32, requires_grad=True)

    def descritization(self, x, levels=5):
        bins = self.get_ew_bins(x, levels)
        dis = self.discretize_by_bins(x, bins)
        ic(dis)
        return dis
    
    def forward(self, x):
        h1 = self.l_in(self.descritization(x, self.levels))
        h2 = self.l_h(h1)
        h3 = self.l_h(h2)
        h4 = self.l_h(h3)
        h5 = self.l_h(h4)
        h6 = self.l_h(h5)
        y = self.l_out(h6)
        d2_loss = 0
        return y, d2_loss


if __name__ =="__main__":
    x = torch.rand(4, 4)
    # ic(x)
    d = D1Layer(5, 8)
    print(d(x))
        