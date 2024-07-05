from icecream import ic
from torch import nn
# from torch import FloatTensor, rand, pow
import torch
import numpy as np

class D1Layer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(D1Layer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
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
        ic(x_res.shape)
        ind = self.get_ind(x_res)
        ic(ind.shape)
        ic(ind)
        q = ind.view_as(x.T).T
        ic(q)
        ic(q.shape)
        emb_val = self.embeddings(ind)
        ic(emb_val.shape)
        ic(x_res.shape)
        q_latent_loss = 0
        if self.training:
            q_latent_loss = torch.nn.functional.mse_loss(emb_val, x_res)
            e_latent_loss = torch.nn.functional.mse_loss(x, q.detach())
            q_latent_loss = q_latent_loss + 0.25 * e_latent_loss
        q = x + (q-x).detach().contiguous()

        return q, q_latent_loss
    
    def get_ind(self, flat_x):
        sm = torch.sum(flat_x, dim=1).unsqueeze(1)
        # ic(sm.unsqueeze(1).shape)
        # ic(sm.unsqueeze(0).shape)
        ic(sm.shape)
        emb = torch.sum(self.embeddings.weight**2, dim=1).unsqueeze(0)
        ic(self.embeddings.weight.shape)
        ic(emb.shape)
        dist = sm + emb - 2. * torch.matmul(flat_x, self.embeddings.weight.t())
        ic(dist.shape)
        encoding_ind = torch.argmin(dist, dim=1)
        return encoding_ind

if __name__ =="__main__":
    x = torch.rand(4, 4)
    # ic(x)
    d = D1Layer(5, 8)
    print(d(x))
        