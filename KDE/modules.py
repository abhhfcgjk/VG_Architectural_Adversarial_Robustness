import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

class RBFMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, huber_a=0.2, loss_type='huber',
                 dropatt=0, pre_lnorm=False):
        super(RBFMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.huber_a = huber_a
        self.loss_type = loss_type

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
    
    def compute_weights(self, w_init, x, attn_mask=None):
        obj_1 = torch.ones((x.size(0), x.size(1), x.size(2)), device='cuda:'+str(torch.cuda.current_device()))
        diff_norm = torch.square(torch.cdist(x.transpose(0, 2), x.transpose(0, 2), p=2.0)).permute(2,3,1,0)
        if attn_mask is not None and attn_mask.any().item():
            diff_norm.masked_fill_(attn_mask, float('inf'))
        rbf = torch.exp(-self.scale * diff_norm)
        obj_2 = torch.einsum('ij,ijbn->ibn', (w_init, rbf))
        obj_3 = torch.einsum('ij,jbn->ibn', (w_init, torch.einsum('ij,ijbn->ibn', (w_init, rbf))))

        h_norm_2 = obj_1 + obj_2 + obj_3
        h_norm = torch.sqrt(h_norm_2)
        h_norm = h_norm.to(device=torch.device(f'cuda:{torch.cuda.current_device()}'), dtype=torch.float64)
        if self.loss_type == 'huber':
            h_norm_robust = torch.where(h_norm <= self.huber_a, 1., self.huber_a/h_norm).to(dtype=torch.float32)
        elif self.loss_type == 'hampel':
            h_b, h_c = 2 * self.huber_a, 3 * self.huber_a
            cond_1 = h_norm < self.huber_a
            cond_2 = ((h_norm >= self.huber_a) & (h_norm < h_b))
            cond_3 = ((h_norm >= h_b) & (h_norm < h_c))
            cond_4 = h_norm >= h_c
            h_norm_1 = torch.where(cond_1, 1., h_norm)
            h_norm_2 = torch.where(cond_2, self.huber_a/h_norm_1, h_norm_1)
            h_norm_3 = torch.where(cond_3, ((self.huber_a * (h_c - h_norm_2))/((h_c - h_b) * h_norm_2)), h_norm_2)
            h_norm_robust = torch.where(cond_4, 0.01, h_norm_3).to(dtype=torch.float32)
        log_norm = torch.log(h_norm_robust)
        w = log_norm - torch.logsumexp(log_norm, dim=0, keepdim=True)
        return w

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        # if attn_mask is None:
            # attn_mask = torch.triu(
            #     torch.ones_like(h), 
            #     diagonal=1
            #     ).bool()
            # attn_mask = torch.ones((c.size(0),c.size(1), self.n_head
            #                         )).bool().to(device='cuda:'+str(torch.cuda.current_device()))
        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        # divide the output into 2 parts, which are k and v
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)
        import logging
        logging.getLogger().setLevel(logging.INFO)
        
        # here compute approx attention
        head_k = F.normalize(head_k, dim=-1)
        diff_norm = torch.square(torch.cdist(head_q.transpose(0, 2), head_k.transpose(0, 2), p=2.0)).permute(2,3,1,0)
        weights = F.normalize(torch.ones((c.size(0), c.size(0))).type(torch.FloatTensor), dim=1, p=1.0)
        # weights = torch.rand((c.size(0), c.size(0)))
        weights = weights.to(device='cuda:'+str(torch.cuda.current_device()))
        head_kv = torch.cat((head_k, head_v), -1)
        kv_weights = self.compute_weights(weights, head_kv, attn_mask=attn_mask)[None, :, :, :]
        k_weights = self.compute_weights(weights, head_k, attn_mask=attn_mask)[None, :, :, :]

        if attn_mask is not None and attn_mask.any().item():
            diff_norm.masked_fill_(attn_mask, float('inf'))
        scaled_norm = -self.scale * diff_norm
        
        # log_weights = torch.log(weights)[None, :, :, :]
        log_result = kv_weights + scaled_norm - torch.logsumexp(k_weights + scaled_norm, dim=1, keepdim=True)
        
        attn_prob = torch.exp(log_result)
        
        
        # rbf = torch.log(torch.exp(scaled_norm))
        # kv_dist = torch.log(torch.mul(weights[None, :, :, :], rbf))
        # k_dist = torch.log(torch.mul(weights[None, :, :, :], rbf))
        # attn_prob = torch.exp(scaled_norm - torch.logsumexp(scaled_norm, dim=1, keepdim=True))
        attn_prob = self.dropatt(attn_prob)
        
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)
        logging.debug(output)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, attn_type, huber_a=0.2, loss_type='huber',
                 update_mode = 'rbf2keys', scale_w = 1., num_blocks=3, **kwargs):
        super(DecoderLayer, self).__init__()
        attn_func = RBFMultiHeadAttn
        self.dec_attn = attn_func(n_head, d_model, d_head, dropout, huber_a, loss_type, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))
        self.attn_type = attn_type

    def forward(self, *dec_inp, dec_attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        output = self.dec_attn(*dec_inp, 
                               attn_mask=dec_attn_mask, 
                               mems=mems, 
                               carry_over_fast_weight=carry_over_fast_weight)
        if carry_over_fast_weight:
            output, new_mem = output
        output = self.pos_ff(output)
        if carry_over_fast_weight:
            return output, new_mem
        return output

class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)
        # if math.isnan(torch.mean(embed).item()):
        #     pdb.set_trace()
        embed.mul_(self.emb_scale)

        return embed
