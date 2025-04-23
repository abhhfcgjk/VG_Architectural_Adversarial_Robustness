import torch
from torch import nn
import torch.nn.functional as F

from modules import (
    AdaptiveEmbedding,
    DecoderLayer, 
    LogUniformSampler, 
    PositionalEmbedding,
    PositionwiseFF
)
from utils import (
    LogUniformSampler, 
    sample_logits,
    ProjectedAdaptiveLogSoftmax
)

class MemTransformerLM(nn.Module):
    def __init__(self,
                 n_token,
                 n_layer,
                 n_head,
                 d_model,
                 d_head,
                 d_inner,
                 dropout,
                 dropatt,
                 huber_a=0.2,
                 loss_type='huber',
                 num_blocks=3,
                 tie_weight=True,
                 d_embed=None,
                 div_val=1,
                 tie_projs=[False],
                 pre_lnorm=False,
                 tgt_len=None,
                 ext_len=None,
                 mem_len=None,
                 cutoffs=[],
                 adapt_inp=False,
                 same_length=False,
                 attn_type=0,
                 clamp_len=-1,
                 sample_softmax=-1,
                 proj_dim=256,  # for performer layers
                 n_roll=3,  # mirrored attention
                 skip_attn_normalization=False,
                 no_pos=False,  # no positional encoding
                 device='cuda',
                 update_mode='hard',
                 scale_w = 1.,
                 two_proj_matrix = False,
                 learn_scale_w = False):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        self.no_pos = no_pos

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(
            n_token, d_embed, d_model, cutoffs, div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                DecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm,
                    attn_type=attn_type, huber_a=huber_a, loss_type=loss_type,
                    num_blocks=num_blocks)
            )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, 
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None, carry_over_fast_weight=False):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)
        if carry_over_fast_weight:
            mlen = 0
        else:
            mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)
                             ).bool()[:, :, None]
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen),
                diagonal=1+mlen).bool()[:, :, None]

        hids = []

        if self.no_pos:
            core_out = self.drop(word_emb)
        else:
            pos_seq = torch.arange(klen-1, -1, -1.0,
                                    device=word_emb.device,
                                    dtype=word_emb.dtype)
            # pos_seq = torch.arange(0, klen, device=word_emb.device,
            #                        dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            core_out = self.drop(word_emb + pos_emb[-qlen:])

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            if mems_i is not None and i == 0:
                mems_i += pos_emb[:mlen]
            if self.attn_type == 5:
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                    mems=mems_i, redraw=self.training)
            else:
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                    mems=mems_i)
            hids.append(core_out)

        core_out = self.drop(core_out)
        if not carry_over_fast_weight:
            new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems, softmax_keep_order=False,
                carry_over_fast_weight=False):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(
            data, mems=mems, carry_over_fast_weight=carry_over_fast_weight)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.reshape(-1, pred_hid.size(-1)),
                             target.reshape(-1),
                             keep_order=softmax_keep_order)
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems

