import torch
import torch.nn as nn
from einops import rearrange
import numbers
import torch.nn.functional as F

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class MeMA(nn.Module):
    def __init__(self,task_list,channel,num_heads):
        super(MeMA, self).__init__()
        self.c = channel
        self.num_heads = num_heads
        self.mem = nn.Embedding(len(task_list)+1, self.c*256*self.num_heads)
        self.task_list = task_list
        self.proj_k2g = nn.Linear(self.c,self.c*self.num_heads,bias = False)
        self.proj_v2g = nn.Linear(self.c,self.c*self.num_heads,bias = False)
        self.proj_k2t = nn.Linear(self.c,self.c*self.num_heads,bias = False)
        self.proj_v2t = nn.Linear(self.c,self.c*self.num_heads,bias = False)
        self.project_out = nn.Linear(self.c*self.num_heads, self.c)
        self.laynorm = torch.nn.LayerNorm(self.c, elementwise_affine = False)
        nn.init.kaiming_normal_(self.mem.weight)

    def forward(self, feature, tasks):
        b,c,h,w = feature.size()
        task_ids = torch.tensor([self.task_list.index(task) for task in tasks], dtype=torch.long).to(torch.device('cuda'))
        global_ids = torch.tensor([len(self.task_list) for _ in tasks], dtype=torch.long).to(torch.device('cuda'))
        task_query = self.mem(task_ids).view(b,self.num_heads,256,self.c)
        global_query = self.mem(global_ids).view(b,self.num_heads,256,self.c)
        feature_flat = feature.view(feature.size(0), feature.size(1), -1).permute(0, 2, 1)

        # global
        k2g = self.proj_k2g(feature_flat).view(b, h*w, self.num_heads, c).permute(0, 2, 3, 1)
        v2g = self.proj_v2g(feature_flat).view(b, h*w, self.num_heads, c).permute(0, 2, 1, 3)
        global_attn = torch.matmul(global_query, k2g)
        global_attn = F.softmax(global_attn, dim=-1)
        global_output = torch.matmul(global_attn, v2g).permute(0, 2, 1, 3).contiguous().view(b, h*w, -1)

        #task
        k2t = self.proj_k2t(feature_flat).view(b, h*w, self.num_heads, c).permute(0, 2, 3, 1)
        v2t = self.proj_v2t(feature_flat).view(b, h*w, self.num_heads, c).permute(0, 2, 1, 3)
        task_attn = torch.matmul(task_query, k2t)
        task_attn = F.softmax(task_attn, dim=-1)
        task_output = torch.matmul(task_attn, v2t).permute(0, 2, 1, 3).contiguous().view(b, h*w, -1)
        out = self.laynorm(self.project_out(global_output+task_output)).permute(0,2,1).view(b,c,h,w)

        return out
    

def build_MeMA(args):
    return MeMA(
        task_list=args.tasks,
        channel=args.swin_enc_embed_dim * 8,
        num_heads=6
    )