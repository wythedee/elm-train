"""Current fixed-scale 80-token Projector; same equations as the server debug baseline."""
import torch
from torch import nn
from torch.nn import functional as F
class EEGProjector80(nn.Module):
 def __init__(self,hidden=2048):
  super().__init__();self.hidden=hidden;self.output_scale=.08
  self.label_encoder=nn.Sequential(nn.LayerNorm(7500),nn.Linear(7500,hidden,bias=False),nn.GELU())
  self.label_out=nn.Linear(hidden,32*2560,bias=False)
  self.stat_encoder=nn.Sequential(nn.LayerNorm(7500),nn.Linear(7500,hidden,bias=False),nn.GELU())
  self.field_embedding=nn.Parameter(torch.randn(6,hidden)*.02);self.field_scale=nn.Parameter(torch.randn(6,hidden)*.1)
  self.stat_norm=nn.LayerNorm(hidden);self.stat_out=nn.Linear(hidden,8*2560,bias=False)
  self.output_norm=nn.LayerNorm(2560);self.ffn_norm=nn.LayerNorm(2560)
  self.ffn=nn.Sequential(nn.Linear(2560,2560),nn.GELU(),nn.Linear(2560,2560))
 def forward(self,label_bank,label_valid,stat_bank,stat_valid):
  assert label_bank.shape[1:]==stat_bank.shape[1:]==(1,150,50)
  label=(label_bank.float()*label_valid[:,None,:,None].float()).flatten(1)
  stat=(stat_bank.float()*stat_valid[:,None,:,None].float()).flatten(1)
  label=self.label_out(self.label_encoder(label)).reshape(len(label),32,2560)
  h=self.stat_encoder(stat)
  c=self.stat_norm(F.gelu(h[:,None]*(1+torch.tanh(self.field_scale))[None]+self.field_embedding[None]))
  stat=self.stat_out(c).reshape(len(h),48,2560)
  z=self.output_norm(torch.cat([label,stat],1))
  return self.output_scale*(z+self.ffn(self.ffn_norm(z)))
def token_alignment(pred,target):
 target=target.detach().float();pred=pred.float()
 assert pred.shape==target.shape and pred.shape[1:]==(80,2560)
 mse=((pred-target).square().mean(-1)/target.square().mean(-1).clamp_min(1e-8)).mean()
 cosine=(1-F.cosine_similarity(pred,target,dim=-1)).mean()
 return mse+.1*cosine
