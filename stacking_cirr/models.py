import torch
from torch import nn
import random
import numpy as np
    
class LightDW(nn.Module):
    def __init__(self, in_ch=4, exp=8, out_ch=4, kernel_size=3):
        super().__init__()
        hidden = in_ch * exp                    
        pad = (kernel_size - 1) // 2

        self.pw1 = nn.Conv1d(in_ch, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = nn.ReLU6(inplace=True)

        self.dw = nn.Conv1d(hidden, hidden, kernel_size,
                            padding=pad, groups=hidden, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.pw2 = nn.Conv1d(hidden, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch)

        self.use_res = (in_ch == out_ch)
    
    def forward(self, x):
        out = self.pw1(x)          
        out = self.bn1(out)       
        out = self.act(out)        

        out = self.dw(out)         
        out = self.bn2(out)        
        out = self.act(out)        

        out = self.pw2(out)        
        out = self.bn3(out)        

        if self.use_res:
            out = out + x          

        return out                 

class LightFusion(nn.Module):
    def __init__(self, n_algo=4):
        super().__init__()
        self.backbone = LightDW(in_ch=4, exp=8, out_ch=4, kernel_size=3)
        self.w = nn.Conv1d(4, 1, 1, bias=False)

    def forward(self, x):
        x = self.backbone(x)      
        x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-8)
        x = self.w(x).squeeze(1)  
        return x


class CIRPlus(nn.Module):
    def __init__(self, dataset, tau=0.01, device=torch.device('cpu')):
        super().__init__()

        self.dataset = dataset
        self.device = device
        random.seed(42)        
        np.random.seed(42)     
        torch.manual_seed(42)
        self.net = LightFusion().to(device)
        with torch.no_grad():
            self.net.backbone.pw2.weight.fill_(1e-9)
            self.net.w.weight[:, :, 0] = torch.tensor([1.0, 1.0, 1.0, 2.0])
            self.net.backbone.pw2.weight.requires_grad = False
        
        self.tau = tau
        self.crossentropy_criterion = nn.CrossEntropyLoss()
    
    def save(self):
        torch.save(self.net, f'stacking_{self.dataset}/model/{self.dataset}.pt')

    def load_ckpt(self):
        self.net = torch.load(f'stacking_{self.dataset}/model/{self.dataset}.pt', map_location='cpu', weights_only=False)

    def extract_bank_features(self):
        self.clip_feature = torch.load(f'feature_{self.dataset}/clip/train_feature.pt')
        self.tgcir_feature = torch.load(f'feature_{self.dataset}/tgcir/train_feature.pt')
        self.blip_feature = torch.load(f'feature_{self.dataset}/blip/train_feature.pt')
        self.blip2_feature = torch.load(f'feature_{self.dataset}/blip2/train_feature.pt')

    def bank_large_step(self, loss, indexs, target_indexs,
                        refer_indexs=None):
        clip_fea = self.clip_feature[refer_indexs].detach().to(self.device)
        tgcir_fea = self.tgcir_feature[refer_indexs].detach().to(self.device)
        blip_fea = self.blip_feature[refer_indexs].detach().to(self.device)
        blip2_fea = self.blip2_feature[refer_indexs].detach().to(self.device)
        query_feats = torch.stack([clip_fea, tgcir_fea, blip_fea, blip2_fea], dim=1)
        target_neg_loss = self.infonce_loss(query_feats, target_indexs, tau=self.tau)
        loss['bank_loss'] = target_neg_loss

    def forward(self, text, indexs, target_indexs, refer_indexs, refer_image=None, target_image=None):
        loss = dict()
        self.bank_large_step(loss, indexs, target_indexs, refer_indexs)
        return loss

    def infonce_loss(self, query_features, labels=None, tau=0.01):
        logits = self.net(query_features) / tau              
        loss = self.crossentropy_criterion(logits, labels)
        self.tau = self.tau * (1 - loss.item()/1e6)
        return loss
