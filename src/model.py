import torch
from torch import nn
import open3d as o3d
import numpy as np

class SDF(nn.Module):
    def __init__(self):
        super(SDF, self).__init__()
        fc_1 = FC(3, 32, True, 'relu')
        fc_2 = FC(32, 64, True, 'relu')
        fc_3 = FC(64, 128, True, 'relu')
        fc_4 = FC(128, 256, True, 'relu')
        fc_5 = FC(256, 512, True, 'relu')
        fc_6 = FC(512, 128, True, 'relu')
        fc_7 = FC(128, 1, False, 'none')
        self.fc = nn.Sequential(fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7)
    def forward(self, sdf_q):
        sdist_out = self.fc(sdf_q).squeeze(-1)
        return sdist_out

class StoneSDF(nn.Module):
    def __init__(self):
        super(StoneSDF, self).__init__()
        fc_1 = FC(3, 32, True, 'relu')
        fc_2 = FC(32, 64, True, 'relu')
        fc_3 = FC(64, 128, True, 'relu')
        fc_4 = FC(128, 256, True, 'relu')
        fc_5 = FC(256, 512, True, 'relu')
        fc_6 = FC(512, 128, True, 'relu')
        fc_7 = FC(128, 1, False, 'none')
        self.fc = nn.Sequential(fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7)
    def forward(self, sdf_q):
        sdist_out = self.fc(sdf_q).squeeze(-1)
        return sdist_out

class Decoder(nn.Module):
    def __init__(self, D=128, bn=True, activate="relu", bias=False):
        super(Decoder, self).__init__()
        fc_5 = FC(D, 64, bn, activate, bias=bias)
        fc_6 = FC(64, 32, bn, activate, bias=bias)
        fc_7 = FC(32, 1, False, 'none', bias=False)
        self.fc = nn.Sequential(fc_5, fc_6, fc_7)
        
    def forward(self, sdf_q):
        sdist_out = self.fc(sdf_q).squeeze(-1)#输入需要normalization
        return sdist_out

class transSDF(nn.Module):
    def __init__(self, D): #D是256，num_pp=128
        super(transSDF, self).__init__()
        
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))#embedding layer
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            dim_feedforward=D,
            nhead=8,
            dropout=0.1,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder=Decoder(D)

    def forward(self, sdf_q):
        #qs, eq: bsz x 3
        #lines: bsz x trj_len x 3 
        #sdf_q.requires_grad=True
        min_vals,_=torch.min(sdf_q,dim=0)
        max_vals,_=torch.max(sdf_q,dim=0)
        n_sdf_q=-1+2*(sdf_q-min_vals)/(max_vals-min_vals)
        e_sdf = self.lifting(n_sdf_q)
        e_out = self.transformer(e_sdf)
        sdf_out=self.decoder(e_out)
        return sdf_out

class finalSDF(nn.Module):
    def __init__(self, D=128): #D是256，num_pp=128
        super(finalSDF, self).__init__()
        
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))#embedding layer
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            dim_feedforward=D,
            nhead=8,
            dropout=0.1,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.decoder=Decoder()
        
    @property
    def device(self):
        return next(self.parameters()).device
            
    def forward(self, sdf_q):
        #qs, eq: bsz x 3
        #lines: bsz x trj_len x 3 
        #sdf_q.requires_grad=True
        e_sdf = self.lifting(sdf_q)
        e_out = self.transformer(e_sdf)
        sdf_out=self.decoder(e_out)
        return sdf_out
    
    
#from: https://github.com/osiriszjq/complex_encoding/blob/main/3D_separable_points_GD.py#L58C21-L66
class encoding_func_3D:
    def __init__(self, name, param=None):
        self.name = name
        if name == 'none': self.dim=2
        elif name == 'basic': self.dim=4
        else:
            self.dim = param[1]
            if name == 'RFF':
                self.b = param[0]*torch.randn((int(param[1]/2),3))
            elif name == 'rffb':
                self.b = param[0]
            elif name == 'Linf':
                self.b = torch.linspace(2.**0., 2.**param[0], steps=int(param[1]/6)).reshape(-1,1)
            elif name == 'Logf':
                self.b = 2.**torch.linspace(0., param[0], steps=int(param[1]/6)).reshape(-1,1)
            elif name == 'Gau':
                self.dic = torch.linspace(0., 1, steps=int(param[1]/3)+1)[:-1].reshape(1,-1)
                self.sig = param[0]
            elif name == 'Tri':
                self.dic = torch.linspace(0., 1, steps=(param[1] // 3) + 2 )[:-1].reshape(1,-1)
                if param[0] is None: self.d = 1/param[1]
                else: self.d = param[0]
            else:
                print('Undifined encoding!')
        
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'basic':
            emb = torch.cat((torch.sin((2.*np.pi*x)),torch.cos((2.*np.pi*x))),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'RFF')|(self.name == 'rffb'):
            emb = torch.cat((torch.sin((2.*np.pi*x) @ self.b.T),torch.cos((2.*np.pi*x) @ self.b.T)),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'Linf')|(self.name == 'Logf'):
            emb1 = torch.cat((torch.sin((2.*np.pi*x[...,:1]) @ self.b.T),torch.cos((2.*np.pi*x[...,:1]) @ self.b.T)),1)
            emb2 = torch.cat((torch.sin((2.*np.pi*x[...,1:2]) @ self.b.T),torch.cos((2.*np.pi*x[...,1:2]) @ self.b.T)),1)
            emb3 = torch.cat((torch.sin((2.*np.pi*x[...,2:3]) @ self.b.T),torch.cos((2.*np.pi*x[...,2:3]) @ self.b.T)),1)
            emb = torch.cat([emb1,emb2,emb3],-1)
            emb = emb/(emb.norm(dim=-1).max())
            return emb
        elif self.name == 'Gau':
            emb1 = (-0.5*(x[...,:1]-self.dic)**2/(self.sig**2)).exp()
            emb2 = (-0.5*(x[...,1:2]-self.dic)**2/(self.sig**2)).exp()
            emb3 = (-0.5*(x[...,2:3]-self.dic)**2/(self.sig**2)).exp()
            emb = torch.cat([emb1,emb2,emb3],-1)
            emb = emb/(emb.norm(dim=-1).max())
            return emb
        elif self.name == 'Tri':
            self.dic = self.dic.to(x.device)
            emb1 = (1-(x[...,:1]-self.dic).abs()/self.d)
            emb1 = emb1*(emb1>0)
            emb2 = (1-(x[...,1:2]-self.dic).abs()/self.d)
            emb2 = emb2*(emb2>0)
            emb3 = (1-(x[...,2:3]-self.dic).abs()/self.d)
            emb3 = emb3*(emb3>0)
            emb = torch.cat([emb1,emb2,emb3],-1)
            emb = emb/(emb.norm(dim=-1).max())
            return emb[..., :self.dim]

class encoding_func_4D:
    def __init__(self, name, param=None):
        self.name = name
        in_dim = 4
        if name == 'none': self.dim=2
        elif name == 'basic': self.dim=4
        else:
            self.dim = param[1]
            if name == 'RFF':
                self.b = param[0]*torch.randn((int(param[1]/2),in_dim))
            elif name == 'rffb':
                self.b = param[0]
            elif name == 'Linf':
                self.b = torch.linspace(2.**0., 2.**param[0], steps=int(param[1]/in_dim * 2)).reshape(-1,1)
            elif name == 'Logf':
                self.b = 2.**torch.linspace(0., param[0], steps=int(param[1]/in_dim * 2)).reshape(-1,1)
            elif name == 'Gau':
                self.dic = torch.linspace(0., 1, steps=int(param[1]/in_dim)+1)[:-1].reshape(1,-1)
                self.sig = param[0]
            elif name == 'Tri':
                self.dic = torch.linspace(0., 1, steps=(param[1] // in_dim) + 2 )[:-1].reshape(1,-1)
                if param[0] is None: self.d = 1/param[1]
                else: self.d = param[0]
            else:
                print('Undifined encoding!')
        
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'basic':
            emb = torch.cat((torch.sin((2.*np.pi*x)),torch.cos((2.*np.pi*x))),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'RFF')|(self.name == 'rffb'):
            emb = torch.cat((torch.sin((2.*np.pi*x) @ self.b.T),torch.cos((2.*np.pi*x) @ self.b.T)),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'Linf')|(self.name == 'Logf'):
            emb1 = torch.cat((torch.sin((2.*np.pi*x[...,:1]) @ self.b.T),torch.cos((2.*np.pi*x[...,:1]) @ self.b.T)),1)
            emb2 = torch.cat((torch.sin((2.*np.pi*x[...,1:2]) @ self.b.T),torch.cos((2.*np.pi*x[...,1:2]) @ self.b.T)),1)
            emb3 = torch.cat((torch.sin((2.*np.pi*x[...,2:3]) @ self.b.T),torch.cos((2.*np.pi*x[...,2:3]) @ self.b.T)),1)
            emb4 = torch.cat((torch.sin((2.*np.pi*x[...,3:4]) @ self.b.T),torch.cos((2.*np.pi*x[...,3:4]) @ self.b.T)),1)
            emb = torch.cat([emb1,emb2,emb3,emb4],-1)
            emb = emb/(emb.norm(dim=-1).max())
            return emb
        elif self.name == 'Gau':
            emb1 = (-0.5*(x[...,:1]-self.dic)**2/(self.sig**2)).exp()
            emb2 = (-0.5*(x[...,1:2]-self.dic)**2/(self.sig**2)).exp()
            emb3 = (-0.5*(x[...,2:3]-self.dic)**2/(self.sig**2)).exp()
            emb3 = (-0.5*(x[...,3:4]-self.dic)**2/(self.sig**2)).exp()
            emb = torch.cat([emb1,emb2,emb3,emb4],-1)
            emb = emb/(emb.norm(dim=-1).max())
            return emb
        elif self.name == 'Tri':
            self.dic = self.dic.to(x.device)
            emb1 = (1-(x[...,:1]-self.dic).abs()/self.d)
            emb1 = emb1*(emb1>0)
            emb2 = (1-(x[...,1:2]-self.dic).abs()/self.d)
            emb2 = emb2*(emb2>0)
            emb3 = (1-(x[...,2:3]-self.dic).abs()/self.d)
            emb3 = emb3*(emb3>0)
            emb4 = (1-(x[...,3:4]-self.dic).abs()/self.d)
            emb4 = emb4*(emb4>0)
            emb = torch.cat([emb1,emb2,emb3,emb4],-1)
            emb = emb/(emb.norm(dim=-1).max())
            print(emb.shape)
            return emb[..., :self.dim]


class SimpleSDF(nn.Module):
    def __init__(self, D=128, num_layers=5): #D是256，num_pp=128
        super().__init__()
        
        # self.pos_emb = encoding_func_3D('Tri', [3/64, D])
        self.pos_emb = encoding_func_3D('Tri', [None, D])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            dim_feedforward=D,
            nhead=8,
            dropout=0.1,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.proj = nn.Linear(D, 1, bias=True)
        self.proj = Decoder(
                D,
                bn=False, activate="leaky_relu", bias=True
            )

    def forward(self, sdf_q):
        #qs, eq: bsz x 3
        #lines: bsz x trj_len x 3 
        #sdf_q.requires_grad=True
        e_sdf = self.pos_emb(sdf_q)
        e_out = self.transformer(e_sdf)
        sdf_out=self.proj(e_out).squeeze(-1)
        return sdf_out


class simSDF(nn.Module):
    def __init__(self, D): #D是256，num_pp=128
        super(simSDF, self).__init__()
        
        self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))#embedding layer
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            dim_feedforward=D,
            nhead=8,
            dropout=0.1,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder=nn.Linear(D,1)
    def forward(self, sdf_q):
        #qs, eq: bsz x 3
        #lines: bsz x trj_len x 3 
        #sdf_q.requires_grad=True
        e_sdf = self.lifting(sdf_q)
        e_out = self.transformer(e_sdf)
        sdf_out=self.decoder(e_out)
        return sdf_out.squeeze(-1)

class DecoderResCat(nn.Module):
    def __init__(self,  in_features,hidden_size, out_features=3):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size, True, 'relu')
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class coordinator(nn.Module):
    def __init__(self):
        super(coordinator, self).__init__()
        fc_1 = FC(3, 32, True, 'relu')
        fc_2 = FC(32, 64, True, 'relu')
        fc_3 = FC(64, 128, True, 'relu')
        fc_4 = FC(128, 256, True, 'relu')
        fc_5 = FC(256, 512, True, 'relu')
        fc_6 = FC(512, 1024, True, 'relu')
        fc_7 = FC(1024, 512, True, 'relu')
        fc_8 = FC(512, 128, True, 'relu')
        fc_9 = FC(128, 3, False, 'none')
        self.fc = nn.Sequential(fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7,fc_8,fc_9)
    def forward(self, sdf_q):
        sdist_out = self.fc(sdf_q)
        return sdist_out


class MLPs(nn.Module):
    def __init__(self): 
        super(MLPs, self).__init__()
        mlp_1 = MLP(3, 32, True, 'relu')
        mlp_2 = MLP(32, 64, True, 'relu')
        mlp_3 = MLP(64, 128, True, 'relu')
        mlp_4 = MLP(128, 64, True, 'relu')
        mlp_5 = MLP(64, 32, True, 'relu')
        mlp_6 = MLP(32, 1, False, 'none')
        self.spf_head = nn.Sequential(mlp_1, mlp_2, mlp_3, mlp_4,mlp_5,mlp_6)
    def forward(self, lines):
        sdf = self.spf_head(lines).squeeze(-1)
        return sdf
    
class FC(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, bias=False):
        super(FC, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'tanh', 'sigmoid', 'leaky_relu']
        self.is_bn = is_bn
        self.nl = nl
        self.linear = nn.Linear(ic, oc, bias=bias)
        if self.is_bn:
            self.bn = nn.BatchNorm1d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leaky_relu':
            self.activate = nn.LeakyReLU(inplace=True)            
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [batch_size, ic]
        y = self.linear(x) # [batch_size, oc]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y # [batch_size, oc]
    
class MLP(nn.Module):
    def __init__(self, ic, oc, is_bn, nl):
        super(MLP, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'tanh', 'sigmoid']
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [batch_size, num_points, ic]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1) # [batch_size, ic, num_points, 1]
        y = self.conv(x) # [batch_size, oc, num_points, 1]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)   
        y = y.squeeze(-1).permute(0, 2, 1).contiguous() 
        return y # [batch_size, num_points, oc]

def get_gt_sdf(scene,point):
    query_point=o3d.core.Tensor(point.cpu().detach().numpy(),dtype=o3d.core.Dtype.Float32) #输入的点一定要是二维的
    sdf=scene.compute_signed_distance(query_point)
    return sdf