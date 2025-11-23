import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
import random

from .init_func import bn_init, conv_branch_init, conv_init

EPS = 1e-4

class stlgcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 A1,
                 A2,
                 A3,
                 c=1,
                 ratio=0.25,
                 ctr='T',
                 ada='T'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.ratio = ratio
        self.c = c

        #
        self.split_1 = [0, 1, 2, 3, 20]  # 1, 2, 3, 4, 21
        self.split_2 = [4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24]  # 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24, 25
        self.split_3 = [12, 13, 14, 15, 16, 17, 18, 19]  # 13, 14, 15, 16, 17, 18, 19, 20

        self.x = stlgcn_child(self.in_channels, self.out_channels, self.A, self.c, self.ratio)
        self.x5 = stlgcn_child(self.in_channels, self.out_channels, self.A1, self.c, self.ratio)
        self.x12 = stlgcn_child(self.in_channels, self.out_channels, self.A2, self.c, self.ratio)
        self.x8 = stlgcn_child(self.in_channels, self.out_channels, self.A3, self.c, self.ratio)


    def forward(self, x, A=None):
        n, c, t, v = x.shape  # (16,3,100,25)

        depth = self.c
        if depth != 5 and depth != 8:
            y = self.x(x)
            return y

        else:
            x_5 = x[:, :, :, self.split_1]
            x_12 = x[:, :, :, self.split_2]
            x_8 = x[:, :, :, self.split_3]

            device = x.device

            x = self.x(x)
            x5 = self.x5(x_5)
            x12 = self.x12(x_12)
            x8 = self.x8(x_8)

            full_result = torch.zeros(n, self.out_channels, t, v).to(device)
           # 将每个拆分结果按索引拼接到完整骨架的正确位置
            full_result[:, :, :, self.split_1] = x5
            full_result[:, :, :, self.split_2] = x12
            full_result[:, :, :, self.split_3] = x8

            y = x + full_result
            return y

class stlgcn_child(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 c=1,
                 ratio=0.25,  #0.125
                 ctr='T',
                 ada='T',
                 subset_wise=False,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()

        self.in_channels = in_channels      #3
        self.out_channels = out_channels    #64

        num_subsets = A.size(0)            #8
        self.num_subsets = num_subsets     #8
        self.ctr = ctr           #T
        self.ada = ada            #T
        self.ada_act = ada_act    #softmax
        self.ctr_act = ctr_act     #tanh
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']

        self.subset_wise = subset_wise       #False

        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:              #False
            ratio = 1 / self.num_subsets
        self.ratio = ratio * 2              #0.125
        mid_channels = int(ratio * out_channels)        #64*0.125 = 8  / 16 / 32
        self.mid_channels = mid_channels      #8

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())

        # Introduce non-linear
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),     # 3->8*8
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)  # 8*8 -> 64

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))  #A 8
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))   #B 8

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1) #3->8*8
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1) #3->8*8

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),  # 3->64
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

        #
        if c <= 5:
            seq = 100
        elif c <= 8:
            seq = 50
        else:
            seq = 25

        self.v = self.A  #(8,25,25)
        self.adj_t = nn.Parameter(torch.zeros(seq, seq))  # （50,50）
         
        self.adj_conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)


    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape  #(16,3,100,25)

        res = self.down(x)    # 3->64  (16,64,100,25)
        A = self.A            #[8,25,25]

        #T
        pre_T = self.pre(x)   #(b,64,t,v)
        pre_T = pre_T.permute(0,1,3,2).contiguous() #(b,c,v,t)
        x_T = torch.einsum('kt,bcvt->bcvk', self.adj_t, pre_T)
        x_T = x_T.permute(0,1,3,2).contiguous()

        #vt
        pre_vt = self.pre(x)
        pre_vt = pre_vt.permute(0,2,3,1).contiguous()    #(b,t,v,c)
        adj_vt = torch.einsum('ij, ab -> iab', self.adj_t, self.v.mean(dim=0))  #(50,25,25)
        adj_vt = adj_vt.unsqueeze(0).unsqueeze(0)  # (1,1,50,25,25)
        adj_vt = self.adj_conv3d(adj_vt)           
        adj_vt = adj_vt.squeeze(0).squeeze(0)      # (50,25,25)
        x_vt = torch.einsum('tkv,btvc->btkc', adj_vt, pre_vt)
        x_vt = x_vt.permute(0,3,1,2).contiguous()

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]  #(1,8,1,1,25,25)
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)  #(16, 8*8, 100, 25)
        # * The shape of pre_x is N, K, C, T, V                                    # -> (16,8,8,100,25)

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)    #torch.Size([16, 3, 1, 25])

            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)  #(16,8*8,1,25) ->(16,8,8,1,25)
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)  #(16,8*8,1,25) ->(16,8,8,1,25)


        if self.ctr is not None:   #True
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)   #([16, 8, 8, 1, 25, 25])
            ada_graph = getattr(self, self.ctr_act)(diff)  #tanh(x)

            if self.subset_wise:      #False
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]  #([16, 8, 8, 1, 25, 25])
            A = ada_graph + A    # (16, 8, 8, 1, 25, 25) + (1,8,1,1,25,25) -> (16, 8, 8, 1, 25, 25)

        if self.ada is not None: #True
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None] #（16,8,1,1,25,25）
            ada_graph = getattr(self, self.ada_act)(ada_graph)  #softmax(x)

            if self.subset_wise:   #False
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]   #([16, 8, 1, 1, 25, 25])
            A = ada_graph + A  #([16, 8, 1, 1, 25, 25])  + (16, 8, 8, 1, 25, 25) ->(16, 8, 8, 1, 25, 25)

        if self.ctr is not None or self.ada is not None:  #True
            assert len(A.shape) == 6
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:   #()
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)       #(16,8,8,25,25)               (16,8,8,100,25)
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()  #(16,8,8,100,25)
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()

        x = x.reshape(n, -1, t, v)  #([16, 64, 100, 25])
        x = self.post(x)            #(16, 64, 100, 25)

        x_T = self.post(x_T)
        x_vt = self.post(x_vt)

        x = x + x_T + x_vt

        return self.act(self.bn(x) + res)
