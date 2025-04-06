# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py

from typing import Tuple
from collections import OrderedDict
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_VIT_B16_PATH = "/root/autodl-tmp/patient_triple/code/open-metric-learning/my_code/pretrained_models/ViT-B-16.pt"
DWCONV3D_DISABLE_CUDNN = True
class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, T):
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x)
        x_id[:, 1:, :] += x
        return x_id


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=True):
        super(GroupedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.bias = bias

        assert in_features % groups == 0, "in_features must be divisible by groups"
        assert out_features % groups == 0, "out_features must be divisible by groups"

        self.weight = nn.Parameter(torch.Tensor(groups, in_features // groups, out_features // groups))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(groups, out_features // groups))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight,0)
        nn.init.constant_(self.bias,0)
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, C = x.size()
        assert C == self.in_features, "Input feature dimension does not match"

        x = x.view(B, self.groups, C // self.groups)
        x = torch.einsum('bgi,gio->bgo', x, self.weight)

        if self.bias is not None:
            x = x + self.bias

        x = x.reshape(B, self.out_features)
        return x
    
class GroupedMLP(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=True):
        super(GroupedMLP, self).__init__()
        self.l1 = GroupedLinear(in_features, in_features//4,1)
        self.l2 = GroupedLinear(in_features//4, out_features,groups)
        
        
    def forward(self, x):
        return self.l2 (F.leaky_relu(self.l1(x),0.1))
    
class GroupedMLP(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=True):
        super(GroupedMLP, self).__init__()
        self.l11 = GroupedLinear(in_features, in_features//2,1)
        self.l21 = GroupedLinear(in_features//2, in_features//4,1)
        self.l31 = GroupedLinear(in_features//4, out_features,groups)
        
        
    def forward(self, x):
        res = x
        x = self.l21 (F.leaky_relu(self.l11(x),0.1,True))
        x = F.leaky_relu(x,0.1,True)
        return self.l31(x)
        

class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 lora_rank ,
                 ft_all
                 ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        if not ft_all:
            for _ in self.attn.parameters():
                _.requires_grad = False
            for _ in self.ln_1.parameters():
                _.requires_grad = False
            for _ in self.mlp.parameters():
                _.requires_grad = False
            for _ in self.ln_2.parameters():
                _.requires_grad = False

        adapter_class = functools.partial(
            Adapter,
            in_channels=d_model,
            adapter_channels=adapter_width,
            kernel_size=adapter_kernel_size,
        )
        self.adapter_pre_attn = \
            adapter_class() if adapter_pre_attn else None
        self.adapter_pre_mlp = \
            adapter_class() if adapter_pre_mlp else None
        self.d_model = d_model
        self.lora_rank = lora_rank
        if lora_rank>0:
            self.lora_rank_dynamic = lora_rank_dynamic = lora_rank
            G = 16
            self.lora_mlp_q_a = GroupedMLP(d_model, lora_rank_dynamic * d_model,groups=G)
            self.lora_mlp_q_b = GroupedMLP(d_model, lora_rank_dynamic * d_model,groups=G)
            self.lora_mlp_k_a = GroupedMLP(d_model, lora_rank_dynamic * d_model,groups=G)
            self.lora_mlp_k_b = GroupedMLP(d_model, lora_rank_dynamic * d_model,groups=G)
            self.lora_mlp_v_a = GroupedMLP(d_model, lora_rank_dynamic * d_model,groups=G)
            self.lora_mlp_v_b = GroupedMLP(d_model, lora_rank_dynamic * d_model,groups=G)
            
            self.linear_a_q_static = nn.Parameter(torch.zeros(1, self.lora_rank, self.d_model))
            self.linear_b_q_static = nn.Parameter(torch.zeros(1, self.lora_rank, self.d_model))
            self.linear_a_k_static = nn.Parameter(torch.zeros(1, self.lora_rank, self.d_model))
            self.linear_b_k_static = nn.Parameter(torch.zeros(1, self.lora_rank, self.d_model))
            self.linear_a_v_static = nn.Parameter(torch.zeros(1, self.lora_rank, self.d_model))
            self.linear_b_v_static = nn.Parameter(torch.zeros(1, self.lora_rank, self.d_model))
            # if ft_all:
            #     self.linear_a_q_static.requires_grad = False
            #     self.linear_b_q_static.requires_grad = False
            #     self.linear_a_k_static.requires_grad = False
            #     self.linear_b_k_static.requires_grad = False
            #     self.linear_a_v_static.requires_grad = False
            #     self.linear_b_v_static.requires_grad = False
            # LORA parameter generation MLPs -for FFN
            G = 32
            self.lora_mlp_c_fc_a = GroupedMLP(d_model, lora_rank_dynamic * (d_model * 4),groups=G)
            self.lora_mlp_c_fc_b = GroupedMLP(d_model, lora_rank_dynamic * d_model,groups=G)
            self.lora_mlp_c_proj_a = GroupedMLP(d_model, lora_rank_dynamic * d_model,groups=G)
            self.lora_mlp_c_proj_b = GroupedMLP(d_model, lora_rank_dynamic * (d_model * 4),groups=G)
            
            self.linear_a_c_fc_static = nn.Parameter(torch.zeros(1, self.lora_rank,  self.d_model * 4))
            self.linear_b_c_fc_static = nn.Parameter(torch.zeros(1, self.lora_rank,  self.d_model))
            self.linear_a_c_proj_static = nn.Parameter(torch.zeros(1, self.lora_rank,  self.d_model))
            self.linear_b_c_proj_static = nn.Parameter(torch.zeros(1, self.lora_rank,  self.d_model * 4))
            # if ft_all:
                # self.linear_a_c_fc_static.requires_grad = False
                # self.linear_b_c_fc_static.requires_grad = False
                # self.linear_a_c_proj_static.requires_grad = False
                # self.linear_b_c_proj_static.requires_grad = False
        
        
    def attention(self, x: torch.Tensor,x_route_context) -> torch.Tensor:
        
        B, L, C = x.size()
        H = self.attn.num_heads
        
        if self.lora_rank>0:
            # B C x_route_context
            # Generate LORA parameters
            linear_a_q = self.lora_mlp_q_a(x_route_context).view(B, self.lora_rank_dynamic, C)+self.linear_a_q_static
            linear_b_q = self.lora_mlp_q_b(x_route_context).view(B, self.lora_rank_dynamic, C)+self.linear_b_q_static
            linear_a_k = self.lora_mlp_k_a(x_route_context).view(B, self.lora_rank_dynamic, C)+self.linear_a_k_static
            linear_b_k = self.lora_mlp_k_b(x_route_context).view(B, self.lora_rank_dynamic, C)+self.linear_b_k_static
            linear_a_v = self.lora_mlp_v_a(x_route_context).view(B, self.lora_rank_dynamic, C)+self.linear_a_v_static
            linear_b_v = self.lora_mlp_v_b(x_route_context).view(B, self.lora_rank_dynamic, C)+self.linear_b_v_static
        
            
            new_q = torch.einsum('bkc,blk->blc', linear_b_q, torch.einsum('bri,bti->btr', linear_a_q, x)) 
            ##  B K C * B L K
            new_k = torch.einsum('bkc,blk->blc', linear_b_k, torch.einsum('bri,bti->btr', linear_a_k, x))
            new_v = torch.einsum('bkc,blk->blc', linear_b_v, torch.einsum('bri,bti->btr', linear_a_v, x))
            ##

        qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        if self.lora_rank>0:
            qkv[:, :, :self.d_model] += new_q
            qkv[:, :, self.d_model:-self.d_model] += new_k
            qkv[:, :, -self.d_model:] += new_v
        qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([H, H, H], dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).flatten(-2)
        out = self.attn.out_proj(out)

        return out
    
    def mlp_wrapper(self, x: torch.Tensor,x_route_context) -> torch.Tensor:
        if self.lora_rank>0:
            B, L, C = x.size()
            linear_a_c_fc = self.lora_mlp_c_fc_a(x_route_context).view(B, self.lora_rank, C * 4)+self.linear_a_c_fc_static
            linear_b_c_fc = self.lora_mlp_c_fc_b(x_route_context).view(B, self.lora_rank, C)+self.linear_b_c_fc_static
        #     linear_a_c_proj = self.lora_mlp_c_proj_a(x_route_context).view(B, self.lora_rank, C)+self.linear_a_c_proj_static
        #     linear_b_c_proj = self.lora_mlp_c_proj_b(x_route_context).view(B, self.lora_rank, C * 4)+self.linear_b_c_proj_static
            
        c_fc_out = F.linear(x, weight=self.mlp.c_fc.weight, bias=self.mlp.c_fc.bias)
        
        if self.lora_rank>0:
            new_c_fc = torch.einsum('bkc,blk->blc', linear_a_c_fc, torch.einsum('bri,bti->btr', linear_b_c_fc, x))
            ### B K C *  B K L
            c_fc_out += new_c_fc
        
        c_fc_out = self.mlp.gelu(c_fc_out)
        
        # Compute c_proj with LORA
        c_proj_out = F.linear(c_fc_out, weight=self.mlp.c_proj.weight, bias=self.mlp.c_proj.bias)
        # if self.lora_rank>0:
        #     new_c_proj = torch.einsum('bkc,blk->blc', linear_a_c_proj, torch.einsum('bri,bti->btr', linear_b_c_proj, c_fc_out))
        #     c_proj_out += new_c_proj

        return c_proj_out
        

    def forward(self,
                x: torch.Tensor,
                num_frames: int,
                x_route_context
                ) -> torch.Tensor:
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, num_frames)
        x = x + self.attention(self.ln_1(x),x_route_context)
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, num_frames)
        x = x + self.mlp_wrapper(self.ln_2(x),x_route_context)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 lora_rank,
                 ft_all
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size=adapter_kernel_size,
                adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
                lora_rank=lora_rank,
                ft_all = ft_all
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor, num_frames: int,x_route_context) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, num_frames,x_route_context)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 num_classes: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 lora_rank,
                 args
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False)
        self.router_spatial_transform = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=width,
            kernel_size=1, stride=1,padding=0, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(in_channels=width, out_channels=width,
            kernel_size=1, stride=1,padding=0, bias=True),
        )
        self.router_group = args.code_number
        self.knowledge_group = 1
        self.router_MLP = nn.Sequential(
            nn.Linear(width,width),
            nn.LeakyReLU(0.1,True),
            nn.Linear(width,self.knowledge_group*self.router_group ),
        )
        self.ft_comp_MLP = nn.Sequential(
            nn.LeakyReLU(0.1,False),
            nn.Linear(width,width),
            nn.LeakyReLU(0.1,False),
            nn.Linear(width,width),
        )
        nn.init.constant_(self.ft_comp_MLP[3].weight,0)
        nn.init.constant_(self.ft_comp_MLP[3].bias,0)
        self.knowledge_pool = nn.Parameter(torch.randn(self.knowledge_group,self.router_group, 768//self.router_group))
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                (input_resolution // patch_size) ** 2 + 1, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,
            adapter_width, adapter_layers, adapter_kernel_size,
            adapter_pre_attn, adapter_pre_mlp,lora_rank  =lora_rank,ft_all =args.ft_all)

        self.ln_post = LayerNorm(width)
        self.lora_rank = lora_rank
        # for n, p in self.named_parameters():
        #   if 'adapter' not in n:
        #     p.requires_grad_(False)
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(width, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0.)
        self.last_mlp = nn.Sequential(
            nn.Linear(768,768),
            nn.LeakyReLU(0.1),
            nn.Linear(768,768),
        )
        self.args = args

    def forward(self, x: torch.Tensor,return_mode = "default"):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x_for_route = self.router_spatial_transform(x)
        x_for_route_root  =x_for_route.mean(-1).mean(-1)
        x_for_route = self.router_MLP(x_for_route_root) ##B K
        x_for_ft_comp = self.ft_comp_MLP(x_for_route_root)
        x_for_route = x_for_route.reshape(B*T,self.knowledge_group,self.router_group ) ## B, KG, RG
        x_for_route = F.softmax(x_for_route,dim=-1) ## B K
        
        #### 1  K C
        x_route_context = self.knowledge_pool.unsqueeze(0) * x_for_route.unsqueeze(-1)## B, KG, RG, C//RG
        x_route_context = x_route_context.sum(1) ## B RG, C//RG
        x_route_context = x_route_context.reshape(B*T, 768)
        if self.args.no_codebook:
            x_route_context = x_for_route_root
        if return_mode == "modality_token":
            return x_route_context.reshape(B,T,768).mean(1)
        spatial_size = tuple(x.size()[2:])
        x = x.flatten(-2).permute(0, 2, 1)##B N C
        if self.lora_rank>0:
            x = x + x_for_ft_comp.unsqueeze(1)
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
            ], dim=1)  # [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1) # BT, L, D

        x = self.transformer(x, T,x_route_context)

        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
        # x_global = x[:, :, 0, :].mean(dim=1)
        x_local = x[:, :, 1:, :].mean(dim=1)
        x_global = x_local.mean(1)
        H,W = 14,14
        B,S,C = x_local.size()
        assert S == H*W
        x_local  =x_local.reshape(B,H,W,C).permute(0,3,1,2)
        # x_global = self.ln_post(x_global)
        return self.last_mlp(x_global), x_local



def clip_vit_base_patch16_adapter24x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=False,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py.'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    model.load_state_dict(checkpoint.visual.state_dict(), strict=False)
    return model

def clip_vit_base_patch16(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=False,
        adapter_pre_mlp=False,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py.'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    model.load_state_dict(checkpoint.visual.state_dict(), strict=False)
    print(model)
    return model

def clip_vit_base_patch16_adapter12x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_pre_attn=False,
        adapter_pre_mlp=False,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model