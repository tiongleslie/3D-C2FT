import torch
import torch.nn as nn
from functools import partial

from utils.regularizer import DropPath
from models.hybrid_backbone import Backbone_MultiView


class Attention(nn.Module):
    """
        Multi-Head Attention Layer
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Args:
        :param dim (int): patch embedding dimension
        :param num_heads (int): number of attention heads
        :param qkv_bias (bool): enable bias for qkv if True
        :param attn_drop (float): attention dropout rate
        :param proj_drop (float): projection dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
        C2F Block Structure - Encoder
    """
    def __init__(self, dim, out_dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Args:
        :param dim (int): patch embedding dimension
        :param out_dim (int): output of patch embedding dimension
        :param num_heads (int): number of attention heads
        :param mlp_ratio (bool): ratio of mlp hidden dim to embedding dim
        :param qkv_bias (bool): enable bias for qkv if True
        :param drop (float): dropout rate
        :param attn_drop (float): attention dropout rate
        :param drop_path:
        :param act_layer (nn.Module): activation layer
        :param norm_layer (nn.Module): normalization layer
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    """
        MLP layer
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        Args:
        :param in_features (int): input layer size
        :param hidden_features (int): hidden layer size
        :param out_features (int): out layer size
        :param act_layer (nn.Module): activation layer
        :param drop (float): dropout rate
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Encoder(nn.Module):
    """
        Encoder
        ------------------------
        We borrow the work from:
            A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, hybrid_backbone="densenet121", img_size=224, in_chans=3, embed_dim=768, encoder_C2F_block=3,
                 layer_depth=4, num_heads=8, reduce_ratio=2, mlp_ratio=4., qkv_bias=True, norm_layer=None,
                 embed_layer=Backbone_MultiView, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., **kwargs):
        """
        Args:
        :param hybrid_backbone (string): backbone network, default:'densenet121'
        :param img_size (int, tuple): input image size, default:224
        :param in_chans (int): number of input channels, default:3
        :param embed_dim (int): embedding dimension, default:768
        :param encoder_C2F_block (int): depth of C2F block, default:3
        :param layer_depth (int): depth of transformer block layer, default:4
        :param num_heads (int): number of attention heads, default=12
        :param reduce_ratio (int): reduction ratio for C2F block
        :param mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        :param qkv_bias (bool): enable bias for qkv if True
        :param drop_rate (float): dropout rate
        :param attn_drop_rate (float): attention dropout rate
        :param drop_path_rate (float): stochastic depth rate
        :param embed_layer (nn.Module): patch embedding layer
        :param norm_layer (nn.Module): normalization layer
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_C2F_block * layer_depth)]

        self.blocks = [nn.Sequential(*[Block(dim=embed_dim, out_dim=embed_dim, num_heads=num_heads,
                                             mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j])
                                       for j in range(layer_depth)])]
        current_dim = embed_dim
        self.cat_dim = embed_dim

        # Creating C2F Blocks
        for i in range(encoder_C2F_block - 1):
            block_layers = []
            for j in range(layer_depth):
                if j < layer_depth - 1:
                    block_layers.append(Block(dim=current_dim, out_dim=current_dim, num_heads=num_heads,
                                              mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              drop_path=dpr[layer_depth * i + j]))
                else:
                    block_layers.append(Block(dim=current_dim, out_dim=current_dim // reduce_ratio, num_heads=num_heads,
                                              mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              drop_path=dpr[layer_depth * i + j]))
            self.blocks.append(nn.Sequential(*block_layers))
            current_dim = current_dim // reduce_ratio
            self.cat_dim += current_dim
        self.blocks = nn.ModuleList(self.blocks)
        self.norm = norm_layer(self.cat_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        num_views = x.shape[1]

        x = self.patch_embed(x)  # shape => [B, N, hw, D]
        x = x.reshape(batch_size * num_views, x.shape[-2], x.shape[-1])  # shape => [BN, hw, D]
        cls_token = self.cls_token.expand(batch_size * num_views, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        block_features = []
        for blk in self.blocks:
            x = blk(x)
            block_features.append(x)
        block_features.reverse()
        x = torch.cat(block_features, dim=-1)  # shape => [BN, hw, R]
        x = self.norm(x)

        x = x.reshape(batch_size, num_views, x.shape[-2], x.shape[-1])  # shape => [B, N, hw, R]
        return x


class Decoder(torch.nn.Module):
    """
        Decoder
    """

    def __init__(self, cat_dim, patch_size=4, output_size=32, num_heads=8, dropout=0.):
        """
        Args:
        :param cat_dim:
        :param patch_size  (int): patch embedding size, default=4
        :param output_size (int): 3D voxelized output size, default = 32
        :param num_heads (int): number of heads, default = 8
        :param dropout (float): decoder attention dropout rate
        """
        super().__init__()
        self.output_size = output_size
        self.view_norm = view_norm(dim=cat_dim, patch_size=patch_size, output_size=output_size,
                                   num_heads=num_heads, attn_drop=dropout, output_dropout=dropout)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], -1)  # shape => [B, Nhw, R]
        x = self.view_norm(x)  # B, 512, 64
        x = x.reshape(x.shape[0], self.output_size, self.output_size, self.output_size)
        decoder_x = torch.sigmoid(x)  # B, 32, 32, 32

        return decoder_x


class view_norm(torch.nn.Module):
    """
        3D Reconstruction Block Structure in Decoder
    """

    def __init__(self, dim, patch_size=4, output_size=32, num_heads=8, attn_drop=0., output_dropout=0.):
        """
        Args:
        :param dim (int): patch embedding dimension
        :param patch_size (int): number of patches
        :param output_size (int): 3D voxelized output size, default = 32
        :param num_heads (int): number of attention heads, default=8
        :param attn_drop (float): attention dropout rate
        :param output_dropout (float): output dropout rate
        """
        super().__init__()

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        grid_size = output_size // patch_size
        self.q = nn.Parameter(torch.zeros(grid_size ** 3, dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, patch_size ** 3)
        self.proj_drop = nn.Dropout(output_dropout)

    def forward(self, x):
        attn = (self.q @ x.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ x
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Refiner(torch.nn.Module):
    """
        Refiner
    """

    def __init__(self, output_size=32, num_heads=8, refiner_layer_depth=6, dropout=0., refiner_drop_path_rate=0.):
        """
        Args:
        :param output_size (int): 3D voxelized output size, default = 32
        :param num_heads (int): number of heads, default=8
        :param refiner_layer_depth (int): the I-layer of 3D patch attention block
        :param dropout (float): refiner dropout rate
        :param refiner_drop_path_rate (float):
        """
        super().__init__()
        self.output_size = output_size

        # 3D Patch Embedding Attention 1
        PEB_1 = [x.item() for x in torch.linspace(0, refiner_drop_path_rate, refiner_layer_depth)]
        self.refiner1 = nn.ModuleList(
            [nn.Sequential(*[refiner_block(dim=(self.output_size // 4) ** 3, num_heads=num_heads,
                                           mlp_ratio=4., qkv_bias=False,
                                           act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                           drop=dropout, attn_drop=dropout, drop_path=PEB_1[j])
                             for j in range(refiner_layer_depth)])])

        # 3D Patch Embedding Attention 2
        PEB_2 = [x.item() for x in torch.linspace(0, refiner_drop_path_rate, refiner_layer_depth)]
        self.refiner2 = nn.ModuleList(
            [nn.Sequential(*[refiner_block(dim=(self.output_size // 8) ** 3, num_heads=num_heads,
                                           mlp_ratio=4., qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                           drop=dropout, attn_drop=dropout, drop_path=PEB_2[j])
                             for j in range(refiner_layer_depth)])])

    def forward(self, x):
        ''' Refiner '''
        # B, 32, 32, 32 => B, 64, 8*8*8 => B, 512, 4*4*4
        refiner_x = x.reshape(x.shape[0], -1, (self.output_size // 4) ** 3)

        for blk in self.refiner1:
            refiner_x = blk(refiner_x)
        # B, 64, 8*8*8
        refiner_x = refiner_x.reshape(refiner_x.shape[0], -1, (self.output_size // 8) ** 3)
        for blk in self.refiner2:
            refiner_x = blk(refiner_x)
        # B, 512, 4*4*4

        refiner_x = refiner_x.reshape(refiner_x.shape[0], self.output_size, self.output_size, self.output_size)
        refiner_x = torch.sigmoid(refiner_x)  # B, 32, 32, 32

        return refiner_x


class refiner_block(nn.Module):
    """
        3D Patch Attention Block Structure - Refiner
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Args:
        :param dim (int): patch embedding dimension
        :param num_heads (int): number of attention heads, default=8
        :param mlp_ratio (float): ratio of mlp hidden dim to embedding dim
        :param qkv_bias (bool): enable bias for qkv if True
        :param drop (float): dropout rate
        :param attn_drop (float): attention dropout rate
        :param drop_path (float): stochastic depth rate
        :param act_layer (nn.Module): activation layer
        :param norm_layer (nn.Module): normalization layer
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Model_3DC2FT(torch.nn.Module):
    """
        3D-C2FT
    """

    def __init__(self, decoder_output_size=32,
                 decoder_patch_size=4, decoder_num_heads=8, refiner_layer_depth=2,
                 decoder_dropout=0., refiner_drop_path_rate=0., **kwargs):
        """
        Args:
        :param decoder_output_size (int): the dimension of 3D voxelized output [32 * 32 * 32]
        :param decoder_patch_size (int): the patch size for decoder's input [4 * 4* 4]
        :param decoder_num_heads (int): the number of heads
        :param refiner_layer_depth (int): the I-layer in each 3D patch attention block
        :param decoder_dropout (float): dropout rate for decoder
        :param refiner_drop_path_rate (float): dropout rate for refiner

        """
        super().__init__()

        self.encoder = Encoder(**kwargs)
        print("Encoder params       : {}".format(count_params(self.encoder)))

        self.decoder = Decoder(cat_dim=self.encoder.cat_dim, patch_size=decoder_patch_size,
                               output_size=decoder_output_size, num_heads=decoder_num_heads,
                               dropout=decoder_dropout)
        print("Decoder params       : {}".format(count_params(self.decoder)))

        self.refiner = Refiner(output_size=decoder_output_size, num_heads=decoder_num_heads,
                               refiner_layer_depth=refiner_layer_depth, dropout=decoder_dropout,
                               refiner_drop_path_rate=refiner_drop_path_rate)
        print("Refiner params       : {}".format(count_params(self.refiner)))

        print("Total params         : {}".format(count_params(self)))

    def forward(self, x):
        x = self.encoder(x)  # shape => [B, N, hw, R]
        y_hat = self.decoder(x[:, :, 1:])  # shape => [B, x, y, z]
        y = self.refiner(y_hat)  # shape => [B, x, y, z]

        return y


def count_params(model):
    """
        Calculate total of parameters for the given model
    Args:
    :param: model (nn.Module): Pytorch Model
    :return: Total of parameters
    """
    p_p = 0
    for p in list(model.parameters()):
        n_n = 1
        for s in list(p.size()):
            n_n *= s
        p_p += n_n
    return p_p
