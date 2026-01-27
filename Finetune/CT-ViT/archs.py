from pathlib import Path
import copy
import math
import os
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.autograd import grad as torch_grad
from torchvision import transforms as T, utils


import torchvision

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# from vector_quantize_pytorch import VectorQuantize

# from transformer_maskgit.attention import Attention, Transformer, ContinuousPositionBias

# helpers

import math
from beartype import beartype
from typing import Tuple

from transformer import pair, ContinuousPositionBias, Transformer, exists,divisible_by

# helpers


class CTViT_Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        temporal_patch_size,
        spatial_depth,
        temporal_depth,
        discr_base_dim=16,
        dim_head=64,
        heads=8,
        channels=1,
        use_vgg_and_gan=True,
        vgg=None,
        discr_attn_res_layers=(16,),
        use_hinge_loss=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        # self.to_patch_emb_first_frame = nn.Sequential(
        #     Rearrange(
        #         "b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)",
        #         p1=patch_height,
        #         p2=patch_width,
        #     ),
        #     nn.LayerNorm(channels * patch_width * patch_height),
        #     nn.Linear(channels * patch_width * patch_height, dim),
        #     nn.LayerNorm(dim),
        # )

        self.to_patch_emb = nn.Sequential(
            Rearrange(
                "b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)",
                p1=patch_height,
                p2=patch_width,
                pt=temporal_patch_size,
            ),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim),
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        temporal_transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
            causal=True
        )
        self.enc_spatial_transformer = Transformer(
            depth=spatial_depth, **transformer_kwargs
        )
        self.enc_temporal_transformer = Transformer(
            depth=temporal_depth, **transformer_kwargs
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
        # self.drop = nn.Dropout(p=0.0)
        # self.pool = nn.AdaptiveAvgPool1d()

    # @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    # @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs, strict=False)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))  #!
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]
        return self.decode(codes)

    @property
    def patch_height_width(self):
        return (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

    def encode(self, tokens):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        device = torch.device("cuda")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=device)
        # print("PE done")
        tokens = self.enc_spatial_transformer(
            tokens, attn_bias=attn_bias, video_shape=video_shape
        )
        # print("spatial done")

        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)

        # encode - temporal

        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")

        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        # print("temporal done")

        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)

        return tokens

    def forward(
        self,
        video,
        mask=None,
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4
        # print(video.shape)

        if is_image:
            video = rearrange(video, "b c h w -> b c 1 h w")
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device
        device = torch.device("cuda")
        # assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f
        # assert divisible_by(
        #     f - 1, self.temporal_patch_size
        # ), f"number of frames ({f}) minus one ({f - 1}) must be divisible by temporal patch size ({self.temporal_patch_size})"

        # first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]
        #! CHANGED
        rest_frames = video

        # derive patches
        # print(first_frame.shape)
        # first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb(rest_frames)
        # tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)
        #! CHANGED
        tokens = rest_frames_tokens
        # save height and width in

        shape = tokens.shape
        *_, h, w, _ = shape
        # print(shape)
        # os.system("nvidia-smi")
        # encode - spatial

        tokens = self.encode(tokens)
        # tokens = self.drop(tokens)
        tokens = rearrange(tokens, "b t h w d -> b (t h w) d")
        tokens = tokens.mean(dim=1)

        logits = self.mlp_head(tokens)
        return logits


def extract_encoder(weight_file):
    pt = torch.load(weight_file)
    new_pt = {}
    for name, param in pt.items():
        # if "to_patch" not in name and "to_pixels" not in name:
            # if "rel_pos" in name or "enc" in name:
                # print(name)
                new_pt[name] = param.data
    return new_pt


if __name__ == "__main__":
    # model = CTViT(
    #     dim=512,
    #     codebook_size=8192,
    #     image_size=128,
    #     patch_size=16,
    #     temporal_patch_size=2,
    #     spatial_depth=4,
    #     temporal_depth=4,
    #     dim_head=32,
    #     heads=8,
    # )
    model = CTViT_Encoder(
        dim=512,
        codebook_size=8192,
        image_size=160,
        patch_size=16,
        temporal_patch_size=2,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )
    new_pt = extract_encoder("../saved_weights/model_best_0.7623780916901571.pth.tar")
    model.load_state_dict(new_pt)

    sample = torch.rand((4, 1, 165, 164, 164))

    logits = model.forward(sample)
