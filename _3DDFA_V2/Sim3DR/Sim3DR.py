# coding: utf-8

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '.')
add_path(lib_path)

import numpy as np


import Sim3DR_Cython
# coding: utf-8





def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    Sim3DR_Cython.get_normal(normal, vertices, triangles, vertices.shape[0], triangles.shape[0])
    return normal


def rasterize(vertices, triangles, colors, bg=None,
              height=None, width=None, channel=None,
              reverse=False, alpha=1.):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.uint8)

    buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    Sim3DR_Cython.rasterize(bg, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel,
                            reverse=reverse, alpha=alpha)
    return bg
def rasterize_adv(vertices, triangles, colors, bg=None,
              height=None, width=None, channel=None, buffer=None,
              reverse=False, alpha=1.):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.float32)

    used_area = np.zeros_like(bg, dtype=np.int32)
    ref_colors = np.zeros((np.shape(triangles)[0]), dtype=np.int32)
    if buffer is None:
        buffer = np.zeros((height, width), dtype=np.float32) - 1e8


    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    if bg.dtype != np.float32:
        bg = bg.astype(np.float32)
    Sim3DR_Cython.rasterize_adv(bg, used_area,ref_colors, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel,
                            reverse=reverse, alpha=alpha)
    return bg,used_area,ref_colors, buffer
def get_image_color(vertices, triangles, colors, bg=None,
              height=None, width=None, channel=None, buffer=None,
              reverse=False, alpha=1.):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.float32)

    used_area = np.zeros_like(bg, dtype=np.int16)
    ref_colors = np.zeros((np.shape(triangles)[0]), dtype=np.int32)
    if buffer is None:
        buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    if bg.dtype != np.float32:
        bg = bg.astype(np.float32)
        
    Sim3DR_Cython.get_image_color(bg, used_area,ref_colors, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel,
                            reverse=reverse, alpha=alpha)
    return colors,used_area,ref_colors, buffer
# def get_img_color(vertices, triangles, colors, bg=None,
#               height=None, width=None, channel=None, buffer=None,
#               reverse=False, alpha=1.):
#     if bg is not None:
#         height, width, channel = bg.shape
#     else:
#         assert height is not None and width is not None and channel is not None
#         bg = np.zeros((height, width, channel), dtype=np.float32)
#
#     used_area = np.zeros_like(bg, dtype=np.int)
#     ref_colors = np.zeros((np.shape(triangles)[0]), dtype=np.int)
#     if buffer is None:
#         buffer = np.zeros((height, width), dtype=np.float32) - 1e8
#
#     if colors.dtype != np.float32:
#         colors = colors.astype(np.float32)
#     if bg.dtype != np.float32:
#         bg = bg.astype(np.float32)
#     Sim3DR_Cython.get_image_color(bg, used_area,ref_colors, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel,
#                             reverse=reverse, alpha=alpha)
#     return colors,used_area,ref_colors, buffer