# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np
import os.path as osp

from Sim3DR import rasterize, rasterize_adv
from _3DDFA_V2.utils.functions import plot_image
from _3DDFA_V2.utils.io import _load, _dump
from _3DDFA_V2.utils.tddfa_util import _to_ctype

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def calc_ncc_code():
    from bfm import bfm

    # formula: ncc_d = ( u_d - min(u_d) ) / ( max(u_d) - min(u_d) ), d = {r, g, b}
    u = bfm.u
    u = u.reshape(3, -1, order='F')

    for i in range(3):
        u[i] = (u[i] - u[i].min()) / (u[i].max() - u[i].min())

    _dump('../configs/ncc_code.npy', u)


def pncc(img, ver_lst, tri, show_flag=False, wfp=None, with_bg_flag=True):
    ncc_code = _load(make_abs_path('../configs/ncc_code.npy'))
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)
    # rendering pncc

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose

        #(3, 38365) ncc_code
        _, _,ref_triangles,buffer= rasterize_adv(ver, tri, ncc_code.T, bg=overlap, alpha=0.5)  # m x 3
        buffer=buffer-0.1
        # print(np.sum(ref_triangles<0))
        # print(ver[:,2])
        overlap, used_area,ref_triangles,buffer= rasterize_adv(ver, tri, ncc_code.T, bg=overlap, alpha=0.5,buffer=buffer)  # m x 3
        print(np.sum(ref_triangles>0))
        #(112, 112, 3)
        #print(np.shape(used_area))
        overlap=overlap.astype(np.uint8)
        #print(np.sum(used_area>0)/3)

    if wfp is not None:
        cv2.imwrite(wfp, overlap)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(overlap)

    return overlap


def main():
    # `configs/ncc_code.npy` is generated by `calc_nnc_code` function
    # calc_ncc_code()
    pass


if __name__ == '__main__':
    main()
