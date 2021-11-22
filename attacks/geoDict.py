import torch
import torch.nn.functional as F
# 3DDFA_V2

import cv2
import yaml
from _3DDFA_V2.utils.functions import plot_image
from _3DDFA_V2.TDDFA import TDDFA
import numpy as np
import scipy.io as sio
from _3DDFA_V2.Sim3DR import rasterize_adv,rasterize
from _3DDFA_V2.utils.tddfa_util import _to_ctype

import os.path as osp
from _3DDFA_V2.utils.io import _load
make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
def pncc_render(img, ver_, tri, depth, ncc_code, show_flag=True, wfp=None, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)
    # rendering pncc
    ver = _to_ctype(ver_.T)  # transpose
    #(3, 38365) ncc_code
    depth=depth-0.1
    # print(np.sum(ref_triangles<0))
    overlap, used_area,ref_triangles,buffer= rasterize_adv(ver, tri, ncc_code, bg=overlap, alpha=1,buffer=depth)  # m x 3

    if wfp is not None:
        cv2.imwrite(wfp, overlap)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        overlap2=overlap*255.
        overlap2=overlap2.astype(np.uint8)
        plot_image(overlap2)

    return overlap, ref_triangles
def get_colors(img, x,y):
    # nearest-neighbor sampling
    #print(img.shape)
    [h, w, _] = img.shape
    x = np.minimum(np.maximum(x, 0), w - 1)  # x
    y = np.minimum(np.maximum(y, 0), h - 1)  # y
    ind_x = np.round(x).astype(np.int32)
    ind_y = np.round(y).astype(np.int32)
    colors = img[ind_x, ind_y, :]  # n x 3
    return colors
def bilinear_interpolate(img, x, y,z=None,depth=None):
    """
    https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    """
    if type(depth) is torch.Tensor:
        depth=depth.numpy()
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]
    if z is not None:
        i_a[z<depth[y0, x0]-5]=-1
        i_b[z<depth[y1, x0]-5]=-1
        i_c[z<depth[y0, x1]-5]=-1
        i_d[z<depth[y1, x1]-5]=-1

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa[..., np.newaxis] * i_a + wb[..., np.newaxis] * i_b + wc[..., np.newaxis] * i_c + wd[..., np.newaxis] * i_d


def load_uv_coords(fp):
    C = sio.loadmat(fp)
    uv_coords = C['UV'].copy(order='C').astype(np.float32)
    return uv_coords


def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1), dtype=np.float32)))  # add z
    return uv_coords
class GeoDict(object):


    def __init__(self, length=100):

        self.denominator=10.

        # (3, 38365)
        self.dummy_ncc_code=np.zeros((38365,3))
        self.uv_dict=None
        self.img_feature_dict=None
        g_uv_coords = load_uv_coords(make_abs_path('../_3DDFA_V2/configs/BFM_UV.mat'))
        indices = _load(make_abs_path('../_3DDFA_V2/configs/indices.npy'))
        g_uv_coords = g_uv_coords[indices, :]
        self.uv_w=112
        self.uv_h=112
        self.uv_c=3

        self.uv_coords = process_uv(g_uv_coords, uv_h=self.uv_h, uv_w=self.uv_w)
        config='_3DDFA_V2/configs/mb1_120x120.yml'
        # Init FaceBoxes and TDDFA, recommend using onnx flag
        use_onnx=True
        cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
        if use_onnx==True:
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'
            from _3DDFA_V2.TDDFA_ONNX import TDDFA_ONNX

            self.tddfa = TDDFA_ONNX(**cfg)
        else:
            gpu_mode = False
            self.tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    def add_params(self, alpha,prob):
        if self.get_len_dict() == 0:
            self.alpha_dict = alpha
            self.prob_dict = prob
        else:
            self.alpha_dict = torch.cat((self.alpha_dict, alpha), 0)
            self.prob_dict = torch.cat((self.prob_dict, prob), 0)
    def add_item(self, img_feature, uv_image, only_one=False):
        if self.get_len_dict()==0 or only_one==True:
            self.uv_dict=uv_image
            self.img_feature_dict=img_feature
        else:
            self.uv_dict=torch.cat((self.uv_dict,uv_image),0)
            self.img_feature_dict=torch.cat((self.img_feature_dict,img_feature),0)
    def init_one_item(self,img_features, original_uv_images=None):
        # Find or init

        reduction_factor=4
        batch_size = img_features.shape[0]
        if self.get_len_dict() == 0:
            output_uv_imgs = 0.5*torch.randn((1,self.uv_c,self.uv_h//reduction_factor,self.uv_w//reduction_factor)).repeat(batch_size, 1, 1, 1)
            output_uv_imgs=torch.nn.functional.upsample_bilinear(output_uv_imgs, (self.uv_h,self.uv_w))
            #-original_uv_images
        else:
            for b in range(batch_size):
                img_feature=img_features
                dist = torch.sum(torch.pow(self.img_feature_dict-torch.unsqueeze(img_feature,0),2),1)
                min_idx = torch.argmin(dist,0)
                if b==0:
                    output_uv_imgs=torch.unsqueeze(self.uv_dict[min_idx],0)
                else:
                    output_uv_imgs=torch.cat((output_uv_imgs,torch.unsqueeze(self.uv_dict[min_idx],0)),0)
        return output_uv_imgs
    #

    def init_item(self,img_features, original_uv_images=None,use_dict=True):
        # Find or init

        batch_size = img_features.shape[0]
        if self.get_len_dict() == 0 or use_dict==False:
            output_uv_imgs = torch.randn((batch_size,self.uv_c,self.uv_h,self.uv_w))#original_uv_images

        else:
            for b in range(batch_size):
                img_feature=img_features[b]
                dist = torch.sum(torch.pow(self.img_feature_dict-torch.unsqueeze(img_feature,0),2),1)
                min_idx = torch.argmin(dist,0)
                if b==0:
                    output_uv_imgs=torch.unsqueeze(self.uv_dict[min_idx],0)
                else:
                    output_uv_imgs=torch.cat((output_uv_imgs,torch.unsqueeze(self.uv_dict[min_idx],0)),0)
        return output_uv_imgs


    def get_len_dict(self):
        if self.uv_dict is not None:
            return np.shape(self.uv_dict)[0]
        else:
            return 0
    def get_face_alignment(self,imgs):
        if type(imgs) is torch.Tensor:
            imgs=imgs.numpy()
        batch_size=imgs.shape[0]
        for b in range(batch_size):
            img = imgs[b].transpose(1, 2, 0)
            img = img*255.0 #[:,:,::-1]
            img=img.astype(np.uint8)
            boxes=[[0.0, 0.0, 112.0, 112.0, 1.0]]
            # 3D Face Alignment
            param_lst, roi_box_lst = self.tddfa(img,boxes )
            # Visualization and serialization
            dense_flag = True
            out = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
            overlap = img.copy()/255.
            ver_=out[0]

            ver = _to_ctype(ver_.T)  # transpose
            _, _, ref_triangles, buffer = rasterize_adv(ver, self.tddfa.tri, self.dummy_ncc_code, bg=overlap, alpha=1)  # m x 3
            if b==0:
                ver_lst=np.expand_dims(ver_,0)
                depth_lst=np.expand_dims(buffer,0)
            else:
                ver_lst=np.concatenate((ver_lst,np.expand_dims(ver_,0)),axis=0)
                depth_lst=np.concatenate((depth_lst,np.expand_dims(buffer,0)),axis=0)
        ver_lst=torch.FloatTensor(ver_lst)

        depth_lst = torch.FloatTensor(depth_lst)
        return ver_lst,depth_lst
    def get_image_color(self, imgs,ver_lst,depth_lst=None): # For efficient targeted attacks

        if imgs.dtype==torch.float32:
            imgs=imgs.numpy()
        if type(ver_lst) is torch.Tensor:
            ver_lst = ver_lst.numpy()
        if depth_lst is not None:
            batch_size = imgs.shape[0]
            for b in range(batch_size):
                img = imgs[b].transpose(1, 2, 0)
                ver_=ver_lst[b]
                ver = _to_ctype(ver_.T)  # transpose
                ncc_code = bilinear_interpolate(img, ver[:, 0], ver[:, 1], ver[:, 2],depth_lst[b])
                used_ncc_code=(ncc_code>=0).astype(np.float)
                ncc_code=np.clip(ncc_code,0,1)
                if b == 0:
                    ncc_codes = np.expand_dims(ncc_code, axis=0)
                    used_ncc_codes=np.expand_dims(used_ncc_code,axis=0)
                else:
                    ncc_codes = np.concatenate((ncc_codes,np.expand_dims(ncc_code, axis=0)), axis=0)
                    used_ncc_codes = np.concatenate((used_ncc_codes,np.expand_dims(used_ncc_code, axis=0)), axis=0)


            ncc_codes=torch.FloatTensor(ncc_codes)
            used_ncc_codes=torch.BoolTensor(used_ncc_codes)
            return ncc_codes, used_ncc_codes
        else:
            batch_size = imgs.shape[0]
            for b in range(batch_size):
                img = imgs[b].transpose(1, 2, 0)
                ver_ = ver_lst[b]
                ver = _to_ctype(ver_.T)  # transpose
                ncc_code = bilinear_interpolate(img, ver[:, 0], ver[:, 1])
                if b == 0:
                    ncc_codes = np.expand_dims(ncc_code, axis=0)
                else:
                    ncc_codes = np.concatenate((ncc_codes, np.expand_dims(ncc_code, axis=0)), axis=0)

            ncc_codes = torch.FloatTensor(ncc_codes)
            return ncc_codes
    def convert_uv_2_ncc(self, uv_imgs):  # For efficient targeted attacks
        if type(uv_imgs) is torch.Tensor:
            uv_imgs = uv_imgs.numpy()
        batch_size = uv_imgs.shape[0]
        for b in range(batch_size):
            uv_img = uv_imgs[b].transpose(1, 2, 0)
            # uv_img = uv_img[:, :, ::-1]
            colors = bilinear_interpolate(uv_img, self.uv_coords[:, 0], self.uv_coords[:, 1])
            # colors = get_colors(uv_img,self.uv_coords[:, 0], self.uv_coords[:, 1])

            if b == 0:
                ncc_codes = np.expand_dims(colors, 0)
            else:
                ncc_codes = np.concatenate((ncc_codes,np.expand_dims(colors, 0)), 0)
        ncc_codes=torch.FloatTensor(ncc_codes)
        return ncc_codes

    def convert_ncc_2_uv(self, ncc_codes):  # For efficient targeted attacks
        if type(ncc_codes) is torch.Tensor:
            ncc_codes = ncc_codes.numpy()
        batch_size = ncc_codes.shape[0]
        for b in range(batch_size):
            colors=ncc_codes[b]
            uv_img,_,_,_ = rasterize_adv(self.uv_coords, self.tddfa.tri, colors, height=self.uv_h, width=self.uv_w, channel=self.uv_c)
            if b == 0:
                uv_imgs = np.expand_dims(uv_img, 0)
            else:
                uv_imgs = np.concatenate((uv_imgs,np.expand_dims(uv_img, 0)), 0)

        uv_imgs = uv_imgs.transpose(0, 3, 1, 2)
        uv_imgs=torch.FloatTensor(uv_imgs)
        return uv_imgs
    def project_adv_perturbation(self, imgs, ver_lst, depth_lst, ncc_codes):

        if type(imgs) is torch.Tensor:
            imgs=imgs.numpy()
        if type(ver_lst) is torch.Tensor:
            ver_lst = ver_lst.numpy()
        if type(depth_lst) is torch.Tensor:
            depth_lst = depth_lst.numpy()
        if type(ncc_codes) is torch.Tensor:
            ncc_codes = ncc_codes.numpy()
        batch_size=imgs.shape[0]
        for b in range(batch_size):
            img = imgs[b].transpose(1, 2, 0)
            # img = img[:,:,::-1]
            ver=ver_lst[b]
            ncc_code=ncc_codes[b]
            depth=depth_lst[b]
            rendered_img, ref_triangles=pncc_render(img, ver, self.tddfa.tri,depth, ncc_code,show_flag=False, wfp=None, with_bg_flag=True)
            if b==0:
                rendered_img_lst=np.expand_dims(rendered_img,0)
                # ref_triangles_lst=np.expand_dims(ref_triangles,0)
            else:
                rendered_img_lst=np.concatenate((rendered_img_lst,np.expand_dims(rendered_img,0)),0)
                # ref_triangles_lst=np.cat(np.expand_dims(ref_triangles,0),0)

        rendered_img_lst = torch.FloatTensor(rendered_img_lst.transpose(0,3, 1, 2))
        rendered_img_lst=torch.clamp(rendered_img_lst,0,1)
        return rendered_img_lst
###############################################################

