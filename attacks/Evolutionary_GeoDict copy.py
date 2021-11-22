

import torch.nn as nn
import torchvision.datasets as dsets

import torchvision.transforms as transforms
import torchvision.models as torch_models
import matplotlib.pyplot as plt
import numpy as np
import torch
from attacks.geoDict import GeoDict
from attacks.imgDict import ImgDict
import os
# from utils import get_label
# from utils import valid_bounds, clip_image_values
from PIL import Image
from torch.autograd import Variable
from numpy import linalg
import math
import cv2
import os
import sys
import numpy as np
import sys
import collections
import cv2
import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def plot_img(img_tensor, file_name):
    img = np.array(img_tensor[0].cpu().numpy()).transpose(1, 2, 0) * 255.
    img = img.astype(np.uint8)

    height, width = img.shape[:2]

    from PIL import Image
    im = Image.fromarray(img)
    im.save("imgs/" + file_name + ".png")




class Evolutionary_Geo_Attack(object):

    def binary_infinity(self, x_a, x, x_img, y, k, model, targeted, batch_indices, ver_lst, depth_lst):
        '''
        linf binary search
        :param k: the number of binary search iteration
        '''
        b = x_a.size(0)
        l = torch.zeros(b)
        u, _ = (x_a - x).reshape(b, -1).abs().max(1)
        for _ in range(k):
            mid = (l + u) / 2
            adv = self.project_infinity(x_a, x, mid)
            x_a_adv = self.geoDict.convert_uv_2_ncc(adv)
            x_a_img = self.geoDict.project_adv_perturbation(x_img, ver_lst, depth_lst, x_a_adv)
            check = self.is_adversarial(x_a_img, y, targeted, batch_indices)
            u[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
            check = check < 1
            l[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
        return self.project_infinity(x_a, x, u)

    def project_infinity(self, x_a, x, l):
        '''
        linf projection
        '''
        return torch.max(x - l[:, None, None, None], torch.min(x_a, x + l[:, None, None, None]))

    def __init__(self, model, dict_model, use_geo=True, use_dict=True, dimension_reduction=None,
                 random_background=False, only_one=False):
        self.model = model
        self.dict_model = dict_model
        self.geoDict = GeoDict()
        self.imgDict = ImgDict()
        self.use_dict = use_dict
        self.use_geo = use_geo
        self.dimension_reduction = dimension_reduction
        self.count = 0
        self.decay_factor = 0.99
        self.c = 0.001
        self.mu = 1e-2
        self.sigma = 3e-2
        self.num_trial = 200
        self.only_one = only_one
        self.random_background = random_background
        self.visualize = False

    def get_predict_label(self, x, batch_indices):
        return self.model(x, batch_indices=batch_indices, unnormalization=False).argmax(1)

    # 0.0048530106
    # 0
    # 0
    # 8700
    # tensor(3.8901)
    # tensor(0.1397)
    def is_adversarial(self, x, y, targeted=False, batch_indices=0, random_background=False):
        '''
        check whether the adversarial constrain holds for x
        '''
        x = x.to(device).contiguous()
        if random_background:
            if torch.min(self.model.get_num_queries(torch.arange(0, x.size(0)))) % 20 != 0:
                if self.use_geo == True:
                    noised_x = (x + (torch.randn_like(x).to(device) * 0.04) * self.x_back_mask)
                else:
                    noised_x = (x + torch.randn_like(x).to(device) * 0.02)
            else:
                noised_x = x
            if targeted:
                return torch.LongTensor((self.get_predict_label(noised_x, batch_indices) == y) + 0)
            else:
                return torch.LongTensor((self.get_predict_label(noised_x, batch_indices) != y) + 0)

        else:
            if targeted:
                return torch.LongTensor((self.get_predict_label(x, batch_indices) == y) + 0)
            else:
                return torch.LongTensor((self.get_predict_label(x, batch_indices) != y) + 0)

    def clip_and_mask(self, x, uv_mask, original_uv):
        x = x * uv_mask
        x = torch.min(torch.max(x, -original_uv), 1 - original_uv)

        return x

    def attack(self, x, y, x_s=None, targeted=False, max_queries=1000, total_search=False):
        face_features = self.dict_model.get_features(x.to(device))
        b = x.size(0)
        # indices for unsuccessful images
        indices = torch.ones(b) > 0
        num_indices = torch.arange(0, b).long()
        background_attack_indices = torch.ones(b) > 0

        x_dtype = np.float
        if self.visualize == True:  # For visualization
            plot_img(x, 'x' + str(self.count))
        if self.use_geo == True:
            ver_lst, depth_lst = self.geoDict.get_face_alignment(x)
            # initialize
            myT = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.5,
                                       contrast=0.5,
                                       saturation=0.5,
                                       hue=0.5),
                transforms.ToTensor()])

            if targeted:

                assert x_s is not None

                original_ncc_codes, used_original_uv_codes = self.geoDict.get_image_color(x, ver_lst, depth_lst)
                original_uv_img = self.geoDict.convert_ncc_2_uv(original_ncc_codes)

                target_ver_lst, target_depth_lst = self.geoDict.get_face_alignment(x_s)
                target_ncc_codes, used_target_uv_codes = self.geoDict.get_image_color(x_s, target_ver_lst,
                                                                                      target_depth_lst)
                # target_ncc_codes[used_original_uv_codes]
                # target_ncc_codes[torch.logical_and(used_original_uv_codes,used_target_uv_codes)] =target_ncc_codes[torch.logical_and(used_original_uv_codes,used_target_uv_codes)] -original_ncc_codes[torch.logical_and(used_original_uv_codes,used_target_uv_codes)]
                target_ncc_codes[torch.logical_and(used_original_uv_codes, ~used_target_uv_codes)] = original_ncc_codes[
                    torch.logical_and(used_original_uv_codes, ~used_target_uv_codes)]

                # original_ncc_codes[used_original_uv_codes]=target_ncc_codes[used_original_uv_codes]-original_ncc_codes[used_original_uv_codes]

                x_a_uv_o = self.geoDict.convert_ncc_2_uv(target_ncc_codes)
                # overlap2 = np.array(x_a_uv_o[0].cpu().numpy()).transpose(1, 2, 0) * 255.
                # overlap2 = overlap2.astype(np.uint8)
                #
                # plot_image(overlap2)
                x_a_uv = x_a_uv_o - original_uv_img
                x_a_ncc = self.geoDict.convert_uv_2_ncc(x_a_uv)
                x_a = self.geoDict.project_adv_perturbation(x, ver_lst, depth_lst, x_a_ncc)

                # overlap2 = np.array(x_a[0].cpu().numpy()).transpose(1, 2, 0) * 255.
                # overlap2 = overlap2.astype(np.uint8)
                # plot_image(overlap2)

                check = self.is_adversarial(x_a, y, targeted, batch_indices=num_indices,
                                            random_background=self.random_background)
                background_attack_indices[check == True] = False
                iters = 0
                while check.sum() < np.shape(y)[0]:
                    # Data augmentation

                    for n in num_indices[background_attack_indices]:
                        x_a_uv[n] = myT(x_a_uv_o[n]) - original_uv_img[n]

                    x_a_ncc[background_attack_indices] = self.geoDict.convert_uv_2_ncc(
                        x_a_uv[background_attack_indices])
                    x_a[background_attack_indices] = self.geoDict.project_adv_perturbation(x[background_attack_indices],
                                                                                           ver_lst[
                                                                                               background_attack_indices],
                                                                                           depth_lst[
                                                                                               background_attack_indices],
                                                                                           x_a_ncc[
                                                                                               background_attack_indices])
                    check[background_attack_indices] = self.is_adversarial(x_a[background_attack_indices],
                                                                           y[background_attack_indices], targeted,
                                                                           num_indices[background_attack_indices],
                                                                           random_background=self.random_background)
                    background_attack_indices[check == True] = False
                    iters += 1
                    if iters > self.num_trial:
                        # overlap2 = np.array(x_a[0].cpu().numpy()).transpose(1, 2, 0) * 255.
                        # overlap2 = overlap2.astype(np.uint8)
                        # plot_image(overlap2)
                        print('Initialization Failed!')
                        print('Turn to combination mode')
                        background_attack_indices[check == True] = False
                        x_a[background_attack_indices] = x_s[background_attack_indices]
                        check = self.is_adversarial(x_a, y, targeted, num_indices,
                                                    random_background=self.random_background)
                        # self.count+=1
                        # print(self.count, ' Error')
                        break

                if check.sum() < y.size(0):
                    print('Some initial images do not belong to the target class!')
                    return x, torch.zeros(b)
                check = self.is_adversarial(x, y, targeted, num_indices)
                if check.sum() > 0:
                    print('Some original images already belong to the target class!')
                    return x, torch.zeros(b)

            else:  # Untargeted Attack

                check = self.is_adversarial(x, y, True, num_indices)
                if check.sum() < y.size(0):
                    print('Some original images do not belong to the original class!')
                    return x, torch.zeros(b)

                original_ncc_codes = self.geoDict.get_image_color(x, ver_lst)
                original_uv_img = self.geoDict.convert_ncc_2_uv(original_ncc_codes)
                if total_search:

                    x_a_uv = self.geoDict.init_item(face_features, original_uv_images=original_uv_img)
                    x_a_uv = torch.min(torch.max(x_a_uv, -original_uv_img), 1 - original_uv_img)
                    # x_a_uv=x_a_uv_o.clone()
                    x_a_ncc = self.geoDict.convert_uv_2_ncc(x_a_uv)
                    x_a = self.geoDict.project_adv_perturbation(x, ver_lst, depth_lst, x_a_ncc)
                    min_norm = torch.norm(x_a.reshape(b, -1) - x.reshape(b, -1), dim=1)

                    l = self.geoDict.get_len_dict()
                    for i in range(l):
                        x_a_uv_temp = torch.unsqueeze(self.geoDict.uv_dict[i], 0)
                        x_a_uv_temp = torch.min(torch.max(x_a_uv_temp, -original_uv_img), 1 - original_uv_img)
                        x_a_ncc_temp = self.geoDict.convert_uv_2_ncc(x_a_uv_temp)
                        x_a_temp = self.geoDict.project_adv_perturbation(x, ver_lst, depth_lst, x_a_ncc_temp)
                        check = self.is_adversarial(x_a_temp, y, targeted, num_indices)
                        perturbation_norm = torch.norm(x_a_temp.reshape(b, -1) - x.reshape(b, -1), dim=1)
                        x_a_uv[perturbation_norm < min_norm and check] = x_a_uv_temp[
                            perturbation_norm < min_norm and check]
                        min_norm[perturbation_norm < min_norm and check] = perturbation_norm[
                            perturbation_norm < min_norm and check]
                else:
                    x_a_uv = self.geoDict.init_item(face_features, original_uv_images=original_uv_img)
                    x_a_uv = torch.min(torch.max(x_a_uv, -original_uv_img), 1 - original_uv_img)
                    # x_a_uv=x_a_uv_o.clone()
                    x_a_ncc = self.geoDict.convert_uv_2_ncc(x_a_uv)
                    x_a = self.geoDict.project_adv_perturbation(x, ver_lst, depth_lst, x_a_ncc)

                    if self.visualize == True:  # For visualization

                        from _3DDFA_V2.utils.io import _load, _dump
                        import os.path as osp
                        make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
                        ncc_code = _load(make_abs_path('../_3DDFA_V2/configs/ncc_code.npy')).T * 0.6 - \
                                   original_ncc_codes[0].numpy() * 0.2

                        def _to_ctype(arr):
                            if not arr.flags.c_contiguous:
                                return arr.copy(order='C')
                            return arr

                        ncc_code = _to_ctype(ncc_code)
                        x_a_3d = self.geoDict.project_adv_perturbation(x, ver_lst,
                                                                       depth_lst,
                                                                       np.expand_dims(ncc_code, 0))
                        plot_img(x_a_3d, 'x_a_face_img_' + str(self.count))
                        plot_img(x_a, 'x_a_img_' + str(self.count))
                        plot_img(x_a_uv * 50 + 0.5,
                                 'x_a_' + str(self.count))

                self.x_back_mask = 1 - self.geoDict.project_adv_perturbation(x, ver_lst, depth_lst,
                                                                             torch.ones_like(x_a_ncc))
                self.x_back_mask = self.x_back_mask.to(device)
                iters = 0

                # print(torch.min(x_a_uv), torch.max(x_a_uv))

                check = self.is_adversarial(x_a, y, targeted, num_indices)

                while check.sum() < np.shape(y)[0]:

                    x_a_uv[check == False] = x_a_uv[check == False] * 1.05
                    x_a_uv[check == False] = torch.min(
                        torch.max(x_a_uv[check == False], -original_uv_img[check == False]+0.2),
                        1 - original_uv_img[check == False]-0.2)

                    x_a_ncc[check == False] = self.geoDict.convert_uv_2_ncc(x_a_uv[check == False])
                    x_a[check == False] = self.geoDict.project_adv_perturbation(x[check == False],
                                                                                ver_lst[check == False],
                                                                                depth_lst[check == False],
                                                                                x_a_ncc[check == False])

                    check = self.is_adversarial(x_a, y, targeted, num_indices, random_background=self.random_background)
                    iters += 1

                    if iters > self.num_trial:
                        print('Initialization failed for some images!')
                        break

                background_attack_indices[check == True] = False
                if check.sum() < np.shape(y)[0]:
                    iters = 0
                    x_a[check == False] = self.imgDict.init_item(face_features[check == False],
                                                                 original_images=x[check == False])

                    check = self.is_adversarial(x_a, y, targeted, num_indices)
                    while check.sum() < np.shape(y)[0]:
                        x_a[check == False] = x_a[check == False] + torch.randn_like(
                            x_a[check == False]) * iters / self.num_trial
                        # x_a[check == False] = x[check == False] + (x_a[check == False] - x[check == False]) * 1.05
                        x_a[check == False] = x_a[check == False].clamp(0, 1)
                        check = self.is_adversarial(x_a, y, targeted, num_indices,
                                                    random_background=self.random_background)
                        iters += 1
                        background_attack_indices[check == True] = False
                        if iters > self.num_trial:
                            print('Initialization failed!')
                            return x, torch.zeros(b)
        else:  # Wihout GeoDict
            # initialize
            if targeted:
                x_a = x_s
                check = self.is_adversarial(x_a, y, targeted, num_indices)
                if check.sum() < y.size(0):
                    print('Some initial images do not belong to the target class!')
                    return x, torch.zeros(b)
                check = self.is_adversarial(x, y, targeted, num_indices)
                if check.sum() > 0:
                    print('Some original images already belong to the target class!')
                    return x, torch.zeros(b)
            else:  # Untargeted Attack
                check = self.is_adversarial(x, y, True, num_indices)

                if check.sum() < y.size(0):
                    print('Some original images do not belong to the original class!')
                    return x, torch.zeros(b)

                background_attack_indices = indices
                iters = 0
                x_a = self.imgDict.init_item(face_features, original_images=x)
                check = self.is_adversarial(x_a, y, targeted, num_indices)
                if self.visualize == True:  # For visualization
                    plot_img(x_a, 'x_a_img_' + str(self.count))

                while check.sum() < np.shape(y)[0]:

                    # x_a[check == False] = x_a[check == False] + torch.randn_like(
                    #     x_a[check == False]) * iters / self.num_trial
                    x_a[check == False] = x[check == False] + (x_a[check == False] - x[check == False]) * 1.05
                    x_a[check == False] = x_a[check == False].clamp(0, 1)
                    check = self.is_adversarial(x_a, y, targeted, num_indices)
                    iters += 1
                    if iters > self.num_trial:
                        print('Initialization failed!')
                        return x, torch.zeros(b)

        background_attack_batch_size = background_attack_indices.sum()

        if background_attack_batch_size > 0:
            x_shape = x.size()
            pert_shape_img = (x_shape[1], *self.dimension_reduction)
            N_img = np.prod(pert_shape_img)
            K_img = int(N_img / 20)
            evolution_path_img = torch.zeros((b, *pert_shape_img))
            diagonal_covariances_img = np.ones((b, *pert_shape_img), dtype=x_dtype)
            mu_img = torch.ones(b) * self.mu
            stats_adversarial_img = [collections.deque(maxlen=30) for i in range(b)]
            x_a_img = x_a.clone()

        if b - background_attack_batch_size > 0:

            # x_a_uv = self.binary_infinity(x_a_uv,torch.zeros_like(x_a_uv), x,y, 10, self.model, targeted, num_indices,ver_lst,depth_lst)

            x_uv_shape = x_a_uv.size()
            pert_shape = (x_uv_shape[1],
                          *self.dimension_reduction)  # int(x_uv_shape[2]/resize_factor),int(x_uv_shape[3]/resize_factor))
            N = np.prod(pert_shape)
            K = int(N / 20)
            evolution_path = torch.zeros((b, *pert_shape))

            x_a_uv_c = x_a_uv.clone()
            diagonal_covariances = np.ones((b, *pert_shape), dtype=x_dtype)
            x_a_uv_mask = self.geoDict.convert_ncc_2_uv(torch.ones_like(original_ncc_codes))
            x_a_uv_mask[:, :, :20, :] = 1
            x_a_uv_mask[:, :, 36:76, 36:76] = 1
            x_a_uv = x_a_uv * x_a_uv_mask
            diagonal_covariances = diagonal_covariances * torch.nn.functional.upsample_bilinear(x_a_uv_mask,
                                                                                                self.dimension_reduction).numpy()

            stats_adversarial = [collections.deque(maxlen=30) for i in range(b)]
            mu = torch.ones(b) * self.mu


        perturbation = torch.zeros((b, x.size(1), *self.dimension_reduction))

        # q_num: current queries
        step = 0

        # x_advs=torch.zeros_like(x)
        while torch.min(self.model.get_num_queries(num_indices)) < max_queries:
            # print(torch.norm(x_a - x_s), torch.norm(x_a - x))
            # if torch.min(self.model.get_num_queries(num_indices)) %100==0:
            #     print(torch.min(x_a_uv),torch.max(x_a_uv))
            Q = self.model.get_num_queries(num_indices)
            for b_c in torch.arange(0, b)[~background_attack_indices].long():
                if self.visualize == True:  # For visualization
                    if Q[b_c] % 1000 == 0:
                        if self.use_geo:
                            x_a = self.geoDict.project_adv_perturbation(x, ver_lst,
                                                                        depth_lst,
                                                                        self.geoDict.convert_uv_2_ncc(
                                                                            x_a_uv_mask))
                            plot_img(x_a, 'x_a_' + str(self.count) + '_' + str(Q[b_c].item()))
                            plot_img(x_a_uv * 50 + 0.5,
                                     'x_a_uv_' + str(self.count) + '_' + str(Q[b_c].item()))
                if Q[b_c] < max_queries:
                    unnormalized_source_direction = -x_a_uv[b_c]
                    source_norm = torch.norm(unnormalized_source_direction)
                    selection_probability = diagonal_covariances[b_c].reshape(-1) / np.sum(diagonal_covariances[b_c])
                    selected_indices = np.random.choice(N, K, replace=False, p=selection_probability)

                    perturbation[b_c] = torch.randn(pert_shape)
                    factor = torch.zeros(N)
                    factor[selected_indices] = 1
                    perturbation[b_c] *= factor.reshape(pert_shape) * np.sqrt(diagonal_covariances[b_c])

                    if self.dimension_reduction:
                        perturbation_large = torch.nn.functional.upsample_bilinear(perturbation[b_c].unsqueeze(0),
                                                                                   x_uv_shape[2:]).squeeze()
                    else:
                        perturbation_large = perturbation[b_c]
                    biased = x_a_uv[b_c] + mu[b_c] * unnormalized_source_direction
                    x_a_uv_c[b_c] = biased + self.sigma * source_norm * perturbation_large / torch.norm(
                        perturbation_large)
                    x_a_uv_c[b_c] = (x_a_uv_c[b_c]) / torch.norm(x_a_uv_c[b_c]) * torch.norm(biased)
                    x_a_uv_c[b_c] = torch.min(torch.max(x_a_uv_c[b_c], -original_uv_img[b_c]), 1 - original_uv_img[b_c])
                    x_a[b_c] = self.geoDict.project_adv_perturbation(x[b_c].unsqueeze(0), ver_lst[b_c].unsqueeze(0),
                                                                     depth_lst[b_c].unsqueeze(0),
                                                                     self.geoDict.convert_uv_2_ncc(
                                                                         x_a_uv_c[b_c].unsqueeze(0)))

            for b_c in torch.arange(0, b)[background_attack_indices].long():
                if self.visualize == True:  # For visualization
                    if Q[b_c] % 1000 == 0:
                        if self.use_geo == False:
                            plot_img(x_a_img, 'x_a_img_final_' + str(self.count))
                            plot_img((x_a_img[background_attack_indices] - x[background_attack_indices]) * 50 + 0.5,
                                     'x_a_' + str(self.count) + '_' + str(Q[b_c]))
                if Q[b_c] < max_queries:
                    unnormalized_source_direction = x[b_c] - x_a_img[b_c]
                    source_norm = torch.norm(unnormalized_source_direction)
                    selection_probability = diagonal_covariances_img[b_c].reshape(-1) / np.sum(
                        diagonal_covariances_img[b_c])
                    selected_indices = np.random.choice(N_img, K_img, replace=False, p=selection_probability)
                    perturbation[b_c] = torch.randn(pert_shape_img)
                    factor = torch.zeros(N_img)
                    factor[selected_indices] = 1
                    perturbation[b_c] *= factor.reshape(pert_shape_img) * np.sqrt(diagonal_covariances_img[b_c])
                    # print(perturbation[b_c])
                    if self.dimension_reduction:
                        perturbation_large = torch.nn.functional.upsample_bilinear(perturbation[b_c].unsqueeze(0),
                                                                                   x_shape[2:]).squeeze()
                    else:
                        perturbation_large = perturbation[b_c]
                    biased = x_a_img[b_c] + mu_img[b_c] * unnormalized_source_direction
                    x_a[b_c] = biased + self.sigma * source_norm * perturbation_large / torch.norm(perturbation_large)
                    x_a[b_c] = x[b_c] - (x[b_c] - x_a[b_c]) / torch.norm(x[b_c] - x_a[b_c]) * torch.norm(
                        x[b_c] - biased)
                    x_a[b_c] = torch.clamp(x_a[b_c], 0, 1)

            is_adversarial = self.is_adversarial(x_a, y, targeted, num_indices,
                                                 random_background=self.random_background)  # Inference

            for b_c in torch.arange(0, b)[~background_attack_indices].long():
                if Q[b_c] < max_queries:
                    stats_adversarial[b_c].appendleft(is_adversarial[b_c])
                    if is_adversarial[b_c]:
                        new_x_adv = x_a_uv_c[b_c]
                        evolution_path[b_c] = self.decay_factor * evolution_path[b_c] + math.sqrt(
                            1 - self.decay_factor ** 2) * perturbation[b_c]
                        diagonal_covariances[b_c] = (1 - self.c) * diagonal_covariances[b_c] + self.c * (
                                    evolution_path[b_c].numpy() ** 2)
                    else:
                        new_x_adv = None
                    if new_x_adv is not None:
                        x_a_uv[b_c] = new_x_adv

                    if len(stats_adversarial[b_c]) == stats_adversarial[b_c].maxlen:
                        p_step = np.mean(stats_adversarial[b_c])
                        mu[b_c] *= np.exp(p_step - 0.2)
                        stats_adversarial[b_c].clear()
            for b_c in torch.arange(0, b)[background_attack_indices].long():
                if Q[b_c] < max_queries:
                    stats_adversarial_img[b_c].appendleft(is_adversarial[b_c])
                    if is_adversarial[b_c]:
                        new_x_adv = x_a[b_c]
                        evolution_path_img[b_c] = self.decay_factor * evolution_path_img[b_c] + math.sqrt(
                            1 - self.decay_factor ** 2) * perturbation[b_c]
                        diagonal_covariances_img[b_c] = (1 - self.c) * diagonal_covariances_img[b_c] + self.c * (
                                evolution_path_img[b_c].numpy() ** 2)
                    else:
                        new_x_adv = None
                    if new_x_adv is not None:
                        x_a_img[b_c] = new_x_adv
                    if len(stats_adversarial_img[b_c]) == stats_adversarial_img[b_c].maxlen:
                        p_step = np.mean(stats_adversarial_img[b_c])
                        mu_img[b_c] *= np.exp(p_step - 0.2)
                        stats_adversarial_img[b_c].clear()
        if self.visualize == True:  # For visualization
            if self.use_geo == False:
                plot_img(x_a_img, 'x_a_final_' + str(self.count))
                plot_img((x_a_img[background_attack_indices] - x[background_attack_indices]) * 50 + 0.5,
                         'x_a_' + str(self.count))
            if self.use_geo:
                x_a = self.geoDict.project_adv_perturbation(x, ver_lst,
                                                            depth_lst,
                                                            self.geoDict.convert_uv_2_ncc(
                                                                x_a_uv))
                plot_img(x_a, 'x_a_final_' + str(self.count))
                plot_img(x_a_uv * 50 + 0.5,
                         'x_a_uv_' + str(self.count))

        if background_attack_batch_size > 0 and self.use_dict:
            self.imgDict.add_item(face_features[background_attack_indices],
                                  x_a_img[background_attack_indices] - x[background_attack_indices], self.only_one)
        if b - background_attack_batch_size > 0 and self.use_dict:
            self.geoDict.add_item(face_features[~background_attack_indices], x_a_uv[~background_attack_indices],
                                  self.only_one)
            print(torch.min(x_a_uv[~background_attack_indices]), torch.max(x_a_uv[~background_attack_indices]))
        return

    def attack_untargeted(self, x_0, y_0, query_limit=20000):
        self.count += 1
        g = self.attack(x=x_0, y=y_0, x_s=x_0, targeted=False, max_queries=query_limit)
        return g

    def attack_targeted(self, x_0, y_0, x_a, query_limit=20000):
        self.count += 1
        g = self.attack(x=x_0, y=y_0, x_s=x_a, targeted=True, max_queries=query_limit)
        return g
