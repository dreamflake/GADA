import torch
import torch.nn.functional as F


import numpy as np
import scipy.io as sio


class ImgDict(object):
    def __init__(self, length=100):
        self.denominator=10.
        self.img_dict=None
        self.img_feature_dict=None
        self.img_w=112
        self.img_h=112
        self.img_c=3
        self.length=length
    def add_params(self, alpha,prob):
        if self.get_len_dict() == 0:
            self.alpha_dict = alpha
            self.prob_dict = prob
        else:
            self.alpha_dict = torch.cat((self.alpha_dict, alpha), 0)
            self.prob_dict = torch.cat((self.prob_dict, prob), 0)
    def add_item(self, img_feature, adv_perturbation, only_one=False):
        if self.get_len_dict()==0 or only_one==True:
            self.img_dict=adv_perturbation
            self.img_feature_dict=img_feature
        else:
            self.img_dict=torch.cat((self.img_dict,adv_perturbation),0)
            self.img_feature_dict=torch.cat((self.img_feature_dict,img_feature),0)
        print('Item added!', self.get_len_dict())


    def init_item(self,img_features,original_images=None):
        # Find or init

        batch_size = img_features.shape[0]

        if self.get_len_dict() == 0:
            output_imgs = torch.rand((batch_size,self.img_c,self.img_h,self.img_w))
        else:
            for b in range(batch_size):
                img_feature=img_features[b]
                dist = torch.sum(torch.pow(self.img_feature_dict-torch.unsqueeze(img_feature,0),2),1)
                min_idx = torch.argmin(dist,0)
                if b==0:
                    output_imgs=torch.unsqueeze(self.img_dict[min_idx]+original_images[b],0)
                else:
                    output_imgs=torch.cat((output_imgs,torch.unsqueeze(self.img_dict[min_idx]+original_images[b],0)),0)

            # print(output_imgs)
        return output_imgs

    def get_len_dict(self):
        if self.img_dict is not None:
            return np.shape(self.img_dict)[0]
        else:
            return 0
###############################################################

