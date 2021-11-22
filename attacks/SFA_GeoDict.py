import torch
import torch.nn.functional as F
from attacks.geoDict import GeoDict
import numpy as np

from _3DDFA_V2.utils.functions import plot_image
from attacks.geoDict import GeoDict
from attacks.imgDict import ImgDict
import torchvision.transforms as transforms
def resize(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bilinear', align_corners=False)





def project_infinity(x_a, x, l):
    '''
    linf projection1
    '''
    return torch.max(x - l[:, None, None, None], torch.min(x_a, x + l[:, None, None, None]))


def get_predict_label(x, model):
    return model(x,unnormalization=False).argmax(1)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SFA_Geo_Attack(object):


    def __init__(self, model, dict_model, use_geo=True, use_dict=True, only_one=False):

        self.model = model
        self.dict_model=dict_model
        self.geoDict = GeoDict()
        self.imgDict = ImgDict()
        self.use_dict=use_dict
        self.use_geo=use_geo
        self.num_trial=200
        self.count=0
        self.only_one=only_one

    def get_predict_label(self,x,batch_indices):
        return self.model(x,batch_indices=batch_indices, unnormalization=False).argmax(1)
    def is_adversarial(self,x, y, targeted=False,batch_indices=0):
        '''
        check whether the adversarial constrain holds for x
        '''
        x = x.to(device).contiguous()
        if targeted:
            return torch.LongTensor((self.get_predict_label(x,batch_indices) == y) + 0)
        else:
            return torch.LongTensor((self.get_predict_label(x,batch_indices) != y) + 0)
    def binary_infinity_uv(self,x_a, x, x_original, y,ver_lst,depth_lst, k,  targeted,batch_indices=None):
        '''
        linf binary search
        :param k: the number of binary search iteration
        '''
        if type(x_a) is np.ndarray:
            x_a=torch.FloatTensor(x_a)
        if type(x) is np.ndarray:
            x=torch.FloatTensor(x)
        b = x_a.size()[0]
        l = torch.zeros(b)
        u, _ = (x_a - x).reshape(b, -1).abs().max(1)
        for _ in range(k):
            mid = (l + u) / 2
            adv_uv =project_infinity(x_a, x, mid)
            adv_ncc=self.geoDict.convert_uv_2_ncc(adv_uv)
            adv=self.geoDict.project_adv_perturbation(x_original,ver_lst,depth_lst,adv_ncc)
            check = self.is_adversarial(adv, y, targeted,batch_indices=batch_indices)
            u[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
            check = check < 1
            l[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
        return project_infinity(x_a, x, u)

    def binary_infinity(self,x_a, x, y, k, targeted, batch_indices):
        '''
        linf binary search
        :param k: the number of binary search iteration
        '''
        b = x_a.size(0)
        l = torch.zeros(b)
        u, _ = (x_a - x).reshape(b, -1).abs().max(1)
        for _ in range(k):
            mid = (l + u) / 2
            adv = project_infinity(x_a, x, mid).clamp(0, 1)
            check = self.is_adversarial(adv, y, targeted, batch_indices)
            u[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
            check = check < 1
            l[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
        return project_infinity(x_a, x, u).clamp(0, 1)
    def SFA(self, x, y, x_s=None, resize_factor=2.,targeted=False, max_queries=1000, sharing=False):
        '''
        Sign Flip Attack: linf decision-based adversarial attack
        :param x: original images, torch tensor of size (b,c,h,w)
        :param y: original labels for untargeted attacks, target labels for targeted attacks, torch tensor of size (b,)
        :param model: target model
        :param resize_factor: dimensionality reduction rate, >= 1.0
        :param x_a: initial images for targeted attacks, torch tensor of size (b,c,h,w). None for untargeted attacks
        :param targeted: attack mode, True for targeted attacks, False for untargeted attacks
        :param max_queries: maximum query number
        :param linf: linf threshold
        :return: adversarial examples and corresponding required queries
        '''

        face_features = self.dict_model.get_features(x.to(device))
        b = x.size(0)

        # indices for unsuccessful images
        indices = torch.ones(b) > 0
        num_indices = torch.arange(0, b).long()
        background_attack_indices = torch.ones(b) > 0

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
                target_ncc_codes[torch.logical_and(used_original_uv_codes, ~used_target_uv_codes)] = original_ncc_codes[
                    torch.logical_and(used_original_uv_codes, ~used_target_uv_codes)]

               
                x_a_uv_o = self.geoDict.convert_ncc_2_uv(target_ncc_codes)
                
                x_a_uv = x_a_uv_o - original_uv_img
                x_a_ncc = self.geoDict.convert_uv_2_ncc(x_a_uv)
                x_a = self.geoDict.project_adv_perturbation(x, ver_lst, depth_lst, x_a_ncc)

                check = self.is_adversarial(x_a, y, targeted, batch_indices=num_indices)
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
                                                                           num_indices[background_attack_indices])

                    background_attack_indices[check == True] = False
                    iters += 1
                    if iters > self.num_trial:
                        print('Initialization Failed!')
                        print('Turn to combination mode')
                        background_attack_indices[check == True] = False
                        x_a[background_attack_indices] = x_s[background_attack_indices]
                        check = self.is_adversarial(x_a, y, targeted, num_indices)
                        self.count += background_attack_indices.sum()
                        print(self.count, ' Error')
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
                if sharing == False:

                    original_ncc_codes, used_uv_codes = self.geoDict.get_image_color(x, ver_lst, depth_lst)

                    original_uv_img = self.geoDict.convert_ncc_2_uv(original_ncc_codes)

                    x_a_uv = self.geoDict.init_item(face_features, original_uv_images=original_uv_img)
                    x_a_uv = torch.min(torch.max(x_a_uv, -original_uv_img), 1 - original_uv_img)
                    # x_a_uv=x_a_uv_o.clone()
                    x_a_ncc = self.geoDict.convert_uv_2_ncc(x_a_uv)
                    x_a = self.geoDict.project_adv_perturbation(x, ver_lst, depth_lst, x_a_ncc)
                    iters = 0

                    # print(torch.min(x_a_uv), torch.max(x_a_uv))

                    check = self.is_adversarial(x_a, y, targeted, num_indices)

                    while check.sum() < np.shape(y)[0]:
                        x_a_uv[check == False] = x_a_uv[check == False] * 1.05
                        x_a_uv[check == False] = torch.min(
                            torch.max(x_a_uv[check == False], -original_uv_img[check == False]),
                            1 - original_uv_img[check == False])

                        x_a_ncc[check == False] = self.geoDict.convert_uv_2_ncc(x_a_uv[check == False])
                        x_a[check == False] = self.geoDict.project_adv_perturbation(x[check == False],
                                                                                    ver_lst[check == False],
                                                                                    depth_lst[check == False],
                                                                                    x_a_ncc[check == False])
                        check = self.is_adversarial(x_a, y, targeted, num_indices)
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
                            check = self.is_adversarial(x_a, y, targeted, num_indices)
                            iters += 1
                            background_attack_indices[check == True] = False
                            if iters > self.num_trial:
                                print('Initialization failed!')
                                return x, torch.zeros(b)
                else:
                    original_ncc_codes = self.geoDict.get_image_color(x, ver_lst, depth_lst)
                    original_uv_img = self.geoDict.convert_ncc_2_uv(original_ncc_codes)
                    num_trials = 0
                    x_a_uv = self.geoDict.init_item(face_features, original_uv_image=original_uv_img)
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
                if sharing == False:
                    iters = 0
                    x_a = self.imgDict.init_item(face_features, original_images=x)
                    check = self.is_adversarial(x_a, y, targeted, num_indices)

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
                else:
                    original_ncc_codes = self.geoDict.get_image_color(x, ver_lst, depth_lst)
                    original_uv_img = self.geoDict.convert_ncc_2_uv(original_ncc_codes)
                    num_trials = 0
                    x_a_uv = self.geoDict.init_item(face_features, original_uv_image=original_uv_img)

        background_attack_batch_size = background_attack_indices.sum()

        delta=x.clone()


        b, c, h, w = x.size()
        assert resize_factor >= 1.
        h_dr, w_dr = int(h // resize_factor), int(w // resize_factor)

        # Q: query number for each image
        Q = torch.zeros(b)
        # q_num: current queries
        q_num = 0
        # 10 queries for binary search
        q_num, Q = q_num + 10, Q + 10

        # indices for unsuccessful images
        unsuccessful_indices = torch.ones(b) > 0

        # hyper-parameter initialization
        alpha = torch.ones(b) * 0.004
        prob = torch.ones_like(delta) * 0.999

        if background_attack_batch_size > 0:
            # linf binary search
            x_a[background_attack_indices] = self.binary_infinity(x_a[background_attack_indices], x[background_attack_indices], y[background_attack_indices], 10, targeted, num_indices[background_attack_indices])
            delta[background_attack_indices] = x_a[background_attack_indices] - x[background_attack_indices]

        if b - background_attack_batch_size > 0:
            x_a_uv_mask = self.geoDict.convert_ncc_2_uv(torch.ones_like(original_ncc_codes[~background_attack_indices]))
            x_a_uv_mask[:, :, :20, :] = 1
            x_a_uv_mask[:, :, 36:76, 36:76] = 1

            original_uv = np.zeros_like(x_a_uv[~background_attack_indices])

            delta[~background_attack_indices] = self.binary_infinity_uv(x_a_uv[~background_attack_indices], original_uv, x[~background_attack_indices], y[~background_attack_indices], ver_lst[~background_attack_indices], depth_lst[~background_attack_indices], 10, targeted, num_indices[~background_attack_indices])

            prob[~background_attack_indices] = prob[~background_attack_indices]* x_a_uv_mask

        print('Face attack: ', b - background_attack_batch_size, ', Background attack: ', background_attack_batch_size)

        prob = resize(prob, h_dr, w_dr)

        # additional counters for hyper-parameter adjustment
        reset = 0
        proj_success_rate = torch.zeros(b)
        flip_success_rate = torch.zeros(b)





        while torch.min(self.model.get_num_queries(num_indices)) < max_queries:
            Q = self.model.get_num_queries(num_indices)
            unsuccessful_indices=Q<max_queries
            unsuccessful_face_indices=torch.logical_and(unsuccessful_indices,~background_attack_indices)
            unsuccessful_background_indices=torch.logical_and(unsuccessful_indices,background_attack_indices)
            reset += 1
            b_cur = unsuccessful_indices.sum()

            # the project step
            eta = torch.randn([b_cur, c, h_dr, w_dr]).sign() * alpha[unsuccessful_indices][:, None, None, None]
            eta = resize(eta, h, w)
            l, _ = delta[unsuccessful_indices].abs().view(b_cur, -1).max(1)


            delta_p = project_infinity(delta[unsuccessful_indices] + eta, torch.zeros_like(eta),
                                       l - alpha[unsuccessful_indices])
            if unsuccessful_face_indices.sum() > 0:
                x_a[unsuccessful_face_indices]=(self.geoDict.project_adv_perturbation(x[unsuccessful_face_indices], ver_lst[unsuccessful_face_indices],
                                                       depth_lst[unsuccessful_face_indices],
                                                       self.geoDict.convert_uv_2_ncc(delta_p[unsuccessful_face_indices]))).clamp(0, 1)
            if unsuccessful_background_indices.sum()>0:
                x_a[unsuccessful_background_indices]=(x[unsuccessful_background_indices] + delta_p[unsuccessful_background_indices]).clamp(0, 1)

            check = self.is_adversarial(x_a[unsuccessful_indices], y[unsuccessful_indices], targeted,num_indices[unsuccessful_indices])
            delta[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] = delta_p[
                check.nonzero().squeeze(1)]
            proj_success_rate[unsuccessful_indices] += check.float()

            # the random sign flip step
            s = torch.bernoulli(prob) * 2 - 1
            delta_s = delta * resize(s, h, w).sign()
            if unsuccessful_face_indices.sum() > 0:
                x_a[unsuccessful_face_indices] = (
                self.geoDict.project_adv_perturbation(x[unsuccessful_face_indices], ver_lst[unsuccessful_face_indices],
                                                      depth_lst[unsuccessful_face_indices],
                                                      self.geoDict.convert_uv_2_ncc(
                                                          delta_s[unsuccessful_face_indices]))).clamp(0, 1)
            if unsuccessful_background_indices.sum() > 0:
                x_a[unsuccessful_background_indices] = (
                        x[unsuccessful_background_indices] + delta_s[unsuccessful_background_indices]).clamp(0, 1)


            check = self.is_adversarial(x_a[unsuccessful_indices], y[unsuccessful_indices], targeted,num_indices[unsuccessful_indices])
            prob[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] -= s[check.nonzero().squeeze(
                1)] * 1e-4
            prob.clamp_(0.99, 0.9999)
            flip_success_rate[unsuccessful_indices] += check.float()
            delta[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] = delta_s[
                check.nonzero().squeeze(1)]

            # hyper-parameter adjustment
            if reset % 10 == 0:
                proj_success_rate /= reset
                flip_success_rate /= reset
                alpha[proj_success_rate > 0.7] *= 1.5
                alpha[proj_success_rate < 0.3] /= 1.5
                prob[flip_success_rate > 0.7] -= 0.001
                prob[flip_success_rate < 0.3] += 0.001
                prob.clamp_(0.99, 0.9999)
                reset *= 0
                proj_success_rate *= 0
                flip_success_rate *= 0
            # query count
            # q_num += 2
            # update indices for unsuccessful perturbations


        if background_attack_batch_size>0 and self.use_dict:
            self.imgDict.add_item(face_features,delta[background_attack_indices], self.only_one)
        if b - background_attack_batch_size > 0 and self.use_dict:
            self.geoDict.add_item(face_features,delta[~background_attack_indices], self.only_one)


        return x_a.clamp(0, 1), Q
    ###############################################################

    # def SFA(self, x, y, x_s=None, resize_factor=2.,targeted=False, max_queries=1000, sharing=False):
    def attack_untargeted(self, x_0, y_0, query_limit=20000):

        adv_img, q = self.SFA(x=x_0,
                         y=y_0,
                         x_s=None,
                         resize_factor=2.,
                         targeted=False,
                         max_queries=query_limit)
        return adv_img
    def attack_targeted(self, x_0, y_0, target_img,query_limit=20000):
        adv_img, q = self.SFA(x=x_0,
                         y=y_0,
                         x_s=target_img,
                         resize_factor=2.,
                         targeted=True,
                         max_queries=query_limit)
        return adv_img

###############################################################

