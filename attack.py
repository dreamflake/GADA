
import torch
import torch.nn as nn
import torch.optim as optim
import math
from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from backbone.model_irse_before import IR_50
from _facenet_pytorch.inception_resnet_v1 import InceptionResnetV1
from blackbox_model import BlackBoxModel
from modified_art.pytorch import PyTorchClassifier
from attacks.Sign_OPT import OPT_attack_sign_SGD
from attacks.SFA_GeoDict import SFA_Geo_Attack
from attacks.Evolutionary_GeoDict import Evolutionary_Geo_Attack
from util.utils import master_seed
import os
import time
import numpy as np
import argparse
from modified_art.hop_skip_jump import HopSkipJump


parser = argparse.ArgumentParser(description='Runs GADA')
parser.add_argument('-c', '--config', type=str, default='_3DDFA_V2/configs/mb1_120x120.yml')

parser.add_argument('--model', type=int, default=2, help='index of configurations')
parser.add_argument('--dict_model', type=int, default=3, help='index of configurations')

parser.add_argument('--attack', type=str,  default='EAGD', help='attack method')
parser.add_argument('--dataset', type=str, default='LFW', help='dataset')

parser.add_argument('--defense', type=str, default='none', help='defense method') # SD for Stateful Detection

parser.add_argument('--result_dir', type=str, default='results', help='directory for saving results')
parser.add_argument('--batch_size', type=int, default=1, help='number of image samples')

parser.add_argument('--max_num_queries', type=int, default=10000, help='maximum number of queries')
parser.add_argument('--log_interval', type=int, default=1000, help='log interval')
parser.add_argument('--num_imgs', type=int, default=500, help='number of test images')


parser.add_argument('--seed', type=int, default=1234, help='seed')
parser.add_argument('--attack_batch_size', type=int, default=100, help='Internal batch size for HSJA')

parser.add_argument('--resume', action='store_true', help='resume attack')
parser.add_argument('--targeted',  action='store_true', help='perform impersonation attack')

parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')

args = parser.parse_args()


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# args.targeted=True
print(args)
models={1:'IR101',2:'IR50',3:'FaceNet'}
def plot_img(img_tensor,file_name):
    img = np.array(img_tensor[0]).transpose(1, 2, 0) * 255. #.cpu().numpy()
    img = img.astype(np.uint8)

    height, width = img.shape[:2]

    from PIL import Image
    im = Image.fromarray(img)
    im.save("imgs/" + file_name + ".png")

if __name__ == '__main__':

    with torch.no_grad():
        #======= hyperparameters & data loaders =======#
        cfg = configurations[args.model]
        torch.backends.cudnn.benchmark = True
        SEED = args.seed # random seed for reproduce results
        master_seed(SEED)
        #torch.manual_seed(SEED)

        DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
        BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint

        BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        INPUT_SIZE = cfg['INPUT_SIZE']
        BATCH_SIZE = cfg['BATCH_SIZE']
        EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
        GPU_ID = cfg['TEST_GPU_ID'] # specify your GPU ids
        print("Overall Configurations:")
        print(cfg)
        savefile = '%s/%s_%d_%d_%s_%s_%s_%s.pth' % (
            args.result_dir, args.dataset,args.num_imgs,args.targeted,args.attack, BACKBONE_NAME, args.defense, args.save_suffix)
        dict_savefile = '%s/%s_%d_%d_%s_%s_%s_%s.dict' % (
            args.result_dir, args.dataset, args.num_imgs, args.targeted, args.attack, BACKBONE_NAME, args.defense,
            args.save_suffix)

        print('SAVE_FILE : ', savefile)
        #======= model =======#
        BACKBONE_DICT = {'ResNet_50': ResNet_50,
                         'ResNet_101': ResNet_101,
                         'ResNet_152': ResNet_152,
                         'IR_50': IR_50,
                         'IR_101': IR_101,
                         'IR_152': IR_152,
                         'IR_SE_50': IR_SE_50,
                         'IR_SE_101': IR_SE_101,
                         'IR_SE_152': IR_SE_152,
                         'FaceNet':InceptionResnetV1,
                         }
        print("=" * 60)
        print("{} Backbone Generated".format(BACKBONE_NAME))
        print("=" * 60)
        if BACKBONE_NAME == 'FaceNet':
            BACKBONE = BACKBONE_DICT[BACKBONE_NAME](pretrained='vggface2')
        else:
            BACKBONE = BACKBONE_DICT[BACKBONE_NAME](INPUT_SIZE)
            if BACKBONE_RESUME_ROOT:
                print("=" * 60)
                if os.path.isfile(BACKBONE_RESUME_ROOT):
                    print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
                    BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
                else:
                    print("No Checkpoint Found at '{}'".format(BACKBONE_RESUME_ROOT))
                    exit()
                print("=" * 60)
        BACKBONE.to(device)

        ###############################################
        # ======= hyperparameters & data loaders =======#
        dict_cfg = configurations[args.dict_model]

        DICT_BACKBONE_NAME = dict_cfg['BACKBONE_NAME'] 
        # ======= model =======#
        print("=" * 60)
        print("{} Dict Model Generated".format(DICT_BACKBONE_NAME))
        print("=" * 60)
        if DICT_BACKBONE_NAME == 'FaceNet':
            DICT_BACKBONE = BACKBONE_DICT[DICT_BACKBONE_NAME](pretrained='vggface2')
        else:
            print('Cannot support other models for dict models!!')
        DICT_BACKBONE.to(device)

        ###############################################
        # MODEL Load Complete #########################


        if args.dataset=='LFW':
            checkpoint = torch.load('LFW_'+str(500)+'_DATA.pth')
            images = checkpoint['tp_images'].numpy()
            labels = checkpoint['tp_labels'].numpy()
            threshold=cfg['LFW_THRESHOLD']
            shifted_idx = checkpoint['shift_idx'].numpy()
        elif args.dataset=='CPLFW':
            print('CPLFW')
            checkpoint = torch.load('CPLFW_'+str(500)+'_DATA.pth')
            images = checkpoint['tp_images'].numpy()
            labels = checkpoint['tp_labels'].numpy()
            threshold=cfg['CPLFW_THRESHOLD']
            shifted_idx = checkpoint['shift_idx'].numpy()
        dataset_imgs=images*0.5+0.5 # -1~1 -> 0~1
        shifted_images = dataset_imgs[shifted_idx * 2]
        images=dataset_imgs[0::2]
        pair_images=dataset_imgs[1::2]



        RGB_MEAN =cfg['RGB_MEAN']
        RGB_STD = cfg['RGB_STD']
        DATASET_MEAN = np.reshape(np.array(RGB_MEAN), [1, 3, 1, 1])
        DATASET_STD = np.reshape(np.array(RGB_STD), [1, 3, 1, 1])

        model = BlackBoxModel(BACKBONE, defense=args.defense, threshold=threshold, mean=RGB_MEAN, std=RGB_STD,stateful_detection=args.defense=='SD').to(device)

        model.eval()

        RGB_MEAN =dict_cfg['RGB_MEAN']
        RGB_STD = dict_cfg['RGB_STD']
        if args.dataset=='LFW':
            threshold=dict_cfg['LFW_THRESHOLD']
        elif args.dataset=='CPLFW':
            threshold=dict_cfg['CPLFW_THRESHOLD']
        dict_model = BlackBoxModel(DICT_BACKBONE, defense=args.defense, threshold=threshold, mean=RGB_MEAN, std=RGB_STD).to(device)
        dict_model.eval()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        Single_Attacks= ['HSJA', 'GD', 'SO']
        if args.attack in Single_Attacks:
            args.batch_size=1


        classifier = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 112, 112),
            nb_classes=2,
            preprocessing=(DATASET_MEAN, DATASET_STD)
        )

        attack_setting = {}
        attack_setting['attack'] = args.attack
        attack_setting['log_interval'] = args.log_interval
        attack_setting['max_num_queries'] = args.max_num_queries



        timestart = time.time()
        if args.attack == 'SO':
            attack = OPT_attack_sign_SGD(model)
        elif args.attack == 'EA':
            attack = Evolutionary_Geo_Attack(model,dict_model,dimension_reduction=(60,60),use_geo=False, use_dict=False)
        elif args.attack == 'EAR':
            attack = Evolutionary_Geo_Attack(model,dict_model,dimension_reduction=(60,60),use_geo=False, use_dict=False,random_background=True)
        elif args.attack == 'EAD':
            attack = Evolutionary_Geo_Attack(model, dict_model, dimension_reduction=(60, 60), use_geo=False,
                                             use_dict=True)
        elif args.attack == 'EADO':
            attack = Evolutionary_Geo_Attack(model, dict_model, dimension_reduction=(60, 60), use_geo=False,
                                             use_dict=True,only_one=True)
        elif args.attack == 'EAG':
            attack = Evolutionary_Geo_Attack(model,dict_model,dimension_reduction=(60,60),use_geo=True, use_dict=False)
        elif args.attack == 'EAGR':
            attack = Evolutionary_Geo_Attack(model,dict_model,dimension_reduction=(60,60),use_geo=True, use_dict=False,random_background=True)
        elif args.attack == 'EAGD':
            attack = Evolutionary_Geo_Attack(model, dict_model, dimension_reduction=(60, 60), use_geo=True,
                                             use_dict=True)
        elif args.attack == 'EAGDR':
            attack = Evolutionary_Geo_Attack(model, dict_model, dimension_reduction=(60, 60), use_geo=True,
                                             use_dict=True,random_background=True)
        elif args.attack == 'EAGDO':
            attack = Evolutionary_Geo_Attack(model, dict_model, dimension_reduction=(60, 60), use_geo=True,
                                             use_dict=True,only_one=True)

        elif args.attack == 'SFA':
            attack = SFA_Geo_Attack(model, dict_model, use_geo=False, use_dict=False)
        elif args.attack == 'SFAD':
            attack = SFA_Geo_Attack(model, dict_model, use_geo=False, use_dict=True)
        elif args.attack == 'SFADO':
            attack = SFA_Geo_Attack(model, dict_model, use_geo=False, use_dict=True,only_one=True)
        elif args.attack == 'SFAG':
            attack = SFA_Geo_Attack(model, dict_model, use_geo=True, use_dict=False)
        elif args.attack == 'SFAGD':
            attack = SFA_Geo_Attack(model,dict_model, use_geo=True, use_dict=True)
        elif args.attack == 'SFAGDO':
            attack = SFA_Geo_Attack(model, dict_model, use_geo=True, use_dict=True,only_one=True)


        start_idx=0

        if args.resume == True and os.path.exists(savefile):
            log_data = torch.load(savefile)
            total_num_queries = log_data['total_num_queries']
            total_log_l_2 = log_data['total_log_l_2']
            total_log_l_inf = log_data['total_log_l_inf']
            total_log_is_adversarial = log_data['total_log_is_adversarial']
            total_log_dist = log_data['total_log_dist']
            total_log_last_adv_imgs = log_data['total_log_last_adv_imgs']
            total_log_best_l_2_adv_imgs = log_data['total_log_best_l_2_adv_imgs']
            total_log_best_l_inf_adv_imgs = log_data['total_log_best_l_inf_adv_imgs']
            total_log_best_l_2 = log_data['total_log_best_l_2']
            total_log_best_l_inf = log_data['total_log_best_l_inf']
            start_idx = log_data['upper_idx'] // args.batch_size
            if args.defense == 'SD':
                total_log_k_avg_dist = log_data['total_log_k_avg_dist']
            print('Resume at ', start_idx)
            if args.attack == 'EAD' or args.attack == 'EAGD' or args.attack == 'EADO' or args.attack == 'EAGDR' or args.attack == 'EAGDO' or args.attack == 'SFAD' or args.attack == 'SFAGD' or args.attack == 'SFAGDO' or args.attack == 'SFADO':
                log_data = torch.load(dict_savefile)
                attack.imgDict.img_feature_dict = log_data['ImgDict_img_feature_dict']
                attack.imgDict.img_dict = log_data['ImgDict_img_dict']
                attack.geoDict.uv_dict = log_data['GeoDict_uv_dict']
                attack.geoDict.img_feature_dict = log_data['GeoDict_img_feature_dict']
        else:
            print('Initialize!')
            total_num_queries = torch.zeros(args.num_imgs).long()
            total_log_l_2 = -torch.ones(args.num_imgs,args.max_num_queries)
            total_log_l_inf = -torch.ones(args.num_imgs, args.max_num_queries)
            total_log_is_adversarial = -torch.ones(args.num_imgs, args.max_num_queries).byte()
            total_log_dist = -torch.ones(args.num_imgs, args.max_num_queries)
            total_log_last_adv_imgs = (torch.zeros(args.num_imgs, args.max_num_queries // args.log_interval, 3, 112, 112)).byte()
            total_log_best_l_2_adv_imgs = (torch.zeros(args.num_imgs, args.max_num_queries // args.log_interval, 3, 112, 112)).byte()
            total_log_best_l_inf_adv_imgs = (torch.zeros(args.num_imgs, args.max_num_queries // args.log_interval, 3, 112,112)).byte()
            total_log_best_l_2 = -torch.ones(args.num_imgs, args.max_num_queries // args.log_interval)
            total_log_best_l_inf = -torch.ones(args.num_imgs, args.max_num_queries // args.log_interval)
            if args.defense=='SD':
                total_log_k_avg_dist = -torch.ones(args.num_imgs, args.max_num_queries)


        # args.batch_size=10

        N = int(math.floor(float(args.num_imgs) / float(args.batch_size)))
        for i in range(start_idx,N):

            lower= (i * args.batch_size)

            upper = min((i + 1) * args.batch_size, args.num_imgs)
            images_batch = images[(i * args.batch_size):upper]
            images_batch = images_batch[:, [2, 1, 0], :, :] # RGB -> BGR
            pair_images_batch = pair_images[(i * args.batch_size):upper]
            pair_images_batch = pair_images_batch[:, [2, 1, 0], :, :]

            if args.targeted==True:
                target_images_batch = shifted_images[(i * args.batch_size):upper]
                target_images_batch = target_images_batch[:, [2, 1, 0], :, :]
            labels_batch = labels[(i * args.batch_size):upper]

            x_adv = None
            # plot_img(images_batch,str(i)+'l')
            # plot_img(target_images_batch,str(i)+'r')
            # continue
            #

            print('IMG: ', (i + 1),flush=True)


            ori_image = images_batch
            # print(target_images_batch.shape)
            if args.attack=='EAGDS':
                target_idx=0
                pair_images_batch[:]=pair_images_batch[target_idx]
                labels_batch[:]=labels_batch[target_idx]

            if args.targeted:
                target_images=target_images_batch
                model.init_model(attack_setting, clean_xs=target_images, pair_imgs=pair_images_batch,
                             ys=labels_batch,targeted=args.targeted)
            else:
                model.init_model(attack_setting, clean_xs=ori_image, pair_imgs=pair_images_batch,
                             ys=labels_batch,targeted=args.targeted)

            if args.attack == 'HSJA':
                if args.targeted==True:
                    attack = HopSkipJump(classifier=classifier, max_queries=args.max_num_queries, targeted=True,
                                         max_iter=64,
                                         max_eval=10000, init_eval=100)
                    attack.batch_size = args.num_imgs
                    x_adv = attack.generate(x=target_images_batch, x_adv_init=ori_image, y=labels_batch, resume=False)
                    print(classifier.get_num_queries())
                else:
                    attack = HopSkipJump(classifier=classifier, max_queries=args.max_num_queries, targeted=False,
                                         max_iter=64,
                                         max_eval=10000, init_eval=100)
                    attack.batch_size = args.num_imgs
                    x_adv = attack.generate(x=ori_image, x_adv_init=x_adv, y=labels_batch, resume=False)
            elif args.attack == 'SO':
                if args.targeted == True:
                    adv = attack(torch.FloatTensor(images_batch).to(device), labels_batch,
                                 torch.FloatTensor(target_images_batch).to(device), query_limit=args.max_num_queries,
                                 TARGETED=args.targeted)
                else:
                    adv = attack(torch.FloatTensor(images_batch).to(device), labels_batch, query_limit=args.max_num_queries,
                                 TARGETED=args.targeted)
            elif args.attack == 'EA' or args.attack == 'EAG' or args.attack == 'EAD'or args.attack == 'EAR' or args.attack == 'EADR'or args.attack == 'EADO' or args.attack == 'EAGD'or args.attack == 'EAGDO'or args.attack == 'EAGDR' or args.attack=='EAGR':
                if args.targeted:
                    adv = attack.attack_targeted(torch.FloatTensor(target_images_batch),
                                                 torch.LongTensor(labels_batch), torch.FloatTensor(images_batch),
                                                 query_limit=args.max_num_queries)
                else:
                    adv = attack.attack_untargeted(torch.FloatTensor(images_batch),
                                                   torch.LongTensor(labels_batch),
                                                   query_limit=args.max_num_queries)

            elif args.attack == 'EAGDS':
                if args.targeted:
                    adv = attack.attack_targeted(torch.FloatTensor(target_images_batch),
                                                 torch.LongTensor(labels_batch), torch.FloatTensor(images_batch),
                                                 query_limit=args.max_num_queries)
                else:
                    adv = attack.attack_untargeted(torch.FloatTensor(images_batch),
                                                   torch.LongTensor(labels_batch), target_idx=target_idx,
                                                   query_limit=args.max_num_queries)
            elif args.attack=='SFA' or args.attack=='SFAD' or args.attack=='SFAG' or args.attack=='SFAGD' or args.attack=='SFADO' or args.attack=='SFAGDO':
                if args.targeted == True:
                    adv = attack.attack_targeted(torch.FloatTensor(target_images_batch),
                                                   torch.LongTensor(labels_batch),
                                                 torch.FloatTensor(images_batch),
                                                   query_limit=args.max_num_queries)
                else:
                    adv = attack.attack_untargeted(torch.FloatTensor(images_batch),
                                                   torch.LongTensor(labels_batch),
                                                   query_limit=args.max_num_queries)

            if args.defense == 'SD': # Stateful detection
                (log_num_queries, log_l_2, log_l_inf, log_is_adversarial, log_dist, log_best_l_2, log_best_l_inf, log_last_adv_imgs,
                 log_best_l_2_adv_imgs,log_best_l_inf_adv_imgs,log_k_avg_dist) = model.get_log()
                total_log_k_avg_dist[lower:upper] =log_k_avg_dist
            else:
                (log_num_queries, log_l_2, log_l_inf, log_is_adversarial, log_dist, log_best_l_2, log_best_l_inf, log_last_adv_imgs,
                 log_best_l_2_adv_imgs,log_best_l_inf_adv_imgs) = model.get_log()

            total_num_queries[lower:upper] =log_num_queries
            total_log_l_2[lower:upper] = log_l_2
            total_log_l_inf[lower:upper] =log_l_inf
            total_log_is_adversarial[lower:upper]  =log_is_adversarial
            total_log_dist[lower:upper] =log_dist
            total_log_last_adv_imgs[lower:upper] =log_last_adv_imgs
            total_log_best_l_2_adv_imgs[lower:upper] =log_best_l_2_adv_imgs
            total_log_best_l_inf_adv_imgs[lower:upper] = log_best_l_inf_adv_imgs
            total_log_best_l_2[lower:upper] = log_best_l_2
            total_log_best_l_inf[lower:upper] = log_best_l_inf

            print('best l_2', torch.mean(total_log_best_l_2[:upper,torch.max(total_num_queries[:upper]//args.log_interval,other=torch.ones(1).long())-1]), 'best l_inf', torch.mean(total_log_best_l_inf[:upper,torch.max(total_num_queries[:upper]//args.log_interval,other=torch.ones(1).long())-1]))
            if (np.arange(lower+1,upper+1).astype(np.int)%50==0).sum()>0:
                print('Saving at ',upper)
                if args.defense=='SD':
                    torch.save({'total_num_queries': total_num_queries,
                                'total_log_l_2': total_log_l_2,
                                'total_log_l_inf': total_log_l_inf,
                                'total_log_is_adversarial': total_log_is_adversarial,
                                'total_log_dist': total_log_dist,
                                'total_log_k_avg_dist': total_log_k_avg_dist,
                                'total_log_last_adv_imgs': total_log_last_adv_imgs,
                                'total_log_best_l_2_adv_imgs': total_log_best_l_2_adv_imgs,
                                'total_log_best_l_inf_adv_imgs': total_log_best_l_inf_adv_imgs,
                                'total_log_best_l_2': total_log_best_l_2,
                                'total_log_best_l_inf': total_log_best_l_inf,
                                'upper_idx': upper,
                                }, savefile)
                else:
                    torch.save({'total_num_queries': total_num_queries,
                                'total_log_l_2': total_log_l_2,
                                'total_log_l_inf': total_log_l_inf,
                                'total_log_is_adversarial': total_log_is_adversarial,
                                'total_log_dist': total_log_dist,
                                'total_log_last_adv_imgs': total_log_last_adv_imgs,
                                'total_log_best_l_2_adv_imgs': total_log_best_l_2_adv_imgs,
                                'total_log_best_l_inf_adv_imgs': total_log_best_l_inf_adv_imgs,
                                'total_log_best_l_2': total_log_best_l_2,
                                'total_log_best_l_inf': total_log_best_l_inf,
                                'upper_idx': upper,
                                }, savefile)
                if args.attack == 'EAD' or args.attack == 'EAGD'or args.attack == 'EADO'or args.attack == 'EAGDR'or args.attack == 'EAGDO'or args.attack == 'SFAD' or  args.attack == 'SFAGD' or  args.attack == 'SFAGDO' or  args.attack == 'SFADO':
                    torch.save({'ImgDict_img_feature_dict': attack.imgDict.img_feature_dict,
                                'ImgDict_img_dict': attack.imgDict.img_dict,
                                'GeoDict_img_feature_dict': attack.geoDict.img_feature_dict,
                                'GeoDict_uv_dict': attack.geoDict.uv_dict
                                }, dict_savefile)

        timeend = time.time()
        print("\nTime: %.4f seconds" % (timeend - timestart))
