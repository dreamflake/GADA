



import torch
import os
# dir_list=os.listdir("./total/")
# print(dir_list)
import math
import numpy as np
import glob
import argparse

# import matplotlib as mpl
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


mode='ImageNet'
save_suffix=''
parser = argparse.ArgumentParser(description='Runs GADA')
parser.add_argument('-c', '--config', type=str, default='_3DDFA_V2/configs/mb1_120x120.yml')
parser.add_argument('-f', '--img_fp', type=str, default='_3DDFA_V2/examples/inputs/trump_hillary.jpg')
parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
parser.add_argument('-o', '--opt', type=str, default='pncc',
                    choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])

parser.add_argument('--onnx', action='store_true', default=True)
parser.add_argument('--model', type=int, default=2, help='index of configurations')
parser.add_argument('--dict_model', type=int, default=3, help='index of configurations')

parser.add_argument('--attack', type=str,  default='HSJA', help='attack method')
parser.add_argument('--defense', type=str, default='gaussian', help='defense method')

parser.add_argument('--dataset', type=str, default='LFW', help='dataset')
parser.add_argument('--result_dir', type=str, default='results', help='directory for saving results')
parser.add_argument('--sampled_image_dir', type=str, default='save', help='directory to cache sampled images')

parser.add_argument('--batch_size', type=int, default=1, help='number of image samples')




parser.add_argument('--max_num_queries', type=int, default=10000, help='number of image samples')
parser.add_argument('--log_interval', type=int, default=1000, help='number of image samples')
parser.add_argument('--num_runs', type=int, default=500, help='number of image samples')


parser.add_argument('--seed', type=int, default=1234, help='number of image samples')
parser.add_argument('--attack_batch_size', type=int, default=100, help='number of image samples')

parser.add_argument('--resume', type=bool, default=False, help='resume attack')
parser.add_argument('--targeted', type=bool, default=True, help='perform targeted attack')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')

parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')

args = parser.parse_args()
def get_pth_file_all(directory_path,ext=".pth"):
    return glob.glob(os.path.abspath(directory_path)+"/*"+ext)
def mean(l, logarithm=False):
    if logarithm:
        l_lg = [np.log(x) for x in l]
        return np.exp(sum(l_lg) / len(l_lg))
    else:
        return sum(l) / len(l)

if __name__=='__main__':


    file_list = get_pth_file_all('./')  # full path
    name_list = [os.path.split(x)[1] for x in file_list]
    print(name_list)

    def load_and_plot_linf(file_name, start_img_idx=0, end_img_idx=500, start_iter=100, end_iter=10000, threshold=0.05,
                      logarithm=False):
        log_data = torch.load(file_name)
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

        end_idx = log_data['upper_idx'] // args.batch_size
        print(file_name, end_idx)
        end_idx = min(end_idx, end_img_idx)
        best_log_l_inf = torch.zeros_like(total_log_l_inf[start_img_idx:end_idx])
        best_val = torch.ones(end_idx - start_img_idx) * 5e1
        for i in range(end_iter):
            best_val = torch.min(best_val, 5e1 * (
                        1 - total_log_is_adversarial[start_img_idx:end_idx, i]) + total_log_is_adversarial[
                                                                                  start_img_idx:end_idx,
                                                                                  i] * total_log_l_inf[
                                                                                       start_img_idx:end_idx, i])
            best_log_l_inf[:, i] = best_val
        avg_X = torch.arange(start_iter, end_iter)
        if logarithm == True:
            avg_Y = torch.exp(torch.mean(torch.log(best_log_l_inf[:, start_iter:end_iter]), 0))
        else:
            avg_Y = torch.mean(best_log_l_inf[:, start_iter:end_iter], 0)
        target_idx = [1000, 2000, 5000, 10000]

        norms = torch.mean(best_log_l_inf, 0)[torch.LongTensor(target_idx) - 1].numpy()

        thresholds = [0.05, 0.03]
        Q = np.zeros(len(thresholds))
        for i, t in enumerate(thresholds):
            best_log_l_inf_t = best_log_l_inf.clone()
            best_log_l_inf_t[best_log_l_inf <= t] = 0
            best_log_l_inf_t[best_log_l_inf > t] = 1
            Q[i] = ((torch.sum(best_log_l_inf_t) / (end_idx - start_img_idx)).item())
        return avg_X, avg_Y, norms, Q

    def load_and_plot(file_name,start_img_idx=0,end_img_idx=500,start_iter=100,end_iter=10000, threshold=4,logarithm=True):
        log_data=torch.load(file_name)
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

        end_idx = log_data['upper_idx'] // args.batch_size
        print(file_name,end_idx)
        end_idx=min(end_idx,end_img_idx)
        best_log_l2=torch.zeros_like(total_log_l_2[start_img_idx:end_idx])
        best_val=torch.ones(end_idx-start_img_idx)*5e1
        for i in range(end_iter):
            best_val=torch.min(best_val,5e1*(1-total_log_is_adversarial[start_img_idx:end_idx,i])+total_log_is_adversarial[start_img_idx:end_idx,i]*total_log_l_2[start_img_idx:end_idx,i])
            best_log_l2[:,i]=best_val
        avg_X=torch.arange(start_iter,end_iter)
        if logarithm==True:
            avg_Y=torch.exp(torch.mean(torch.log(best_log_l2[:,start_iter:end_iter]),0))
        else:
            avg_Y=torch.mean(best_log_l2[:,start_iter:end_iter],0)
        target_idx=[1000,2000,5000,10000]

        norms=torch.mean(best_log_l2,0)[torch.LongTensor(target_idx)-1].numpy()

        thresholds=[4,2]
        Q=np.zeros(len(thresholds))
        for i,t in enumerate(thresholds):
            best_log_l2_t=best_log_l2.clone()
            best_log_l2_t[best_log_l2<=t]=0
            best_log_l2_t[best_log_l2>t]=1
            Q[i]=((torch.sum(best_log_l2_t)/(end_idx-start_img_idx)).item())
        return avg_X,avg_Y,norms,Q


    def load_and_plot_SD(file_name, start_img_idx=0, end_img_idx=500, start_iter=100, end_iter=10000, threshold=4,
                      logarithm=True):
        log_data = torch.load(file_name)
        total_num_queries = log_data['total_num_queries']
        total_log_l_2 = log_data['total_log_l_2']
        total_log_l_inf = log_data['total_log_l_inf']
        total_log_is_adversarial = log_data['total_log_is_adversarial']
        total_log_dist = log_data['total_log_dist']
        total_log_last_adv_imgs = log_data['total_log_last_adv_imgs']
        total_log_best_l_2_adv_imgs = log_data['total_log_best_l_2_adv_imgs']
        total_log_best_l_inf_adv_imgs = log_data['total_log_best_l_inf_adv_imgs']
        total_log_k_avg_dist = log_data['total_log_k_avg_dist']
        total_log_best_l_inf = log_data['total_log_best_l_inf']

        end_idx = log_data['upper_idx'] // args.batch_size
        print(file_name, end_idx)
        end_idx = min(end_idx, end_img_idx)
        best_log_l2 = torch.zeros_like(total_log_l_2[start_img_idx:end_idx])
        best_val = torch.ones(end_idx - start_img_idx) * 5e1
        for i in range(end_iter):
            best_val = torch.min(best_val, 5e1 * (
                        1 - total_log_is_adversarial[start_img_idx:end_idx, i]) + total_log_is_adversarial[
                                                                                  start_img_idx:end_idx,
                                                                                  i] * total_log_l_2[
                                                                                       start_img_idx:end_idx, i])
            best_log_l2[:, i] = best_val
        avg_X = torch.arange(start_iter, end_iter)
        avg_Y = best_log_l2[10, start_iter:end_iter]

        target_idx = [2000, 5000, 10000]

        norms = torch.mean(best_log_l2, 0)[torch.LongTensor(target_idx) - 1].numpy()

        total_log_k_avg_dist_t=total_log_k_avg_dist.clone()
        total_log_k_avg_dist_t[total_log_k_avg_dist < 0.002] = 1
        total_log_k_avg_dist_t[total_log_k_avg_dist >= 0.002] = 0

        avg_k_avg_dist=total_log_k_avg_dist[10,start_iter:end_iter]
        Q=torch.mean(torch.sum(total_log_k_avg_dist_t,1)).item()

        return avg_X, avg_Y, avg_k_avg_dist,norms, Q

    def load_and_plot_Dist(file_name, start_img_idx=0, end_img_idx=500, start_iter=100, end_iter=10000, threshold=4,
                      logarithm=True):
        log_data = torch.load(file_name)
        total_num_queries = log_data['total_num_queries']
        total_log_l_2 = log_data['total_log_l_2']
        total_log_l_inf = log_data['total_log_l_inf']
        total_log_is_adversarial = log_data['total_log_is_adversarial']
        total_log_dist = log_data['total_log_dist']
        total_log_last_adv_imgs = log_data['total_log_last_adv_imgs']
        total_log_best_l_2_adv_imgs = log_data['total_log_best_l_2_adv_imgs']
        total_log_best_l_inf_adv_imgs = log_data['total_log_best_l_inf_adv_imgs']
        total_log_best_l_inf = log_data['total_log_best_l_inf']

        end_idx=log_data['upper_idx']
        end_idx = min(end_idx, end_img_idx)
        print(file_name, end_idx)
        best_log_l2 = torch.zeros_like(total_log_l_2[start_img_idx:end_idx,:end_iter])
        best_val = torch.ones(end_idx - start_img_idx) * 5e1
        for i in range(end_iter):
            best_val = torch.min(best_val, 5e1 * (
                        1 - total_log_is_adversarial[start_img_idx:end_idx, i]) + total_log_is_adversarial[
                                                                                  start_img_idx:end_idx,
                                                                                  i] * total_log_l_2[
                                                                                       start_img_idx:end_idx, i])
            best_log_l2[:, i] = best_val

        return best_log_l2
    def load_and_plot_IMG(file_name, start_img_idx=0, end_img_idx=500, start_iter=100, end_iter=10000, idx=0, title=False,first=0):
        log_data = torch.load(file_name)
        total_num_queries = log_data['total_num_queries']
        total_log_l_2 = log_data['total_log_l_2']
        total_log_l_inf = log_data['total_log_l_inf']
        total_log_is_adversarial = log_data['total_log_is_adversarial']
        total_log_dist = log_data['total_log_dist']
        total_log_last_adv_imgs = log_data['total_log_last_adv_imgs']
        total_log_best_l_2_adv_imgs = log_data['total_log_best_l_2_adv_imgs']
        total_log_best_l_inf_adv_imgs = log_data['total_log_best_l_inf_adv_imgs']
        total_log_best_l_inf = log_data['total_log_best_l_inf']

        end_idx = log_data['upper_idx'] // args.batch_size
        print(file_name, end_idx)
        end_idx = min(end_idx, end_img_idx)
        best_log_l2 = torch.zeros_like(total_log_l_2[idx])
        best_val = torch.ones(1) * 5e1
        for i in range(end_iter):
            best_val = torch.min(best_val, 5e1 * (
                        1 - total_log_is_adversarial[idx, i]) + total_log_is_adversarial[
                                                                                  idx,
                                                                                  i] * total_log_l_2[
                                                                                       idx, i])
            best_log_l2[i] = best_val


        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        #fig, ax = plt.subplots(figsize=(10, 1), tight_layout=True)
        im_idx=[0,1,2,5,9]

        fig, axs = plt.subplots(nrows=1, ncols=len(im_idx), gridspec_kw={'wspace': 0, 'hspace': 0},
                                   squeeze=True, tight_layout=True)
        for i,k in enumerate(im_idx):
            if first==0:
                axs[i].text(0.5 * (left + right), 1+0.1, '%dK'%(k+1),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=11, color='black',
                        transform=axs[i].transAxes)
            batch = total_log_best_l_2_adv_imgs[idx, k]
            x_adv_i = np.transpose(batch.numpy(), [1, 2, 0])
            #axs[i].set_title('%.2f'%best_log_l2[(k+1)*1000-1].item())
            axs[i].text(0.5 * (left + right), -0.1, '%.2f'%best_log_l2[(k+1)*1000-1].item(),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=11, color='black',
                    transform=axs[i].transAxes)


            axs[i].axis("off")
            axs[i].imshow(x_adv_i)
        # build a rectangle in axes coords

        axs[0].text(-0.1, 0.5 * (bottom + top), title,
                horizontalalignment='right',
                verticalalignment='center',
                rotation='vertical',
                    fontsize=11, color='black',
                transform=axs[0].transAxes)

    if False:
        img_indices=[190,258,298,430,455]
        for img_idx in img_indices:
            dt='CPLFW'
            targeted='1'
            attacks = ['SO','EA']
            model = 'IR_50'
            defense = 'gaussian'
            colors = ['#1f77b4',
                      '#ff7f0e',
                      '#2ca02c',
                      '#d62728',
                      '#9467bd',
                      '#8c564b',
                      '#e377c2',
                      '#7f7f7f',
                      '#bcbd22',
                      '#17becf',
                      '#1a55FF']
            # colors=['tab:blue','tab:red','g','tab:purple','tab:red','tab:cyan','tab:brown','tab:pink']

            PLOT_INFO = []
            for i, at in enumerate(attacks):
                fname = dt + '_500_' + targeted + '_' + at + '_' + model + '_' + defense + '_.pth'
                if os.path.exists(fname):
                    PLOT_INFO.append((fname, colors[i], at, [0, 500, 1000, 10000], ''))
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            avg_X = []
            avg_y = []
            plot_info = []
            labels = []
            for i, logs in enumerate(PLOT_INFO):
                at = logs[2]
                load_and_plot_IMG(logs[0], start_img_idx=logs[3][0], end_img_idx=logs[3][1],
                                  start_iter=logs[3][2], end_iter=logs[3][3], idx=img_idx, title=at, first=i)
                plt.savefig(dt + '_500_' + targeted + '_' + model + '_IMG_' + str(img_idx) + '_'+ at  + '.pdf', dpi=300, bbox_inches='tight')
                plt.close()




    #
    PLOT_INFO = [
    ('CPLFW_500_0_EA_IR_50_gaussian_.pth', 'm', 'EA',[0,500,1000,10000],''),
    ('CPLFW_500_0_EAGD_IR_50_gaussian_.pth', 'r', 'EAGD',[0,500,1000,10000],''),
    ]
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    avg_X = []
    avg_y = []
    labels = []
    for i, logs in enumerate(PLOT_INFO):
        l2_norm = load_and_plot_Dist(logs[0], start_img_idx=logs[3][0], end_img_idx=logs[3][1],
                                             start_iter=logs[3][2], end_iter=logs[3][3], logarithm=False)

        if i==1:
            print(torch.topk(l2_norm_o[:,-1]-l2_norm[:,-1],30,0))
            #print(torch.sum((l2_norm_o[:,-1]-l2_norm[:,-1])>0))
        l2_norm_o=l2_norm.clone()

    ##############################################
    if False:
        print('SD Graph')
        datasets = ['LFW']  # 'LFW',
        attacks = ['EA','EAR','EAG','EAGR']  #'SO', 'HSJA','GD',
        model = 'IR_50'
        defense = 'SD'
        colors = ['#1f77b4',
                  '#ff7f0e',
                  '#2ca02c',
                  '#d62728',
                  '#9467bd',
                  '#8c564b',
                  '#e377c2',
                  '#7f7f7f',
                  '#bcbd22',
                  '#17becf',
                  '#1a55FF']
        PLOT_INFO = []
        for i, at in enumerate(attacks):
            fname = 'LFW_100_0_' + at + '_' + model + '_' + defense + '_.pth'
            if os.path.exists(fname):
                PLOT_INFO.append((fname, colors[i], at, [0, 100, 500, 10000], ''))
        fig, ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)

        ax1.set_xlabel('# Queries', fontsize=12)
        ax1.set_ylabel('$\ell_2$ norm of perturbation', fontsize=12)
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('k-NN dist in stateful detection', fontsize=12)
        ax2.tick_params(axis='y')
        avg_X = []
        avg_y = []
        plot_info = []
        labels = []
        legends=[]
        legends_labels=[]
        for i, logs in enumerate(PLOT_INFO):
            x, y, k_avg_dist,norms,  detections = load_and_plot_SD(logs[0], start_img_idx=logs[3][0], end_img_idx=logs[3][1],
                                                 start_iter=logs[3][2], end_iter=logs[3][3], logarithm=False)
            x = x[::2].numpy()
            y = y[::2].numpy()
            k_avg_dist = k_avg_dist[::2]
            for n_item in norms:
                print("%.2f " % (n_item), end='')
            print("%.1f" %detections, end='')
            print()
            l=ax1.plot(x, y, logs[1], label=logs[2])
            r=ax2.plot(x, k_avg_dist, logs[1],label=logs[2], alpha=0.5) #linestyle='dashed',
            legends.append((l[0],r[0]))
            legends_labels.append(logs[2])
        threshold=0.002*np.ones_like(k_avg_dist)
        r = ax2.plot(x, threshold, 'k', label='Detection threshold', alpha=1)  # linestyle='dashed',
        legends.append((r[0]))
        legends_labels.append('Detection threshold')
        plt.legend(handles=legends,labels=legends_labels,numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #
        #ax2.set_ylim(0,1.5)
        ax2.set_yscale('log')
        #legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
        #legend.get_frame().set_facecolor('white')
        #ax2.ylim(0,0.1)
        plt.xticks([1000, 2000, 5000, 10000], ['1K', '2K', '5K', '10K'])
        plt.savefig('SD_Graph.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    ##############################################
    if False:
        datasets = ['LFW']  # 'LFW',
        attacks = ['EAGDR', 'EAGDO', 'EAGD']  #'EV', 'EVD', 'EVG','EV', 'EVD', 'EVG','EAGD'
        model = 'IR_50'
        defense = 'gaussian'
        colors = ['#1f77b4',
                  '#ff7f0e',
                  '#2ca02c',
                  '#d62728',
                  '#9467bd',
                  '#8c564b',
                  '#e377c2',
                  '#7f7f7f',
                  '#bcbd22',
                  '#17becf',
                  '#1a55FF']
        # colors=['tab:blue','tab:red','g','tab:purple','tab:red','tab:cyan','tab:brown','tab:pink']
        for targeted in ['0', '1']:
            for dt in datasets:
                PLOT_INFO = []
                for i, at in enumerate(attacks):
                    fname = dt + '_500_' + targeted + '_' + at + '_' + model + '_' + defense + '_.pth'
                    at=at.replace('EAGDO','EAGD1')
                    if os.path.exists(fname):
                        PLOT_INFO.append((fname, colors[i], at, [0, 500, 1000,3000], ''))
                fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

                avg_X = []
                avg_y = []
                plot_info = []
                labels = []
                for i, logs in enumerate(PLOT_INFO):
                    x, y, norms, queries = load_and_plot(logs[0], start_img_idx=logs[3][0], end_img_idx=logs[3][1],
                                                         start_iter=logs[3][2], end_iter=logs[3][3], logarithm=False)
                    x = x[::2].numpy()
                    y = y[::2].numpy()
                    for n_item in norms:
                        print("%.2f " % (n_item), end='')
                    for n_item in queries:
                        print("%d " % (n_item), end='')
                    print()
                    plt.plot(x, y, logs[1], label=logs[2])
                plt.legend()
                plt.title('Dodging attacks on the LFW dataset', fontsize=12)
                #plt.xticks([1000, 2000, 5000, 10000], ['1K', '2K', '5K', '10K'])
                plt.xticks([1000, 2000, 3000], ['1K', '2K', '3K'])
                ax.set_xlabel('# Queries', fontsize=12)
                #ax.set_yscale('log')
                ax.set_ylabel('$\ell_2$ norm of perturbation', fontsize=12)
                plt.savefig('Dodging_LFW.pdf', dpi=300, bbox_inches='tight')
                plt.show()

    # Main Table Results
    # LFW CPLFW
    # Impersonation Attack
    # Dogging Attack
    if True:
        datasets=['LFW','CPLFW'] #'LFW',


        attacks=['SO','HSJA','EA','EAD','EAG','EAGD']
        model='IR_101'
        defense='gaussian'
        colors = ['#1f77b4',
                  '#ff7f0e',
                  '#2ca02c',
                  '#d62728',
                  '#9467bd',
                  '#8c564b',
                  '#e377c2',
                  '#7f7f7f',
                  '#bcbd22',
                  '#17becf',
                  '#1a55FF']
        # colors=['tab:blue','tab:red','g','tab:purple','tab:red','tab:cyan','tab:brown','tab:pink']
        for targeted in ['0','1']:#
            for dt in datasets:
                if targeted == '1':
                    if dt=='LFW':
                        attacks = ['SO', 'HSJA', 'EA', 'EAG']  # 'GD'
                    else:
                        attacks = ['SO', 'HSJA', 'EA', 'EAG']  # 'GD'
                else:
                    attacks = ['SO', 'HSJA', 'EA', 'EAD', 'EAG', 'EAGD']  # 'GD'

                PLOT_INFO=[]
                for i, at in enumerate(attacks):
                    fname=dt+'_500_'+targeted+'_'+at+'_'+model+'_'+defense+'_.pth'
                    if os.path.exists(fname):
                        PLOT_INFO.append((fname,colors[i],at,[0,500,1000,10000],''))
                fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
                avg_X = []
                avg_y = []
                plot_info = []
                labels = []
                for i, logs in enumerate(PLOT_INFO):
                    x, y ,norms,queries= load_and_plot(logs[0], start_img_idx=logs[3][0], end_img_idx=logs[3][1],
                                              start_iter=logs[3][2], end_iter=logs[3][3], logarithm=False)
                    x = x[::10].numpy()
                    y = y[::10].numpy()
                    for n_item in norms:
                        print(" %.2f &"%(n_item),end='')
                    for n_item in queries:
                        print(" %d &"%( n_item), end='')
                    print()
                    at=logs[2]
                    plt.plot(x, y, logs[1], label=at)
                plt.legend()
                #plt.grid(alpha=0.3)
                plt.xticks([1000, 2000, 5000, 10000], ['1K', '2K', '5K', '10K'])
                if targeted=='1':
                    plt.title('Impersonation attacks on the '+dt+' dataset')
                else:
                    plt.title('Dodging attacks on the '+dt+' dataset')

                ax.set_xlabel('# Queries', fontsize=12)
                ax.set_ylabel('$\ell_2$ norm of perturbation', fontsize=12)
                plt.savefig(dt+'_'+targeted+'_'+model+'.pdf', dpi=300, bbox_inches='tight')
                plt.show()

    if False:
        datasets = ['LFW', 'CPLFW']  # 'LFW',
        attacks = ['SFA', 'SFAD', 'SFAG', 'SFAGD']  #
        model = 'IR_50'
        defense = 'gaussian'
        colors = ['#1f77b4',
                  '#ff7f0e',
                  '#2ca02c',
                  '#d62728',
                  '#9467bd',
                  '#8c564b',
                  '#e377c2',
                  '#7f7f7f',
                  '#bcbd22',
                  '#17becf',
                  '#1a55FF']
        for targeted in ['0', '1']:
            for dt in datasets:
                PLOT_INFO = []
                for i, at in enumerate(attacks):
                    fname = dt + '_500_' + targeted + '_' + at + '_' + model + '_' + defense + '_.pth'
                    if os.path.exists(fname):
                        PLOT_INFO.append((fname, colors[i], at, [0, 500, 1000, 10000], ''))
                fig, ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)

                ax1.set_xlabel('# Queries', fontsize=12)
                ax1.set_ylabel('$\ell_\infty$ norm of perturbation', fontsize=12)
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel('$\ell_2$ norm of perturbation', fontsize=12)
                ax2.tick_params(axis='y')
                avg_X = []
                avg_y = []
                plot_info = []
                labels = []
                legends = []
                legends_labels = []

                for i, logs in enumerate(PLOT_INFO):
                    x_2, y_2, norms, queries = load_and_plot(logs[0], start_img_idx=logs[3][0], end_img_idx=logs[3][1],
                                                             start_iter=logs[3][2], end_iter=logs[3][3],
                                                             logarithm=False)

                    x, y, norms, queries = load_and_plot_linf(logs[0], start_img_idx=logs[3][0], end_img_idx=logs[3][1],
                                                              start_iter=logs[3][2], end_iter=logs[3][3],
                                                              logarithm=False)
                    x = x[::10].numpy()
                    y = y[::10].numpy()
                    for n_item in norms:
                        print("%.4f " % (n_item), end='')
                    for n_item in queries:
                        print("%d " % (n_item), end='')
                    print()
                    l = ax1.plot(x, y, logs[1], label=logs[2])
                    r = ax2.plot(x_2, y_2, logs[1], label=logs[2], linestyle='dashed',alpha=0.6)  # linestyle='dashed',
                    legends.append((l[0], r[0]))
                    legends_labels.append(logs[2])

                plt.legend(handles=legends, labels=legends_labels, numpoints=1, handlelength=3,
                           handler_map={tuple: HandlerTuple(ndivide=None)})  #

                plt.xticks([1000, 2000, 5000, 10000], ['1K', '2K', '5K', '10K'])
                plt.savefig('L_infty_Graph.pdf', dpi=300, bbox_inches='tight')
                plt.show()
