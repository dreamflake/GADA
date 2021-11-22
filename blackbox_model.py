import torch
import numpy as np
import torch.nn as nn



from torch.distributions import Beta



import lpips





####################


import numpy as np
import sklearn.metrics.pairwise as pairwise



class Detector(object):


    def __init__(self, K, threshold=None, chunk_size=100):
        self.K = K
        self.threshold = threshold
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        ## Initializing the model
        self.lpips = lpips.LPIPS(net='squeeze', version=0.1)
        if True:
            self.lpips.to(device)
        self.encode = lambda x: [self.lpips(i) for i in x]

        if self.threshold is None and self.training_data is None:
            raise ValueError("Must provide explicit detection threshold or training data to calculate threshold!")

        self.num_queries = 0
        self.buffer = []
        self.chunk_size = chunk_size

        self.history = []  # Tracks number of queries (t) when attack was detected
        self.detected_dists = []  # Tracks knn-dist that was detected

    def process(self, queries):
        queries = self.encode(queries)
        k_avg_dist=torch.zeros(len(queries))

        for i,query in enumerate(queries):
            k_avg_dist[i]=float(self.process_query(query))
        return k_avg_dist

    def process_query(self, query):
        if  len(self.buffer) < self.K:
            self.buffer.append(query)
            self.num_queries += 1
            return 1

        k = self.K
        all_dists = []
        if len(self.buffer) > 0:
            for feat_b in self.buffer:
                all_dists.append(self.lpips.get_dist(query,feat_b))

        dists = np.concatenate(all_dists).squeeze()
        k_nearest_dists = np.partition(dists, k - 1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)

        if len(self.buffer) >= self.chunk_size:
            del self.buffer[0]
        self.buffer.append(query)
        self.num_queries += 1

        is_attack = k_avg_dist < self.threshold
        #print(k_avg_dist)
        if is_attack:
            # print('GOTCHA!', k_avg_dist)

            #self.history.append(self.num_queries)
            # self.history_by_attack.append(num_queries_so_far + 1)
            # self.detected_dists.append(k_avg_dist)
            # print(len(self.detected_dists))

            # print("[encoder] Attack detected:", str(self.history), str(self.detected_dists))
            self.clear_memory()
        return k_avg_dist

    def clear_memory(self):
        self.buffer = []



def calculate_thresholds(training_data, K, encoder=lambda x: x, P=1000, up_to_K=False):

    data = encoder(training_data)

    distances = []
    for i in range(data.shape[0] // P):
        distance_mat = pairwise.pairwise_distances(data[i * P:(i + 1) * P, :], Y=data)
        distance_mat = np.sort(distance_mat, axis=-1)
        distance_mat_K = distance_mat[:, :K]
        distances.append(distance_mat_K)
    distance_matrix = np.concatenate(distances, axis=0)
    start = 0 if up_to_K else K
    THRESHOLDS = []
    K_S = []
    for k in range(start, K + 1):
        dist_to_k_neighbors = distance_matrix[:, :k + 1]
        avg_dist_to_k_neighbors = dist_to_k_neighbors.mean(axis=-1)
        threshold = np.percentile(avg_dist_to_k_neighbors, 0.1)
        K_S.append(k)
        THRESHOLDS.append(threshold)
    return K_S, THRESHOLDS







def l2_normalize(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

# [0,1] -> [-1,1]
# mean, std [0.5, 0.5]
def model_noise(imgs_tensor, sigma=0.001, alpha=0, beta=0):
    RN = torch.randn_like(imgs_tensor)
    if alpha > 0 and beta > 0:
        m = Beta(torch.FloatTensor([alpha]), torch.FloatTensor([beta]))
        mm = m.sample((imgs_tensor.size()[0],)).view(imgs_tensor.size()[0], 1, 1, 1)
        RN = RN * sigma * mm
    else:
        RN = RN * sigma
    return RN


class BlackBoxModel(nn.Module):
    def __init__(self, model, mean, std, defense='gaussian',threshold=1, stateful_detection=True):
        super(BlackBoxModel, self).__init__()
        self.model = model
        self.mean=mean
        self.std=std
        self.attack_mode = False
        self.threshold=threshold
        self.targeted=False
        self.logging=True
        self.sigma=0
        self.l2_epsilon=5.
        self.linf_epsilon=0.031
        self.defense=defense
        self.stateful_detection=stateful_detection



    def invert_normalization(self,imgs):

        imgs_tensor = imgs.clone()
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = imgs_tensor[i, :, :] * self.std[i] + self.mean[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = imgs_tensor[:, i, :, :] * self.std[i] + self.mean[i]
        return imgs_tensor

    # applies the normalization transformations
    def apply_normalization(self,imgs):
        imgs_tensor = imgs.clone()
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - self.mean[i]) / self.std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - self.mean[i]) / self.std[i]
        return imgs_tensor
    def apply_defense(self,batch):

        batch_size = batch.size()[0]
        if self.sigma > 0:
            noised_batch = (batch + model_noise(batch, sigma=self.sigma, alpha=self.alpha,
                                               beta=self.beta)).clamp(0, 1)
        else:
            noised_batch = batch
        if self.defense == 'rnp':
            if batch_size == 1:
                rnd_size = np.random.randint(112, 124 + 1) # 232
                noised_batch = torch.nn.functional.upsample(noised_batch, size=(rnd_size, rnd_size), mode='nearest')
                second_max = 124 - rnd_size
                a = np.random.randint(0, second_max + 1)
                b = np.random.randint(0, second_max + 1)
                pads = (b, second_max - b, a, second_max - a)  # pad last dim by 1 on each side
                noised_batch = self.apply_normalization(noised_batch)
                resized_batch = torch.nn.functional.pad(noised_batch, pads, "constant", 0)  # effectively zero padding
            else:
                resized_batch = torch.zeros((batch_size, 3, 124, 124))
                for nn in range(batch_size):
                    cur_img = noised_batch[nn:nn + 1]
                    rnd_size = np.random.randint(112, 124 + 1)
                    cur_img = torch.nn.functional.upsample(cur_img, size=(rnd_size, rnd_size), mode='nearest')
                    second_max = 124 - rnd_size
                    a = np.random.randint(0, second_max + 1)
                    b = np.random.randint(0, second_max + 1)
                    pads = (b, second_max - b, a, second_max - a)  # pad last dim by 1 on each side
                    cur_img = self.apply_normalization(cur_img)
                    cur_img = torch.nn.functional.pad(cur_img, pads, "constant", 0)  # effectively zero padding
                    resized_batch[nn] = cur_img
            noised_batch=resized_batch
        else:
            noised_batch = self.apply_normalization(noised_batch)


        return noised_batch
    def forward(self, batch,batch_indices=0,unnormalization=True):

        if unnormalization == True:
            batch = self.invert_normalization(batch)
        batch = batch.clamp(0, 1)
        batch_size = batch.size()[0]
        batch=batch+torch.randn_like(batch)*0.01
        noised_batch = self.apply_defense(batch)


        output = l2_normalize((self.model(noised_batch)[0]))
        vec_output = torch.zeros(batch_size, 2)
        # if clean_batch is not None:
        #     if unnormalization == True:
        #         clean_batch = self.invert_normalization(clean_batch)
        #     clean_batch = clean_batch.clamp(0, 1)
        #     noised_clean_batch = self.apply_defense(clean_batch)
        #     clean_output = l2_normalize((self.model(noised_clean_batch)[0]))
        for n in range(self.num_images):
            # print(batch_idx,n)
            if (torch.ones(batch_size)[batch_indices==n]).sum()>0:
                selected_batch_idx=torch.arange(0,batch_size)[batch_indices==n].long().view(-1)
                selected_batch=batch[selected_batch_idx]


                selected_batch_size = selected_batch.size()[0]

                if self.num_queries[n]<self.max_num_queries:

                    self.num_queries[n] += selected_batch_size


                    dist = torch.sum(torch.pow(self.pair_img_features[n:n + 1] - output[selected_batch_idx], 2), 1)
                    pred = (dist < self.threshold).long()
                    vec_output[selected_batch_idx,pred]=1

                    perturbation = selected_batch - self.original_images[n:n+1]
                    l2_norm = perturbation.view(selected_batch_size, -1).norm(2, 1)
                    linf_norm = perturbation.view(selected_batch_size, -1).abs().max(1)[0]
                    start_idx= self.num_queries[n]-selected_batch_size
                    end_idx= min(self.num_queries[n],self.max_num_queries)

                    start_batch_idx=start_idx-self.num_queries[n]+selected_batch_size
                    end_batch_idx=end_idx-self.num_queries[n]+selected_batch_size

                    if self.stateful_detection:
                        self.log_k_avg_dist[n, start_idx:end_idx]  = self.detector[n].process(selected_batch)[start_batch_idx:end_batch_idx]
                    self.log_l_2[n,start_idx:end_idx] = l2_norm[start_batch_idx:end_batch_idx]
                    self.log_l_inf[n,start_idx:end_idx] = linf_norm[start_batch_idx:end_batch_idx]
                    if self.targeted==True:
                        self.log_is_adversarial[n,start_idx:end_idx] = (pred[start_batch_idx:end_batch_idx]==self.ys[n]).long() #
                    else:
                        self.log_is_adversarial[n,start_idx:end_idx] = (pred[start_batch_idx:end_batch_idx]!=self.ys[n]).long() #[n,start_idx-self.num_queries[n]:end_idx-self.num_queries[n]]
                    self.log_dist[n,start_idx:end_idx] = dist[start_batch_idx:end_batch_idx] #[n,start_idx-self.num_queries[n]:end_idx-self.num_queries[n]]
                   
                    for i in range(start_idx,end_idx):

                        if (i+1) % 1000 == 0:
                            if self.stateful_detection:
                                print(n,(i+1), self.best_l_2[n], self.best_l_inf[n],len(torch.nonzero(self.log_k_avg_dist[n, :end_idx]<0.002)), flush=True)
                            else:
                                print(n,(i+1), self.best_l_2[n], self.best_l_inf[n], flush=True)


                        if self.log_is_adversarial[n,i]==True:
                            self.log_last_adv_imgs[n]=(selected_batch[i-start_idx]*255.0).byte()
                            if self.log_l_2[n,i]<self.best_l_2[n]:
                                self.best_l_2[n]=self.log_l_2[n,i]
                                self.log_best_l_2_adv_img[n]=(selected_batch[i-start_idx]*255.0).byte()
                            if self.log_l_inf[n,i] < self.best_l_inf[n]:
                                self.best_l_inf[n] = self.log_l_inf[n,i]
                                self.log_best_l_inf_adv_img[n] = (selected_batch[i-start_idx]*255.0).byte()
                        if (i+1)%self.log_interval==0:
                            self.log_best_l_2_adv_imgs[n,(i+1)// self.log_interval-1] = self.log_best_l_2_adv_img[n]
                            self.log_best_l_2[n,(i+1)// self.log_interval-1]=self.best_l_2[n]
                            self.log_best_l_inf_adv_imgs[n,(i + 1) // self.log_interval - 1] = self.log_best_l_inf_adv_img[n]
                            self.log_best_l_inf[n,(i + 1) // self.log_interval - 1] = self.best_l_inf[n]
        return vec_output

    def predict_label(self, batch):

        output=self.forward(batch,unnormalization=False)
        # print(output)
        preds = output.argmax(1)
        return preds

    def get_features(self, batch, normalization=True):

        if normalization==True:
            batch = self.apply_normalization(batch)
        output = l2_normalize((self.model(batch)[0]))
        return output

    def init_model(self, attack_setting, clean_xs,  pair_imgs, ys, targeted):  # Batch argument.
        self.attack_mode = True
        self.targeted=targeted
        self.log_interval = attack_setting['log_interval']
        self.max_num_queries = attack_setting['max_num_queries']
        self.attack = attack_setting['attack']
        # self.detector.clear_memory()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        clean_batch = self.apply_normalization(torch.FloatTensor(pair_imgs).to(device))
        self.pair_img_features = l2_normalize(self.model(clean_batch)[0])
        self.original_images = torch.FloatTensor(clean_xs).to(device)
        self.ys = torch.LongTensor(ys).to(device)
        num_images = self.original_images.size(0)
        self.num_images=num_images
        self.num_queries = torch.zeros(num_images).long()
        if self.stateful_detection:
            self.detector = [Detector(K=50, threshold=0.002) for i in range(num_images)]
            self.log_k_avg_dist = torch.zeros(num_images,self.max_num_queries)

        self.log_l_2 = -torch.ones(num_images,self.max_num_queries)
        self.log_l_inf = -torch.ones(num_images,self.max_num_queries)
        self.log_is_adversarial = torch.zeros(num_images,self.max_num_queries).byte()
        self.log_dist = torch.zeros(num_images,self.max_num_queries)


        self.log_last_adv_imgs =torch.zeros(num_images,self.max_num_queries // self.log_interval,3,112,112).byte()
        self.log_best_l_2_adv_imgs = torch.zeros(num_images,self.max_num_queries // self.log_interval,3,112,112).byte()
        self.log_best_l_inf_adv_imgs = torch.zeros(num_images,self.max_num_queries // self.log_interval, 3, 112, 112).byte()
        self.log_best_l_2 = torch.zeros(num_images,self.max_num_queries // self.log_interval)
        self.log_best_l_inf = torch.zeros(num_images,self.max_num_queries // self.log_interval)
        self.best_l_inf = 1e3*torch.ones(num_images)
        self.best_l_2 =1e3*torch.ones(num_images)
        self.log_last_adv_img=torch.zeros(num_images,3,112,112).byte()
        self.log_best_l_2_adv_img=torch.zeros(num_images,3,112,112).byte()
        self.log_best_l_inf_adv_img=torch.zeros(num_images,3,112,112).byte()





    def get_num_queries(self,batch_idx=0):
        return self.num_queries[batch_idx]  # Return # of queries.


    def get_log(self):
        if self.stateful_detection:
            return (self.num_queries,
                    self.log_l_2, self.log_l_inf, self.log_is_adversarial, self.log_dist, self.log_best_l_2,
                    self.log_best_l_inf,
                    self.log_last_adv_imgs, self.log_best_l_2_adv_imgs, self.log_best_l_inf_adv_imgs, self.log_k_avg_dist)
        else:
            return (self.num_queries ,
        self.log_l_2, self.log_l_inf, self.log_is_adversarial, self.log_dist, self.log_best_l_2, self.log_best_l_inf,
        self.log_last_adv_imgs, self.log_best_l_2_adv_imgs, self.log_best_l_inf_adv_imgs)