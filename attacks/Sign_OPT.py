import time
import numpy as np 
from numpy import linalg as LA
import torch
import scipy.spatial
from scipy.linalg import qr
#from qpsolvers import solve_qp
import random

start_learning_rate = 1.0
MAX_ITER = 1000
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def quad_solver(Q, b):
    """
    Solve min_a  0.5*aQa + b^T a s.t. a>=0
    """
    K = Q.shape[0]
    alpha = np.zeros((K,))
    g = b
    Qdiag = np.diag(Q)
    for i in range(20000):
        delta = np.maximum(alpha - g/Qdiag,0) - alpha
        idx = np.argmax(abs(delta))
        val = delta[idx]
        if abs(val) < 1e-7: 
            break
        g = g + val*Q[:,idx]
        alpha[idx] += val
    return alpha

def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign==0] = 1
    return y_sign

class OPT_attack_sign_SGD(object):
    def __init__(self, model, k=200, train_dataset=None):
        self.model = model
        self.k = k
        self.train_dataset = train_dataset 
        # self.log = torch.ones(MAX_ITER,2)

    def get_log(self):
        return self.log
    
    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000, query_limit=10000,
                          distortion=None, seed=None, svm=False, momentum=0.0, stopping=0.0001):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        query_count = 0
        ls_total = 0
        # Calculate a good starting point.
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        # print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if model.predict_label(x0+torch.tensor(theta, dtype=torch.float).to(device))!=y0:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)


        if g_theta == float('inf'):
            num_directions = 100
            best_theta, g_theta = None, float('inf')
            print("Searching for the initial direction on %d random directions: " % (num_directions))
            timestart = time.time()
            for i in range(num_directions):
                query_count += 1
                theta = np.random.randn(*x0.shape)
                if model.predict_label(x0+torch.tensor(theta, dtype=torch.float).to(device))!=y0:
                    initial_lbd = LA.norm(theta)
                    theta /= initial_lbd
                    lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        print("--------> Found distortion %.4f" % g_theta)
        timeend = time.time()        
        if g_theta == float('inf'):    
            print("Couldn't find valid initial, failed")
            return x0 
        # print("==========> Found best distortion %.4f in %.4f seconds "
        #       "using %d queries" % (g_theta, timeend-timestart, query_count))
    
        # self.log[0][0], self.log[0][1] = g_theta, query_count
        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            if svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta)
            else:
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)

            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            min_vg = vg
            for _ in range(15):
                if momentum > 0:
#                     # Nesterov
#                     vg_prev = vg
#                     new_vg = momentum*vg - alpha*sign_gradient
#                     new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                    new_vg = momentum*vg - alpha*sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    if momentum > 0:
                        min_vg = new_vg
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    if momentum > 0:
#                         # Nesterov
#                         vg_prev = vg
#                         new_vg = momentum*vg - alpha*sign_gradient
#                         new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                        new_vg = momentum*vg - alpha*sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(
                        model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        if momentum > 0:
                            min_vg = new_vg
                        break
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            xg, gg = min_theta, min_g2
            vg = min_vg
            
            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)

            if model.get_num_queries() >= query_limit:
               break
            
            # if (i+1)%10==0:
            #     print("Iteration %3d distortion %.4f num_queries %d" % (i+1, gg, query_count))
            # self.log[i+1][0], self.log[i+1][1] = gg, query_count
            #if distortion is not None and gg < distortion:
            #    print("Success: required distortion reached")
            #    break

#             if gg > prev_obj-stopping:
#                 print("Success: stopping threshold reached")
#                 break            
#             prev_obj = gg
#         target = model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float).cuda())
        # timeend = time.time()
        # print("\nAdversarial Example Found Successfully: distortion %.4f target"
        #       " %d queries %d \nTime: %.4f seconds" % (gg, target, query_count, timeend-timestart))
        
        # self.log[i+1:,0] = gg
        # self.log[i+1:,1] = query_count
        #print(self.log)
        #print("Distortions: ", distortions)
        return x0 + torch.tensor(gg*xg, dtype=torch.float).to(device)

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = np.zeros(theta.shape)
        queries = 0
        ### USe orthogonal transform
        #dim = np.prod(sign_grad.shape)
        #H = np.random.randn(dim, K)
        #Q, R = qr(H, mode='economic')
        preds = []
        h=0.5
        for iii in range(K):

            u = np.random.randn(*theta.shape)
            #u = Q[:,iii].reshape(sign_grad.shape)
            u /= LA.norm(u)
            sign = 1
            new_theta = theta + h*u
            new_theta /= LA.norm(new_theta)
            pred=self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).to(device))
            # Targeted case.
            if (target is not None and 
                pred== target):
                sign = -1
            # Untargeted case
            preds.append(pred.item())
            if (target is None and
                pred != y0):
                sign = -1
            # print(sign)
            queries += 1
            sign_grad += u*sign
        
        sign_grad /= K
        # print(sign_grad)
#         sign_grad_u = sign_grad/LA.norm(sign_grad)
#         new_theta = theta + h*sign_grad_u
#         new_theta /= LA.norm(new_theta)
#         fxph, q1 = self.fine_grained_binary_search_local(self.model, x0, y0, new_theta, initial_lbd=initial_lbd, tol=h/500)
#         delta = (fxph - initial_lbd)/h
#         queries += q1
#         sign_grad *= 0.5*delta       
        
        return sign_grad, queries
    
    def sign_grad_v2(self, x0, y0, theta, initial_lbd, h=0.001, K=200):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        for _ in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)
            
            ss = -1
            new_theta = theta + h*u
            new_theta /= LA.norm(new_theta)
            if self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).to(device)) == y0:
                ss = 1
            queries += 1
            sign_grad += sign(u)*ss
        sign_grad /= K
        return sign_grad, queries


    def sign_grad_svm(self, x0, y0, theta, initial_lbd, h=0.001, K=100, lr=5.0, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        dim = np.prod(theta.shape)
        X = np.zeros((dim, K))
        for iii in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)
            
            sign = 1
            new_theta = theta + h*u
            new_theta /= LA.norm(new_theta)            
            
            # Targeted case.
            if (target is not None and 
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).to(device)) == target):
                sign = -1
                
            # Untargeted case
            if (target is None and
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).to(device)) != y0):
                sign = -1
                #if self.model.predict_label(x0+torch.tensor((initial_lbd*1.00001)*new_theta, dtype=torch.float).cuda()) == y0:
                #    print "Yes"
                #else:
                #    print "No"
                
            queries += 1
            X[:,iii] = sign*u.reshape((dim,))
        
        Q = X.transpose().dot(X)
        q = -1*np.ones((K,))
        G = np.diag(-1*np.ones((K,)))
        h = np.zeros((K,))
        ### Use quad_qp solver 
        #alpha = solve_qp(Q, q, G, h)
        ### Use coordinate descent solver written by myself, avoid non-positive definite cases
        alpha = quad_solver(Q, q)
        sign_grad = (X.dot(alpha)).reshape(theta.shape)
        
        return sign_grad, queries



    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        if model.predict_label(x0+torch.tensor(lbd*theta, dtype=torch.float).to(device)) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_hi*theta, dtype=torch.float).to(device)) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_lo*theta, dtype=torch.float).to(device)) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
                if self.model.get_num_queries()>=self.max_queries:
                    break

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).to(device)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if self.model.get_num_queries() >= self.max_queries:
                break

        lbd_hi=lbd_hi+1
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0+ torch.cuda.FloatTensor(current_best*theta).to(device)) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0
        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            # print( model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()))
            # print( (x0 + torch.tensor(lbd_mid*theta, dtype=torch.float.cuda()).max())
            # print( model.predict_label(x0 + torch.cuda.FloatTensor(lbd_mid*theta)) != y0,lbd_mid)
            if model.predict_label(x0 + torch.FloatTensor(lbd_mid*theta).to(device)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if self.model.get_num_queries() >= self.max_queries:
                break

        lbd_hi=lbd_hi+1
        return lbd_hi, nquery

    def eval_grad(self, model, x0, y0, theta, initial_lbd, tol=1e-5,  h=0.001, sign=False):
        # print("Finding gradient")
        fx = initial_lbd # evaluate function value at original point
        grad = np.zeros_like(theta)
        x = theta
        # iterate over all indexes in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
        queries = 0
        while not it.finished:

            # evaluate function at x+h
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h # increment by h
            unit_x = x / LA.norm(x)
            if sign:
                if model.predict_label(x0+torch.tensor(initial_lbd*unit_x, dtype=torch.float).to(device)) == y0:
                    g = 1
                else:
                    g = -1
                q1 = 1
            else:
                fxph, q1 = self.fine_grained_binary_search_local(model, x0, y0, unit_x, initial_lbd = initial_lbd, tol=h/500)
                g = (fxph - fx) / (h)
            
            queries += q1
            # x[ix] = oldval - h
            # fxmh, q2 = self.fine_grained_binary_search_local(model, x0, y0, x, initial_lbd = initial_lbd, tol=h/500)
            x[ix] = oldval # restore

            # compute the partial derivative with centered formula
            grad[ix] = g
            it.iternext() # step to next dimension

        # print("Found gradient")
        return grad, queries
    # image는 target_x0로 가길 원함
    # target clas image는 x0
    def attack_targeted(self, x0, y0,x_t=None , alpha = 0.2, beta = 0.001, iterations = 5000, query_limit=40000,
                        distortion=None, seed=None, svm=False, stopping=0.0001):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        print("Targeted attack - Source: {0} and Target: {1}".format(y0, y0.item()))
        
        if (model.predict_label(x0) == y0):

            print("Image already target. No need to attack.")
            return x0, 0.0

        
        if seed is not None:
            np.random.seed(seed)

        num_samples = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        ls_total = 0
        sample_count = 0
        print("Searching for the initial direction on %d samples: " % (num_samples))
        timestart = time.time()

#         samples = set(random.sample(range(len(self.train_dataset)), num_samples))
#         print(samples)
#         train_dataset = self.train_dataset[samples]

        # Iterate through training dataset. Find best initial point for gradient descent.
        xi=x_t
        yi_pred = model.predict_label(xi.to(device))
        query_count += 1
        if yi_pred != y0:
            print("Target image is not recognized as target.")
            return x0, 0.0

        theta = xi.cpu().numpy() - x0.cpu().numpy()
        initial_lbd = LA.norm(theta)
        theta /= initial_lbd
        lbd, count = self.fine_grained_binary_search_targeted(model, x0, y0, y0, theta, initial_lbd, g_theta)
        query_count += count
        if lbd < g_theta:
            best_theta, g_theta = theta, lbd
            print("--------> Found distortion %.4f" % g_theta)


#         xi = initial_xi
#         xi = xi.numpy()
#         theta = xi - x0
#         initial_lbd = LA.norm(theta.flatten(),np.inf)
#         theta /= initial_lbd     # might have problem on the defination of direction
#         lbd, count, lbd_g2 = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, theta)
#         query_count += count
#         if lbd < g_theta:
#             best_theta, g_theta = theta, lbd
#             print("--------> Found distortion %.4f" % g_theta)        

        timeend = time.time()
        if g_theta == np.inf:
            return x0, float('inf')
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" %
              (g_theta, timeend-timestart, query_count))

        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            if svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta, target=y0)
            else:
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta, target=y0)

            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local_targeted(
                    model, x0, y0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local_targeted(
                        model, x0, y0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break
                        
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            xg, gg = min_theta, min_g2
            
            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)

            if self.model.get_num_queries() >= self.max_queries:
                break
            
            # if i%5==0:
            #     print("Iteration %3d distortion %.4f num_queries %d" % (i+1, gg, query_count))
#                 print("Iteration: ", i, " Distortion: ", gg, " Queries: ", query_count,
#                       " LR: ", alpha, "grad_queries", grad_queries, "ls_queries", ls_count)
            
            #if distortion is not None and gg < distortion:
            #    print("Success: required distortion reached")
            #    break

#             if gg > prev_obj-stopping:
#                 print("Success: stopping threshold reached")
#                 break 
#             prev_obj = gg

        # adv_target = model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float).cuda())
        # if (adv_target == y0):
        #     timeend = time.time()
        #     print("\nAdversarial Example Found Successfully: distortion %.4f target"
        #           " %d queries %d LS queries %d \nTime: %.4f seconds" % (gg, y0, query_count, ls_total, timeend-timestart))

        return x0 + torch.tensor(gg*xg, dtype=torch.float).to(device)
        # else:
        #     print("Failed to find targeted adversarial example.")
        #     return x0
    
    def fine_grained_binary_search_local_targeted(self, model, x0, y0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if model.predict_label(x0 + torch.tensor(lbd*theta, dtype=torch.float).to(device)) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_hi*theta, dtype=torch.float).to(device)) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100: 
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_lo*theta, dtype=torch.float).to(device)) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1
                if self.model.get_num_queries()>=self.max_queries:
                    break

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).to(device)) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if self.model.get_num_queries() >= self.max_queries:
                break

#         temp_theta = np.abs(lbd_hi*theta)
#         temp_theta = np.clip(temp_theta - 0.15, 0.0, None)
#         loss = np.sum(np.square(temp_theta))
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, model, x0, y0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0 + torch.tensor(current_best*theta, dtype=torch.float).to(device)) != t:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).to(device)) != t:
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
            if self.model.get_num_queries() >= self.max_queries:
                break
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target, target_img =None, distortion=None, seed=None,
                 svm=False, query_limit=20000, momentum=0., stopping=0.0001, TARGETED=False, epsilon=None):
        self.max_queries=query_limit
        if TARGETED==True:
            adv = self.attack_targeted(target_img, label_or_target,input_xi , distortion=distortion,
                                       seed=seed, svm=svm, query_limit=query_limit, stopping=stopping)
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, distortion=distortion,
                                         seed=seed, svm=svm, query_limit=query_limit, momentum=momentum,
                                         stopping=stopping)
        return adv   
    
        
