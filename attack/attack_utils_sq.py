import torch
import collections
import time
import numpy as np
from attack.group_generator import EquallySplitGrouping
def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p
# binary search, obtain the maximum step length of the gradient upper bound
def fine_grained_binary_search(vid_model, theta, initial_lbd,image_ori,targeted):
    lbd = initial_lbd
    while vid_model((image_ori + lbd * theta)[None,:])[1] != targeted:
        lbd *= 1.05
        if lbd > 1:
            return False, lbd
    num_intervals = 100
    lambdas = np.linspace(0.0, lbd.cpu(), num_intervals)[1:]
    lbd_hi = lbd
    lbd_hi_index = 0
    for i, lbd in enumerate(lambdas):
        if vid_model((image_ori + lbd * theta)[None,:])[1] == targeted:
            lbd_hi = lbd
            lbd_hi_index = i
            break
    lbd_lo = lambdas[lbd_hi_index - 1]
    while (lbd_hi - lbd_lo) > 1e-7:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        if vid_model((image_ori + lbd_mid * theta)[None,:])[1] == targeted:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return True,lbd_hi


# Initialize the adversarial example (Targeted attacks)
def initialize_from_train_dataset_baseline(vid_model,image_ori,image_adv,targeted,MASK):
    theta = (image_adv - image_ori) * MASK      # the initial perturbations
    initial_lbd = torch.norm(theta)
    theta = theta / initial_lbd                 # The initial attack direction
    I,lbd = fine_grained_binary_search(vid_model, theta, initial_lbd,image_ori,targeted)
    # update theta and g_theta, where theta represents the attack direction and g_theta represents the step size
    theta, g2 = theta, lbd
    new_image=image_ori + theta *lbd
    new_image=torch.clamp(new_image, 0., 1.)    # clip operation
    return I,new_image


# get the average Loss and estimated gradient
def sim_rectification_vector(model, vid, tentative_directions, n, sigma, target_class, rank_transform, sub_num,
                             group_gen, untargeted):
    '''
    N is used to estimate the sample number for NES
    sub_num is used to estimate the gradient to prevent insufficient GPU resources if n is too large
    '''
    with torch.no_grad():
        grads = torch.zeros(len(group_gen), device='cuda')   # zeros gradient
        count_in = 0                                         # record the number of hits in each NES
        loss_total = 0                                       # loss per NES
        batch_loss = []                                      # loss per batch
        batch_noise = []                                     # noise per batch
        batch_idx = []                                       # top_idx per batch
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))                      # sub_num samples
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma   # Random noise
            all_noise = torch.cat([noise_list, -noise_list], 0)                               # random noise for nes
            perturbation_sample = group_gen.apply_group_change(tentative_directions, all_noise)    # group noise
            adv_vid_rs += perturbation_sample                                                 # add noise
            del perturbation_sample                                                           # delete noise
            top_val, top_idx, logits = model(adv_vid_rs)                                      # classify
            if untargeted:
                loss = -torch.max(logits, 1)[0]      # Loss function (reduces class confidence of maximum probability)
            else:
                loss = torch.nn.functional.cross_entropy(logits,torch.tensor(target_class, dtype=torch.long,
                                                                             device='cuda').repeat(sub_num), reduction='none')
            batch_loss.append(loss)
            batch_idx.append(top_idx)
            batch_noise.append(all_noise)
        # batch merge
        batch_noise = torch.cat(batch_noise, 0)
        batch_idx = torch.cat(batch_idx)
        batch_loss = torch.cat(batch_loss)
        # sorted loss
        if rank_transform:
            good_idx = torch.sum(batch_idx == target_class, 1).byte()                               # unsuccessful attack id
            changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))    # increase penalties
            loss_order = torch.zeros(changed_loss.size(0), device='cuda')                           #
            sort_index = changed_loss.sort()[1]                                                     # get sorted idx
            loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)  # balance coefficient
            available_number = torch.sum(good_idx).item()                                           # available numbers
            count_in += available_number
            unavailable_number = n - available_number                                               # unavailable numbers
            unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                       loss_order)) / unavailable_number if unavailable_number else torch.tensor(
                0., device='cuda')            # unavailable weights
            rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)           # rank weights
            grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),0)
        else:
            idxs = (batch_idx == target_class).nonzero()   # unsuccessful attack id
            valid_idxs = idxs[:, 0]                        # idx
            valid_loss = torch.index_select(batch_loss, 0, valid_idxs)     # available loss
            loss_total += torch.mean(valid_loss).item()    # average loss
            count_in += valid_loss.size(0)                 # available number
            noise_select = torch.index_select(batch_noise, 0, valid_idxs)  # available noise
            grads += torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),0)
        if count_in == 0:
            return None, None
        return loss_total / count_in, grads


# The input video should be tensor, the input size is [num_frames, c,w,h], and the frames should be normalized to [0,1]
def untargeted_video_attack(vid_model, vid, directions_generator, ori_class, rank_transform=False, eps=0.05,
                            max_lr=1e-2,min_lr=1e-3, sample_per_draw=24,max_iter=30000, sigma=1e-3, sub_num_sample=12,
                            image_split=8,key_list=[]):
    num_iter = 0                                                                       # iteration numbers
    perturbation = (torch.rand_like(vid) * 2 - 1) * eps                                # initial perturbations
    MASK=torch.zeros(vid.size())                                                       #  frame mask
    MASK[key_list, :, :, :] = 1
    perturbation = perturbation*(MASK.cuda())
    adv_vid = torch.clamp(vid.clone() + perturbation, 0., 1.)                          # initial adversarial video
    cur_lr = max_lr                                                                    # max learning rate
    last_p = []                                                                        # current learning rate
    last_score = []                                                                    # predicted probability
    group_gen = EquallySplitGrouping(image_split)                                      # predicted probability
    while num_iter < max_iter:
        top_val, top_idx, _ = vid_model(adv_vid[None, :])                              # The predicted results
        num_iter += 1                                                                  # iteration +1
        if ori_class != top_idx[0][0]:
            # the predict category is inconsistent with the original category, the attack succeds
            #print('early stop at iterartion {}'.format(num_iter))
            return True, num_iter, adv_vid
        idx = (top_idx == ori_class).nonzero()                                         # the index of the original label
        pre_score = top_val[0][idx[0][1]]                                              # get the original probability
        del top_val
        del top_idx
        #print('cur target prediction: {}'.format(pre_score))
        last_score.append(float(pre_score))                                            # record probability
        last_score = last_score[-200:]                                                 # only retain last 200 results
        if last_score[-1] >= last_score[0] and len(last_score) == 200:
            # if the last predicted probability value is greater than the first probability
            #print('FAIL: No Descent, Stop iteration')                                 # attack fails
            return False, pre_score.cpu().item(), adv_vid
        last_p.append(float(pre_score))                      # Record the predicted probability values
        last_p = last_p[-20:]                                # Take the latest 20 probability values
        if last_p[-1] <= last_p[0] and len(last_p) == 20:
            # updating the learning rate
            if cur_lr > min_lr:
                #print("[log] Annealing max_lr")
                cur_lr = max(cur_lr / 2., min_lr)
            last_p = []
        tentative_directions = directions_generator(adv_vid).cuda()   # generate the perturbation direction
        group_gen.initialize(tentative_directions)                    # Initialize the block generator
        # estimate gradient
        l, g = sim_rectification_vector(vid_model, adv_vid, tentative_directions, sample_per_draw, sigma,
                                        ori_class, rank_transform, sub_num_sample, group_gen, untargeted=True)
        if l is None and g is None:
            #print('nes sim fails, try again....')
            continue
        assert g.size(0) == len(group_gen), 'rectification vector size error!'
        rectified_directions = group_gen.apply_group_change(tentative_directions, torch.sign(g))  # gradient correction
        del tentative_directions           # free the space
        num_iter += sample_per_draw        # iteration accumulation
        proposed_adv_vid = adv_vid
        assert proposed_adv_vid.size() == rectified_directions.size(), 'rectification error!'
        # add perturbations
        proposed_adv_vid += cur_lr * rectified_directions*(MASK.cuda())   # update adversary
        # noise clipping
        bottom_bounded_adv = torch.where((vid - eps) > proposed_adv_vid, vid - eps,
                                         proposed_adv_vid)
        bounded_adv = torch.where((vid + eps) < bottom_bounded_adv, vid + eps, bottom_bounded_adv)
        clip_frame = torch.clamp(bounded_adv, 0., 1.)
        adv_vid = clip_frame.clone()
        #print('step {} : loss {} | lr {}'.format(num_iter, l, cur_lr))
    return False, pre_score.cpu().item(), adv_vid

# The input video should be tensor, the input size is [num_frames, c,w,h], and the frames should be normalized to [0,1]
def untargeted_video_square_attack(vid_model, vid, directions_generator, ori_class, rank_transform=False, eps=0.05,
                            max_lr=1e-2,min_lr=1e-3, sample_per_draw=24,max_iter=30000, sigma=1e-3, sub_num_sample=12,
                            image_split=8,key_list=[]):
    num_iter = 0                                                                       # iteration numbers
    perturbation = (torch.rand_like(vid) * 2 - 1) * eps                                # initial perturbations
    MASK=torch.zeros(vid.size())                                                       #  frame mask
    MASK[key_list, :, :, :] = 1
    perturbation = perturbation*(MASK.cuda())
    adv_vid = torch.clamp(vid.clone() + perturbation, 0., 1.)                          # initial adversarial video
    cur_lr = max_lr                                                                    # max learning rate
    last_p = []                                                                        # current learning rate
    last_score = []                                                                    # predicted probability
    group_gen = EquallySplitGrouping(image_split)                                      # predicted probability
    while num_iter < max_iter:
        top_val, top_idx, _ = vid_model(adv_vid[None, :])                              # The predicted results
        num_iter += 1                                                                  # iteration +1
        if ori_class != top_idx[0][0]:
            # the predict category is inconsistent with the original category, the attack succeds
            #print('early stop at iterartion {}'.format(num_iter))
            return True, num_iter, adv_vid
        idx = (top_idx == ori_class).nonzero()                                         # the index of the original label
        pre_score = top_val[0][idx[0][1]]                                              # get the original probability
        del top_val
        del top_idx
        #print('cur target prediction: {}'.format(pre_score))
        last_score.append(float(pre_score))                                            # record probability
        last_score = last_score[-200:]                                                 # only retain last 200 results
        if last_score[-1] >= last_score[0] and len(last_score) == 200:
            # if the last predicted probability value is greater than the first probability
            #print('FAIL: No Descent, Stop iteration')                                 # attack fails
            return False, pre_score.cpu().item(), adv_vid
        last_p.append(float(pre_score))                      # Record the predicted probability values
        last_p = last_p[-20:]                                # Take the latest 20 probability values
        if last_p[-1] <= last_p[0] and len(last_p) == 20:
            # updating the learning rate
            if cur_lr > min_lr:
                #print("[log] Annealing max_lr")
                cur_lr = max(cur_lr / 2., min_lr)
            last_p = []
        tentative_directions = directions_generator(adv_vid).cuda()   # generate the perturbation direction
        group_gen.initialize(tentative_directions)                    # Initialize the block generator
        # estimate gradient
        l, g = sim_rectification_vector(vid_model, adv_vid, tentative_directions, sample_per_draw, sigma,
                                        ori_class, rank_transform, sub_num_sample, group_gen, untargeted=True)
        if l is None and g is None:
            #print('nes sim fails, try again....')
            continue
        assert g.size(0) == len(group_gen), 'rectification vector size error!'
        rectified_directions = group_gen.apply_group_change(tentative_directions, torch.sign(g))  # gradient correction
        del tentative_directions           # free the space
        num_iter += sample_per_draw        # iteration accumulation
        proposed_adv_vid = adv_vid
        assert proposed_adv_vid.size() == rectified_directions.size(), 'rectification error!'
        # add perturbations
        proposed_adv_vid += cur_lr * rectified_directions*(MASK.cuda())   # update adversary
        # noise clipping
        bottom_bounded_adv = torch.where((vid - eps) > proposed_adv_vid, vid - eps,
                                         proposed_adv_vid)
        bounded_adv = torch.where((vid + eps) < bottom_bounded_adv, vid + eps, bottom_bounded_adv)
        clip_frame = torch.clamp(bounded_adv, 0., 1.)
        adv_vid = clip_frame.clone()
        #print('step {} : loss {} | lr {}'.format(num_iter, l, cur_lr))
    return False, pre_score.cpu().item(), adv_vid

def square_attack_linf(vid_model, x, y, corr_classified, eps, n_iters, p_init, metrics_path, targeted, loss_type, key_list, class_no):
    """ The Linf square attack """
    _, _, logits = vid_model(x[None, :])    
    MASK=torch.zeros(x.size())     
    MASK[key_list, :, :, :] = 1    
    x = x.cpu().numpy()
    np.random.seed(0)  # important to leave it here as well
    min_val, max_val = 0, 1 if x.max() <= 1 else 255
    c, w, h = x.shape[1:]
    n_features = c*h*w
    target_list = key_list.squeeze().tolist()
    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)
    preds_correct_class = (logits * torch.nn.functional.one_hot(torch.tensor([y]).cuda(), num_classes=class_no))
    diff = preds_correct_class - logits  # difference between the correct class and all other classes
    diff[0][y] = np.inf  # to exclude zeros coming from f_correct - f_correct
    margin, _ = diff[0].min(0)
    loss_min = margin   
    n_queries = 1  # ones because we have already used 1 query
    last_loss = []
    for i_iter in range(n_iters - 1):
        vid = torch.from_numpy(x).float().cuda()
        vid_best = torch.from_numpy(x_best).float().cuda()
        adv_vid = vid * (1-MASK.cuda()) + vid_best * (MASK.cuda())
        top_val, top_idx, _ = vid_model(adv_vid[None, :])        
        if y == top_idx[0][0]:
            # the predict category is inconsistent with the original category, the attack succeds
            #print('early stop at iterartion {}'.format(num_iter))
            return True, n_queries, adv_vid
        idx = (top_idx == y).nonzero()                                         # the index of the original label
        x_curr, x_best_curr = x, x_best
        loss_min_curr = loss_min
        deltas = x_best_curr - x_curr
        last_loss.append(float(loss_min_curr))
        last_loss = last_loss[-200:]
        if last_loss[-1] == last_loss[0] and len(last_loss) == 200:
            # print(1)
            return False, 0, adv_vid         

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in target_list:
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)
        vid = torch.from_numpy(x).float().cuda()
        vid_best = torch.from_numpy(x_new).float().cuda()
        adv_vid = vid * (1-MASK.cuda()) + vid_best * (MASK.cuda())
        top_val, top_idx, logits = vid_model(adv_vid[None, :]) 
        # logits = model.predict(x_new)
        preds_correct_class = (logits * torch.nn.functional.one_hot(torch.tensor([y]).cuda(), num_classes=class_no))
        diff = preds_correct_class - logits  # difference between the correct class and all other classes
        diff[0][y] = np.inf  # to exclude zeros coming from f_correct - f_correct
        margin, _ = diff.min(1, keepdims=True)
        loss = margin 

        if loss < loss_min_curr:
            loss_min = loss
            x_best = x_new
        n_queries += 1

    return False, 0, adv_vid

# The input video should be tensor, the input size is [num_frames, c,w,h], and the frames should be normalized to [0,1]
def targeted_video_attack(vid_model, vid, target_vid, target_class, directions_generator,rank_transform=False,
                          starting_eps=1., eps=0.05, delta_eps=0.5, max_lr=1e-2, min_lr=1e-3, sample_per_draw=24,
                          max_iter=60000, sigma=1e-6, sub_num_sample=12, image_split=8,key_list=[]):
    MASK = torch.zeros(vid.size())
    MASK[key_list, :, :, :] = 1                                                  # frame mask
    MASK=MASK.cuda()
    num_iter = 0                                                                 # iteration numbers
    # initial adversarial video
    Flag,adv_vid = initialize_from_train_dataset_baseline(vid_model,vid,target_vid,target_class,MASK)
    cur_eps = starting_eps                                                       # primal epsilon
    if (Flag==False):
        return False, cur_eps, adv_vid
    explore_succ = collections.deque(maxlen=5)    # accumulate five results for learning rate judgement
    reduce_eps_fail = 0                           # failure numbers, used for adjusting delta_epsilon
    cur_min_lr = min_lr                           # min learning rate
    cur_max_lr = max_lr                           # max learning rate
    delta_eps_schedule = [0.01, 0.003, 0.001, 0]  # delta_epsilon adjustment policy
    update_steps = [1, 10, 100, 100]              # iteration stamp, adjust delta_epison according to failure numbers
    update_weight = [2, 1.5, 1.5, 1.5]            # delta_epsilon reduce coefficient
    cur_eps_period = 0                            # the current epsilon period
    group_gen = EquallySplitGrouping(image_split)              # block operation
    while num_iter < max_iter:
        top_val, top_idx, _ = vid_model(adv_vid[None,:])       # The predicted results
        num_iter += 1                                          # iteration +1
        tentative_directions = directions_generator(adv_vid).cuda()   # generate the perturbation direction
        group_gen.initialize(tentative_directions)             # Initialize the block generator
        # generate loss and gradient
        l, g = sim_rectification_vector(vid_model, adv_vid, tentative_directions, sample_per_draw, sigma,
                                        target_class, rank_transform, sub_num_sample, group_gen, untargeted=False)
        if l is None and g is None:
            #print('nes sim fails, try again....')
            continue
        assert g.size(0) == len(group_gen), 'rectification vector size error!'
        rectified_directions = group_gen.apply_group_change(tentative_directions, torch.sign(g))  # gradient correction
        if target_class == top_idx[0][0] and cur_eps <= eps:
            # early stop
            #print('early stop at iterartion {}'.format(num_iter))
            return True, num_iter, adv_vid
        idx = (top_idx == target_class).nonzero()   # get index of the targeted category
        pre_score = top_val[0][idx[0][1]]           # get the probability at the attack category
        #print('cur target prediction: {}'.format(pre_score))
        #print('cur eps: {}'.format(cur_eps))
        num_iter += sample_per_draw                                                      # accumulate query numbers
        cur_lr = cur_max_lr                                                              # The current learning rate
        prop_de = delta_eps                                                              # epsilon decay coefficient
        # PGD iteration
        while True:
            num_iter += 1                                                                # iterations +1
            proposed_adv_vid = adv_vid.clone()                                           # initial adversarial video
            assert proposed_adv_vid.size() == rectified_directions.size(), 'rectification error!'
            # PGD attack
            proposed_adv_vid -= cur_lr * rectified_directions * MASK          # update adversarial video（sparse attack）
            proposed_eps = max(cur_eps - prop_de, eps)                        # update epsilon
            # project and clipping
            bottom_bounded_adv = torch.where((vid - proposed_eps) > proposed_adv_vid, vid - proposed_eps,
                                             proposed_adv_vid)
            bounded_adv = torch.where((vid + proposed_eps) < bottom_bounded_adv, vid + proposed_eps, bottom_bounded_adv)
            clip_frame = torch.clamp(bounded_adv, 0., 1.)   # clip
            proposed_adv_vid = clip_frame.clone()
            top_val, top_idx, _ = vid_model(proposed_adv_vid[None,:])       # predicted results
            if target_class in top_idx[0]:
                #print('update with delta eps: {}'.format(prop_de))    # update delta_epsilon
                if prop_de > 0:
                    cur_max_lr = max_lr     # max learning rate
                    cur_min_lr = min_lr     # min learning rate
                    explore_succ.clear()    # clearing The stack of successive success results
                    reduce_eps_fail = 0     # The number of failure
                else:
                    explore_succ.append(True)           # accumulative
                    reduce_eps_fail += 1                # failure number +1
                adv_vid = proposed_adv_vid.clone()
                cur_eps = max(cur_eps - prop_de, eps)   # update epsilon
                break
            # learning rate adjustment
            elif cur_lr >= cur_min_lr * 2:
                cur_lr = cur_lr / 2
            else:
                # attack fail
                if prop_de == 0:
                    explore_succ.append(False)
                    reduce_eps_fail += 1
                    #print('Trying to eval grad again.....')
                    break
                prop_de = 0
                cur_lr = cur_max_lr
        # failure numbers exceed the threshold, adjust delta eps
        if reduce_eps_fail >= update_steps[cur_eps_period]:
            # update delta_epsilon
            delta_eps = max(delta_eps / update_weight[cur_eps_period], delta_eps_schedule[cur_eps_period])
            #print('Success rate of reducing eps is too low. Decrease delta eps to {}'.format(delta_eps))
            if delta_eps <= delta_eps_schedule[cur_eps_period]:
                cur_eps_period += 1    # The current period +1
            if delta_eps < 1e-5:
                # attack fails
                #print('fail to converge at query number {} with eps {}'.format(num_iter, cur_eps))
                return False, cur_eps, adv_vid
            reduce_eps_fail = 0

        # adjust min learning rate and max learning rate
        if len(explore_succ) == explore_succ.maxlen and cur_min_lr > 1e-7:
            succ_p = np.mean(explore_succ)
            if succ_p < 0.5:
                cur_min_lr /= 2
                cur_max_lr /= 2
                explore_succ.clear()
                #print('explore succ rate too low. increase lr scope [{}, {}]'.format(cur_min_lr, cur_max_lr))
        #print('step {} : loss {} | lr {}'.format(num_iter, l, cur_lr))
    return False, cur_eps, adv_vid


def square_attack_linf_targeted(vid_model, x, y, eps, n_iters, p_init, key_list, class_no):
    # y is target class label-onehot
    """ The Linf square attack """
    target_label = 1
    _, _, logits = vid_model(x[None, :])
    MASK=torch.zeros(x.size())
    MASK[key_list, :, :, :] = 1
    x = x.cpu().numpy()
    np.random.seed(0)  # important to leave it here as well
    min_val, max_val = 0, 1 if x.max() <= 1 else 255
    c, w, h = x.shape[1:]
    n_features = c*h*w
    target_list = key_list.squeeze().tolist()
    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)
    # margin_loss
    preds_correct_class = (logits * torch.nn.functional.one_hot(torch.tensor([y]).cuda(), num_classes=class_no))
    diff = preds_correct_class - logits  # difference between the correct class and all other classes
    diff[0][y] = np.inf  # to exclude zeros coming from f_correct - f_correct
    margin = diff[0][target_label]
    loss_min = margin * -1 # if targeted,loss_min = - margin and compute corss_entropy_loss
    # corss_entropy loss
    probs = torch.nn.functional.softmax(logits, 1)
    probs = probs.cpu().numpy()
    corss_entropy_loss = -np.log(probs[0][y])

    n_queries = 1  # ones because we have already used 1 query
    # deltas_ = []
    last_loss = []
    time_start = time.time()
    for i_iter in range(n_iters - 1):
        # print(loss_min)
        vid = torch.from_numpy(x).float().cuda()
        vid_best = torch.from_numpy(x_best).float().cuda()
        adv_vid = vid * (1-MASK.cuda()) + vid_best * (MASK.cuda())
        top_val, top_idx, _ = vid_model(adv_vid[None, :])
        if target_label == top_idx[0][0]:
            # the predict category is inconsistent with the original category, the attack succeds
            return True, n_queries, adv_vid
        idx = (top_idx == target_label).nonzero()                                         # the index of the original label
        # pre_score = top_val[0][idx[0][1]]
        x_curr, x_best_curr = x, x_best
        loss_min_curr = loss_min
        corss_entropy_loss_curr = corss_entropy_loss
        if corss_entropy_loss_curr < corss_entropy_loss:
            corss_entropy_loss = corss_entropy_loss_curr # get min corss_entropy_loss
        deltas = x_best_curr - x_curr
        last_loss.append(float(loss_min_curr))
        last_loss = last_loss[-200:]
        if last_loss[-1] == last_loss[0] and len(last_loss) == 200:
            # print(1)
            return False, 0, adv_vid

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in target_list:
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)
        vid = torch.from_numpy(x).float().cuda()
        vid_best = torch.from_numpy(x_new).float().cuda()
        adv_vid = vid * (1-MASK.cuda()) + vid_best * (MASK.cuda())
        top_val, top_idx, logits = vid_model(adv_vid[None, :])
        preds_correct_class = (logits * torch.nn.functional.one_hot(torch.tensor([y]).cuda(), num_classes=class_no))
        diff = preds_correct_class - logits  # difference between the correct class and all other classes
        diff[0][y] = np.inf  # to exclude zeros coming from f_correct - f_correct
        margin = diff[0][target_label]
        loss = margin * -1

        if loss < loss_min_curr:
            loss_min = loss
            x_best = x_new
        n_queries += 1


    return False, 0, adv_vid
