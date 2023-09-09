import os
import torch
import argparse
import numpy as np
from Utils.utils import *
from attack.untargetedAttack import attack
from model_wrapper.vid_model_top_k import *  # acquire top k results
from Coreset_selection.coreset_bak import CoreSetMIPSampling 

config =argparse.ArgumentParser()
config.add_argument('--model_name',type=str,default='i3d',
                    help='The action recognition')
config.add_argument('--dataset_name',type=str,default='hmdb51',
                    help='The dataset: hmdb51/ucf101')
config.add_argument('--gpus',nargs='+',type=int,required=True,
                    help='The gpus to use')
config.add_argument('--test_num',type=int,default=50,
                    help='The number of testing')
config.add_argument('initial_size', type=int, help="initial sample size for active learning")
args = config.parse_args()
gpus = args.gpus                # gpu setting
image_models = ['resnet50']
os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])    # visible gpu setting
model_name = args.model_name
dataset_name = args.dataset_name
test_num = args.test_num
initial_size = args.initial_size
print('load {} dataset'.format(dataset_name))
test_data = generate_dataset(model_name, dataset_name)  # dataset setting
print('load {} model'.format(model_name))
model = generate_model(model_name, dataset_name)        # get the training dataset
print('Initialize model')
try:
    model.cuda()
except:
    pass
if model_name == 'c3d':
    vid_model = C3D_K_Model(model)
elif model_name == 'lrcn':
    vid_model = LRCN_K_Model(model)
elif model_name == 'i3d':
    vid_model = I3D_K_Model(model)
elif model_name == 'rgb_r2plus1d':
    vid_model = RGB_R2PLUS1D_K_Model(model)
elif model_name == 'rgb_resnext3d':
    vid_model = RGB_RESNEXT3D_K_Model(model)
seq_len = 64
if model_name == 'c3d':
    image_size = 112
elif model_name == 'rgb_r2plus1d':
    image_size = 112
elif model_name == 'rgb_resnext3d':
    image_size = 112
else:
    image_size = 224
# get attack ids
attacked_ids = get_samples(model_name, dataset_name)
def GetPairs(test_data,idx):
    x0, label = test_data[attacked_ids[idx]]
    if model_name == 'rgb_r2plus1d' or model_name == 'rgb_resnext3d':
            x0 = x0.view(-1, 3, 112, 112).transpose(0, 1)
            label = [0, label]
    x0 = image_to_vector(model_name, x0)
    return x0.cuda(),label[1]
number = 40
result_root = 'CLVA/results/{}_{}'.format(model_name,dataset_name)
av_metric = os.path.join(result_root, 'Avmetric.txt')
success_num = 0        # Number of successful attacks
total_P_num = 0        # perturbation sum for all test samples
total_query_num = 0   # total query numbers

for i in range(0, test_num):
    output_path = os.path.join(result_root, 'vid-{}'.format(attacked_ids[i]))
    if os.path.exists(output_path) is not True:
        os.makedirs(output_path)
    vid,vid_label = GetPairs(test_data,i)
    representation = []
    adv_vids = [] 
    for i in range(seq_len):
        masklist = np.zeros((seq_len))
        masklist[i] = 1
        key_list = torch.from_numpy(masklist).nonzero()
        perturbation = (torch.rand_like(vid) * 2 - 1) * 0.05                                # initial perturbations
        MASK=torch.zeros(vid.size())                                                       #  frame mask
        MASK[key_list, :, :, :] = 1
        perturbation = perturbation*(MASK.cuda())
        adv_vid = torch.clamp(vid.clone() + perturbation, 0., 1.)
        adv_vids.append(adv_vid.unsqueeze(0))
        if (i+1)%16 == 0:
            adv_vids_ = torch.cat(adv_vids, 0)         
            _ , _, logits = vid_model(adv_vids_)
            representation.append(logits)
            adv_vids= []
    representation = torch.cat(representation, 0).float().cpu().numpy()  
    query_method = CoreSetMIPSampling(representation, input_shape=(image_size, image_size), num_labels=1, gpu=gpus)
    labeled_idx = np.random.choice(seq_len, initial_size, replace=False)
    vid_ = vid.permute(0, 2, 3, 1)
    vid_ = np.array(vid_.cpu())
    index = query_method.query(vid_, vid_label, labeled_idx, initial_size)
    masklist = np.zeros((seq_len))
    for i in index:
        if i not in labeled_idx:
            masklist[i] = 1
    masklist = torch.from_numpy(masklist).nonzero()
    res, iter_num, adv_vid = attack(vid_model, vid, vid_label, masklist, image_models, gpus)
    if res:
        AP = pertubation(vid, adv_vid)
        total_query_num += iter_num
        total_P_num += AP
        success_num += 1
        metric_path = os.path.join(output_path, 'metric.txt')  # save metric
        adv_path = os.path.join(output_path, 'adv_vid.npy')
        np.save(adv_path, adv_vid.cpu().numpy())
        f = open(metric_path, 'w')
        f.write(str(iter_num))
        f.write('\n')
        f.write(str(AP.cpu()))
        f.write('\n')
        f.write(str(masklist.cpu().data))
        f.close()
        f1 = open(av_metric, 'a')
        f1.write(str(total_query_num))
        f1.write('\n')
        f1.write(str(total_P_num))
        f1.write('\n')
        f1.close()
        print(f'this video use {iter_num} query')
    f1 = open(av_metric,'a')
    f1.write('------------')
    f1.write('\n')
    f1.write(str(success_num))
    f1.write('\n')
    f1.close()
print(f'model: {model_name} dataset: {dataset_name}')
print(f'success {success_num/test_num}')
print(f'map {total_P_num/success_num}')
print(f'mean query{total_query_num/success_num}')