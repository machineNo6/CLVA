import argparse
import pickle

# 获取验证数据集
V_P_HMDB = 'datasets/HMDB51/'
V_P_UCF = 'datasets/UCF101/'
V_A_HMDB = 'datasets/annotations/hmdb51/hmdb51_1.json'
V_A_UCF = 'datasets/annotations/ucf101/ucf101_01.json'


def hmdb51_parse_opts():
    parser = argparse.ArgumentParser()
    # 加载数据集的一系列参数
    parser.add_argument('--dataset',default='hmdb51',type=str,help='The type of dataset. (ucf101 | hmdb51)')
    parser.add_argument('--root_path',default='/mnt/disk1/chenjiefu/SVA/SVA/',type=str,help='Root directory path of data')
    parser.add_argument('--video_path',default=V_P_HMDB,type=str,help='Directory path of Videos')
    parser.add_argument('--annotation_path',default=V_A_HMDB,type=str,help='Annotation file path')
    # assign by outer parser
    parser.add_argument('--data_type',default='flow',type=str,help='RGB or Optical flow, (rgb | flow)')
    # parameters for spatial transforms
    parser.add_argument('--flow_channels',default=3,type=int,help='Random Seed.')
    parser.add_argument('--manual_seed',default=1024,type=int,help='Random Seed.')
    parser.add_argument('--norm_value',default=1,type=int,help='Original pixels/norm_value')
    parser.add_argument('--initial_scale',default=1.0,type=float,help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales',default=5,type=int,help='Number of scales for multiscale cropping')
    parser.add_argument('--no_mean_norm',action='store_true',help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm',action='store_true',help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--train_crop',default='corner',type=str,help='Spatial cropping method(random | corner | center)')
    parser.add_argument('--sample_size',default=112,type=int,help='Height and width of inputs')
    parser.add_argument('--scale_step',default=0.84089641525,type=float,help='Scale step for multiscale cropping')
    # parameters for temporal transforms
    parser.add_argument('--sample_duration',default=64,type=int,help='Temporal duration of inputs')
    parser.add_argument('--scale_in_test',default=1.0,type=float,help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test',default='c',type=str,help='Cropping method (c | tl | tr | bl | br) in test')
    # parameters for target transforms
    args = parser.parse_args()
    return args


def ucf101_parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='ucf101',type=str,help='The type of dataset. (ucf101 | hmdb51)')
    parser.add_argument('--root_path',default='/mnt/disk1/chenjiefu/SVA/SVA/',type=str,help='Root directory path of data')
    parser.add_argument('--video_path',default=V_P_UCF,type=str,help='Directory path of Videos')
    parser.add_argument('--annotation_path',default=V_A_UCF,type=str,help='Annotation file path')
    parser.add_argument('--data_type',default='flow',type=str,help='RGB or Optical flow, (rgb | flow)')
    # parameters for spatial transforms
    parser.add_argument('--flow_channels',default=3,type=int,help='Random Seed.')
    parser.add_argument('--manual_seed',default=1024,type=int,help='Random Seed.')
    parser.add_argument('--norm_value',default=1,type=int,help='Original pixels/norm_value')
    parser.add_argument('--initial_scale',default=1.0,type=float,help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales',default=5,type=int,help='Number of scales for multiscale cropping')
    parser.add_argument('--no_mean_norm',action='store_true',help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm',action='store_true',help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--train_crop',default='corner',type=str,help='Spatial cropping method(random | corner | center)')
    parser.add_argument('--sample_size',default=112,type=int,help='Height and width of inputs')
    parser.add_argument('--scale_step',default=0.84089641525,type=float,help='Scale step for multiscale cropping')
    # parameters for temporal transforms
    parser.add_argument('--sample_duration',default=64,type=int,help='Temporal duration of inputs')
    parser.add_argument('--scale_in_test',default=1.0,type=float,help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test',default='c',type=str,help='Cropping method (c | tl | tr | bl | br) in test')
    # parameters for target transforms
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = hmdb51_parse_opts()
    dict_args = vars(args)
    with open('./hmdb51_params.pkl', 'wb') as opt:
        pickle.dump(dict_args, opt)

    args = ucf101_parse_opts()
    dict_args = vars(args)
    with open('./ucf101_params.pkl', 'wb') as opt:
        pickle.dump(dict_args, opt)