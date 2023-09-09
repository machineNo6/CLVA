from .I3D_model import get_model

def generate_model_i3d(dataset):
    assert dataset in ['hmdb51', 'ucf101']
    if dataset == 'hmdb51':
        checkpoint='models/I3D/checkpoints/hmdb51_save_best.pth'
        num_class=51
    elif dataset == 'ucf101':
        checkpoint = 'models/I3D/checkpoints/ucf101_save_best.pth'
        num_class = 101
    model = get_model(checkpoint,num_class)
    return model