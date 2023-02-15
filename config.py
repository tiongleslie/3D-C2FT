import torch
from datetime import datetime


#   ShapeNet Dataset
ShapeNet_taxonomy_path              = "./dataset/ShapeNet/ShapeNet_taxonomy.json"
ShapeNet_image_path                 = "./dataset/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png"
ShapeNet_voxel_path                 = "./dataset/ShapeNet/ShapeNetVox32/%s/%s/model.binvox"
ShapeNet_selection_mode             = "random"
ShapeNet_eval_set                   = "test"


#   Our Dataset
our_dataset_path                    = "./dataset/Ours"


#    Traning Settings
save_dir                            = "./experiments/3DC2FT_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
lr                                  = 0.01
start_epoch                         = 0
total_epochs                        = 2000


#   Multi-view 2D Images Settings
n_views                             = 8
img_size                            = (224, 224)
crop_size                           = (128, 128)
train_random_bg_color_range         = [[225, 255], [225, 255], [225, 255]]
brightness                          = 0.4
contrast                            = 0.4
saturation                          = 0.4
noise_std                           = 0.1

eval_random_bg_color_range          = [[240, 240], [240, 240], [240, 240]]
mean                                = [0.5, 0.5, 0.5]
std                                 = [0.5, 0.5, 0.5]


#   3D Class Categories
class_names                         = ('airplane', 'bench', 'cabinet', 'car', 'chair', 'display', 'lamp',
                                       'loudspeaker', 'rifle', 'sofa', 'table', 'telephone', 'watercraft')
class_dict                          = {
                                        '02691156': 0,
                                        '02828884': 1,
                                        '02933112': 2,
                                        '02958343': 3,
                                        '03001627': 4,
                                        '03211117': 5,
                                        '03636649': 6,
                                        '03691459': 7,
                                        '04090263': 8,
                                        '04256520': 9,
                                        '04379243': 10,
                                        '04401088': 11,
                                        '04530566': 12,
                                      }


#   NVidia Device Settings
device                              = torch.device("cuda:0")
preload_2d_to_ram                   = False
preload_3d_to_ram                   = False
batch_size                          = 16


#   3D-C2FT Settings
pretrained_weights                  = "pretrained/3D-C2FT.pt"
thres                               = 0.3

encoder_embed_dim                   = 768
encoder_C2F_block                   = 3
C2F_layer_depth                     = 4
encoder_num_heads                   = 12

decoder_patch_size                  = 4
decoder_num_heads                   = 8

refiner_layer_depth                 = 6

encoder_stochastic_drop_rate        = 0.
encoder_dropout_rate                = 0.
decoder_dropout_rate                = 0.
refiner_stochastic_drop_rate        = 0.

voxel_output                        = 32
