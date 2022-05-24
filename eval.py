import torch
import torch.utils.data.dataset
import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse

from data.data_loader import ShapeNet_Dataset
from models.Model_3DC2FT import Model_3DC2FT
from utils import metrics
from data import data_transforms
import config

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def parse_arguments(argv):
    """
        Parameters for calling eval.py
        e.g., python eval.py --dataset_mode "ShapeNet"

    :param argv: --dataset_mode {"ShapeNet", "Ours"}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_mode', help='Dataset for evaluation. {ShapeNet or Ours}', default='Ours')

    return parser.parse_args(argv)


def eval_ShapeNet(data_loader, model):
    """
        Evaluate ShapeNet dataset

    :param data_loader: ShapeNet dataloader
    :param model: 3D-C2FT model
    :return: IoU, F-score
    """

    total_ious, total_fscores, total_objects = 0., 0., 0

    model.eval()
    for i, (imgs, voxs, targets) in enumerate(tqdm(data_loader)):
        with torch.no_grad():

            pred_3d = model(imgs.to(config.device))
            pred_3d[pred_3d > config.thres] = 1
            pred_3d[pred_3d < 1] = 0

        ious = metrics.calculate_iou(pred_3d, voxs.to(config.device))
        fscores = metrics.calculate_fscore(pred_3d, voxs.to(config.device))

        total_ious += ious.sum()
        total_fscores += np.sum(fscores)
        total_objects += len(ious)

    return total_ious / total_objects, total_fscores / total_objects


def eval_our(data_dir, model):
    """
        Evaluate our real-life dataset

    :param data_dir: Our dataset directory path
    :param model: 3D-C2FT model
    """
    eval_transforms = data_transforms.Compose([
                data_transforms.Resize((224, 224)),
                data_transforms.CenterCrop((224, 224), (224, 224)),
                data_transforms.RandomBackground([[240, 240], [240, 240], [240, 240]]),
                data_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                data_transforms.ToTensor(),
            ])
    model.eval()
    print("-----------------------------------------------------")
    print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (datetime.now(), len(os.listdir(data_dir))))
    print("\n  Run testing: Our Dataset ...")

    for i, sample in enumerate(tqdm(os.listdir(data_dir))):
        imgs = [cv2.imread(data_dir + "/" + sample + "/" + fname, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
                for fname in os.listdir(data_dir + "/" + sample)]
        imgs = np.asarray(imgs[:len(imgs)])

        imgs = eval_transforms(imgs)

        with torch.no_grad():
            pred_3d = model(torch.unsqueeze(imgs, dim=0).to(config.device))[0]

        pred_3d[pred_3d > config.thres] = 1
        pred_3d[pred_3d < 1] = 0

        pred_3d = pred_3d.cpu().numpy()
        pred_3d = np.transpose(pred_3d, (0, 2, 1))
        ax2 = plt.figure(figsize=(6, 6), dpi=80).add_subplot(projection='3d')
        ax2.voxels(pred_3d, facecolors='silver', edgecolor='k')
        ax2.azim = 32
        ax2.elev = 10
        ax2.dist = 10
        ax2.axis('off')
        plt.savefig('result/our_dataset/'+sample+'.png', dpi=300)
        plt.close()


def main(args):
    # Choose the dataset for evaluation
    if args.dataset_mode == 'ShapeNet':
        print("-----------------------------------------------------")
        # Define dataset and generator
        print("Loading test dataset and creating generator ...")
        test_dataset = ShapeNet_Dataset(config, dataset_type=config.ShapeNet_eval_set)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size,
                                                  shuffle=True, num_workers=0)
        print("-----------------------------------------------------")

        # Define 3D-C2FT model
        print("\n\nLoading pretrained model - 3D-C2FT ...")
        print("-----------------------------------------------------")
        model = Model_3DC2FT(hybrid_backbone="densenet121", img_size=224, in_chans=3, reduce_ratio=2,
                             embed_dim=config.encoder_embed_dim, encoder_C2F_block=config.encoder_C2F_block,
                             layer_depth=config.C2F_layer_depth, num_heads=config.encoder_num_heads,
                             drop_rate=config.encoder_dropout_rate, attn_drop_rate=config.encoder_dropout_rate,
                             drop_path_rate=config.encoder_stochastic_drop_rate,
                             decoder_dropout=config.decoder_dropout_rate, decoder_output_size=32,
                             decoder_patch_size=config.decoder_patch_size, decoder_num_heads=config.decoder_num_heads,
                             refiner_layer_depth=config.refiner_layer_depth,
                             refiner_drop_path_rate=config.refiner_stochastic_drop_rate
                             ).to(config.device)
        model.load_state_dict(torch.load(config.pretrained_weights, map_location=config.device))
        print("-----------------------------------------------------")
        print("\n  Run testing: ShapeNet Dataset ...")
        mIoU, mFscore = eval_ShapeNet(test_loader, model)

        print("[RESULTS] mean IoU of {} set ({} images) is {:.4f}".format(config.ShapeNet_eval_set, len(test_dataset), mIoU))
        print("[RESULTS] mean F-score of {} set ({} images) is {:.4f}".format(config.ShapeNet_eval_set, len(test_dataset), mFscore))

    elif args.dataset_mode == 'Ours':
        # Define 3D-C2FT model
        print("\n\nLoading pretrained model - 3D-C2FT ...")
        print("-----------------------------------------------------")
        model = Model_3DC2FT(hybrid_backbone="densenet121", img_size=224, in_chans=3, reduce_ratio=2,
                             embed_dim=config.encoder_embed_dim, encoder_C2F_block=config.encoder_C2F_block,
                             layer_depth=config.C2F_layer_depth, num_heads=config.encoder_num_heads,
                             drop_rate=config.encoder_dropout_rate, attn_drop_rate=config.encoder_dropout_rate,
                             drop_path_rate=config.encoder_stochastic_drop_rate,
                             decoder_dropout=config.decoder_dropout_rate, decoder_output_size=32,
                             decoder_patch_size=config.decoder_patch_size, decoder_num_heads=config.decoder_num_heads,
                             refiner_layer_depth=config.refiner_layer_depth,
                             refiner_drop_path_rate=config.refiner_stochastic_drop_rate
                             ).to(config.device)
        model.load_state_dict(torch.load(config.pretrained_weights, map_location=config.device))
        eval_our(config.our_dataset_path, model)
        print("[RESULTS] 3D Reconstruction task is done!")


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    sys.exit(0)
