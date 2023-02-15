import numpy as np
import torch
import torch.optim as optim
import torch.utils.data.dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
import os

from data.data_loader import ShapeNet_Dataset
from models.Model_3DC2FT import Model_3DC2FT
from utils.loss import custom_loss
import config

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


#   Learning rate scheduling
def update_lr(optimizer, epoch):
    if epoch == 0:
        lr = config.lr
    elif epoch == 2:
        lr = 0.001
    elif epoch == 5:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(data_loader, model, criterion, optimizer, epoch, writer):
    model.train()
    decoder_mse_losses = []
    decoder_ssim_losses = []
    total_losses = []

    for i, (inputs, gt_3d, gt_cls) in enumerate(tqdm(data_loader)):
        pred_3d = model(inputs.to(config.device))
        decoder_mse_loss, decoder_ssim_loss = criterion(pred_3d, gt_3d, config.device)
        decoder_mse_losses.append(decoder_mse_loss)
        decoder_ssim_losses.append(decoder_ssim_loss)

        total_loss = decoder_ssim_loss + decoder_mse_loss
        total_losses.append(total_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    decoder_mse_loss = torch.mean(torch.tensor(decoder_mse_losses))
    decoder_ssim_loss = torch.mean(torch.tensor(decoder_ssim_losses))
    total_loss = torch.mean(torch.tensor(total_losses))

    print("    Train mse, ssim, total loss: {:.4f}, {:.4f}, {:.4f}".format(decoder_mse_loss, decoder_ssim_loss,
                                                                           total_loss))

    writer.add_scalar("train/decoder_mse_loss", decoder_mse_loss, epoch)
    writer.add_scalar("train/decoder_ssim_loss", decoder_ssim_loss, epoch)
    writer.add_scalar("train/total_loss", total_loss, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)

    return total_loss


def val(data_loader, model, criterion, epoch, writer):
    model.eval()
    decoder_mse_losses = []
    decoder_ssim_losses = []
    total_losses = []
    for i, (inputs, gt_3d, gt_cls) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            pred_3d = model(inputs.to(config.device))
        decoder_mse_loss, decoder_ssim_loss = criterion(pred_3d, gt_3d, config.device)
        decoder_mse_losses.append(decoder_mse_loss)
        decoder_ssim_losses.append(decoder_ssim_loss)
        total_loss = decoder_ssim_loss + decoder_mse_loss
        total_losses.append(total_loss)

    decoder_mse_loss = torch.mean(torch.tensor(decoder_mse_losses))
    decoder_ssim_loss = torch.mean(torch.tensor(decoder_ssim_losses))
    total_loss = torch.mean(torch.tensor(total_losses))

    print(
        "    Val mse, ssim, total loss: {:.4f}, {:.4f}, {:.4f}".format(decoder_mse_loss, decoder_ssim_loss, total_loss))

    writer.add_scalar("val/decoder_mse_loss", decoder_mse_loss, epoch)
    writer.add_scalar("val/decoder_ssim_loss", decoder_ssim_loss, epoch)
    writer.add_scalar("val/total_loss", total_loss, epoch)

    return total_loss


def main():
    cur_loss = np.Inf

    # Define dataset and generator
    print("-----------------------------------------------------")
    print("Loading dataset and creating generator...")

    train_dataset = ShapeNet_Dataset(config, dataset_type="train", train_augmentation=True,
                                     preload_2d_to_ram=config.preload_2d_to_ram,
                                     preload_3d_to_ram=config.preload_3d_to_ram)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    val_dataset = ShapeNet_Dataset(config, dataset_type="val", train_augmentation=False,
                                   preload_2d_to_ram=config.preload_2d_to_ram,
                                   preload_3d_to_ram=config.preload_3d_to_ram)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print('  No. of train dataset: {}'.format(len(train_dataset)))
    print('  No. of val dataset: {}'.format(len(val_dataset)))
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
    if config.pretrained_weights is not None:
        model.load_state_dict(torch.load(config.pretrained_weights, map_location=config.device))

    print("-----------------------------------------------------")
    # Define loss function, optimizer, tensorboard writer
    print("\n\nLoading other parameters...")
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    criterion = custom_loss
    writer = SummaryWriter(log_dir=config.save_dir + '/summary')
    os.makedirs(config.save_dir + "/checkpoints", exist_ok=True)

    # Start training
    print("-----------------------------------------------------")
    for epoch in range(config.start_epoch, config.total_epochs):
        # Update learning rate
        update_lr(optimizer, epoch)
        get_lr(optimizer)

        print("[EPOCH {}/{}]".format(epoch, config.total_epochs - 1))

        print("  Run training ...")
        train_loss = train(train_loader, model, criterion, optimizer, epoch, writer)

        print("  Run validation ...")
        val_loss = val(val_loader, model, criterion, epoch, writer)

        # Save model weights if total loss decreases
        if cur_loss > val_loss:
            cur_loss = val_loss
            torch.save(model.state_dict(), "{}/checkpoints/best_model.pt".format(config.save_dir))

    writer.close()
    print("-----------------------------------------------------")


if __name__ == '__main__':
    main()
    sys.exit(0)
