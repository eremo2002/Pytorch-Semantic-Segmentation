import datetime
from cmath import inf
import os
import random
import argparse
from random import shuffle
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from model.Res_Unet_PP import ResUnetPlusPlus
from util.dataset import *
from util.loss import *
from util.augment import *


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def dice_coeff_metric(pred, target):
    pred = (pred>0).float()
    return 2.0 * (pred*target).sum() / ((pred+target).sum() +1.0)

def train():
    parser = argparse.ArgumentParser(description='argparse argument')

    parser.add_argument('--epochs',
                        type=int,
                        help='epoch',                        
                        default='300',
                        dest='epochs')

    parser.add_argument('--batch_size',
                        type=int,
                        help='batch_size',                        
                        default='4',
                        dest='batch_size')
    
    args = parser.parse_args()


    # hyper parameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    current_time = datetime.datetime.now()
    log_file_name = f'{current_time.year}-{current_time.month}-{current_time.day}_{current_time.hour}{current_time.minute}{current_time.second}'    
    f_out = open(f'weight\\{log_file_name}.txt', 'w')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(device)
    
    train_transform = transforms.Compose([        
                                        Random_Gamma(p=0.5, range=(0.5, 1.5)),
                                        Random_Cutout(p=0.5),
                                        Random_Brightness(p=0.5, range=0.5),
                                        Random_Contrast(p=0.5, range=0.5),
                                        Horizontal_Flip(p=0.5),                                        
                                        Shift_X(p=0.5, range=50),
                                        Shift_Y(p=0.5, range=50),
                                        Affine_Shear(p=0.5, range_mx=0.5, range_my=0.5),
                                        Rotation(p=0.5, angle=(-20, 20)),                                                                                
                                        Normalize(),
                                        ToTensor()
                                        ])

    train_dataset = Custom_Dataset(root_dir='your path',
                                set_name='your set name',
                                transform=train_transform)

    train_loader = Custom_Dataset(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0)



    val_transform = transforms.Compose([Normalize(),
                                        ToTensor()])
    
    val_dataset = PD_dataset(root_dir='your path',
                            set_name='your set name',
                            transform=val_transform)

    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=0)
    
 
    model = ResUnetPlusPlus(channel=1)
    model.to(device)
    summary(model, (1, 512, 512), batch_size=BATCH_SIZE)

    
    criterion = BCE_DICE_Loss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5, verbose=1)         
    
    # # resume
    # checkpoint = torch.load('./weight/DiceLoss_model_00013.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # resume_epoch = checkpoint['epoch']
    # criterion = checkpoint['loss']

    # tensorboard
    writer = SummaryWriter('runs/')    

    best_val_loss = inf
    
    # resume_epoch = 1
    for epoch in range(1, EPOCHS+1):
        train_loss = 0.
        val_loss = 0.

        train_dice_score_list = []
        val_dice_score_list = []

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        model.train()
        for i, data in loop:
            image = data['image'].to(device)
            mask = data['mask'].to(device)

            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, mask) # pred, gt

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            loop.set_description(f'Epoch [{epoch}/{EPOCHS}')
        
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', current_lr, epoch)
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)

            for j, data in loop:
                image = data['image'].to(device)
                mask = data['mask'].to(device)

                pred = model(image)
                
                loss = criterion(pred, mask)
                val_loss += loss.item()

                loop.set_description(f'valid')
        
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
    

        print(f'Epoch: {epoch}\t train_loss: {train_loss}\t val_loss: {val_loss}')
       

        f_out.write(str(epoch)+','+str(train_loss)+','+str(val_loss)+'\n')

        if best_val_loss > val_loss:
            print('=' * 100)
            print(f'val_loss is improved from {best_val_loss:.8f} to {val_loss:.8f}\t saved current weight')
            print('=' * 100)
            best_val_loss = val_loss

            # torch.save(model, 'model.pth')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion},
                        # f'weight/{str(criterion).split("()")[0]}_model_{epoch:04d}_{val_loss:.4f}.pth'
                        f'weight/ResUNetPP_crop_v2_{str(criterion).split("()")[0]}_model_best.pth')

        torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion},
                        # f'weight/{str(criterion).split("()")[0]}_model_{epoch:04d}_{val_loss:.4f}.pth'
                        f'weight/ResUNetPP_crop_v2_{str(criterion).split("()")[0]}_model_last.pth')


    writer.close()    
    f_out.close()

if __name__ == '__main__':
    seed_everything(42)
    train()
