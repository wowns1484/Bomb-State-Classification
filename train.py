from utils.utils import set_seed, parse_args
from dataset import BombDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from modules.model import create_model
from modules.loss import create_criterion
from modules.optimizer import create_optimizer
from modules.scheduler import create_scheduler
import os

def validation(model, dataloader, criterion, args):
    val_losses, val_acc = [], 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.float().to(args.device), labels.to(args.device)
    
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            val_losses.append(loss.detach().item())
            val_acc += (torch.argmax(outputs, dim=1) == labels).detach().sum().item()
    
    return np.mean(val_losses), val_acc

def train(model, dataloader, val_dataloader, optimizer, criterion, scheduler, args):
    min_loss, patience = 100000, 0
    
    train_avg_losses, train_avg_accs = [], []
    val_avg_losses, val_avg_accs = [], []
    
    for epoch in range(args.epochs):
        model.train()
        train_losses, train_acc = [], 0
        
        for images, labels in tqdm(dataloader):
            images, labels = images.float().to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        
        lr = optimizer.param_groups[0]['lr']
        train_avg_loss = np.mean(train_losses)
        train_avg_acc = train_acc / len(train_dataset.images)
        print(f"Train epoch: {epoch + 1}, lr: {lr}, loss: {train_avg_loss}, acc: {train_avg_acc}")
        
        train_avg_losses.append(train_avg_loss)
        train_avg_accs.append(train_avg_acc)
        
        val_avg_loss, val_acc = validation(model, val_dataloader, criterion, args)
        val_avg_acc = val_acc / len(val_dataset.images)
        print(f"Validation epoch: {epoch + 1}, loss: {val_avg_loss}, acc: {val_avg_acc}")
        
        val_avg_losses.append(val_avg_loss)
        val_avg_accs.append(val_avg_acc)
        
        if val_avg_loss < min_loss:
            min_loss = val_avg_loss
            patience = 0
        else:
            patience += 1
            print(f"Early Stopping patience: {patience}")
                
        if patience > args.patience:
            print(f"Early Stopping at epoch {epoch + 1}.")
            break
    
    train_avg_losses, train_avg_accs = np.array(train_avg_losses), np.array(train_avg_accs)
    val_avg_losses, val_avg_accs = np.array(val_avg_losses), np.array(val_avg_accs)
    train_logs = np.stack([train_avg_losses, train_avg_accs, val_avg_losses, val_avg_accs], axis=1)

    train_log_df = pd.DataFrame(train_logs, columns=["train_loss", "train_accuracy", "validation_loss", "validation_accuracy"])
    train_log_df.to_csv(os.path.join(args.save_dir, f'{args.experiment_name}.csv'), sep=",")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        ToTensorV2()
    ])

    test_transforms = A.Compose([
        A.Resize(224, 224),
        ToTensorV2()
    ])

    train_dataset = BombDataset(args.data_dir, "train", train_transforms, args.split_rate)
    val_dataset = BombDataset(args.data_dir, "val", test_transforms, args.split_rate)
    
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    model = create_model(args.use_model)
    model = model.to(args.device)
    
    criterion = create_criterion(args.use_loss)
    optimizer = create_optimizer(model.parameters(), args.use_optimizer)
    scheduler = create_scheduler(optimizer, args.use_scheduler)
    
    train(model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, args)