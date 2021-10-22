'''
chenyangzhu
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import albumentations as A
import albumentations.pytorch as Apy
# import torchvision
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
import random
from sklearn.model_selection import StratifiedKFold
import cv2
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup
import argparse
# 固定seed
def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./', type=str, help='Data Path')
parser.add_argument('--seed', default=0, type=int, help='Fix Seed, default is 0')
parser.add_argument('--lr', default=1e-5, type=float, help='Learning Rate, default is 1e-5')
parser.add_argument('--bs', default=16, type=int, help='Batch Size, default is 8 * 2')
args = parser.parse_args()

# 预处理数据，获取每个数据集的id和label
train_root = os.path.join(args.path, 'train')
test_root = os.path.join(args.path, 'test')
file = os.path.join(args.path, 'train_clean.csv')
output_path = os.join(args.path, 'output')

img_name = []
img_label = []

pd_data = pd.read_csv(file)
for img in pd_data['image']:
    img_name.append(img)
for label in pd_data['label']:
    img_label.append(label)

# 设置验证集和训练集的idx
img_name, img_label = np.array(img_name), np.array(img_label)

'''
k-fold
'''
fold_num = 5
train_fold, val_fold = [], []
skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=0)
for train_idx, val_index in skf.split(img_name, img_label):
    train_fold.append(train_idx)
    val_fold.append(val_index)

test_name = []
for file in os.listdir(test_root):
    if file[0] != '.':  # MacOS会有._xxx的缓存
        test_name.append(file)
test_name = np.array(test_name)

train_transform = A.Compose([
    A.Resize(440, 440),
    A.RandomCrop(384, 384),
    A.RandomRotate90(),
    A.Transpose(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.ColorJitter(brightness=0, p=0.3),
    # A.HueSaturationValue(sat_shift_limit=20, p=0.3),
    A.ShiftScaleRotate(p=0.2),
    A.GaussNoise(p=0.3),
    A.OneOf([
        A.ISONoise(),
        A.MedianBlur(blur_limit=3),
        A.Blur(blur_limit=3),
        ], p=0.2),
    A.RandomBrightness(limit=(-0.1, 0.4)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),  # imagenet
    A.GridDropout(ratio=0.3),
    Apy.ToTensorV2(p=1.0),
])

test_transform = A.Compose([
    A.Resize(520, 520),
    A.CenterCrop(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    Apy.ToTensorV2(p=1.0),
])

mixup_args = dict(
            mixup_alpha=1, cutmix_alpha=1, cutmix_minmax=None,
            prob=1, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=4)
mixup_fn = Mixup(**mixup_args)

# 三个数据集通用
class Set_data(data.Dataset):
    def __init__(self, root, id, label=None, transform=None, train=False):
        super().__init__()
        self.root = root
        self.id = id 
        self.label = label 
        self.transform = transform
        self.train = train
        if self.train:
            assert len(self.id) == len(self.label)

    def __getitem__(self, index):
        
        img_path = os.path.join(self.root, self.id[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        if self.train:  # train
            label = torch.tensor(self.label[index])
            return img, label
        else:                       # test
            return img

    def __len__(self):
        return len(self.id)

def train_one(train_loader, model, criterion, optimizer, scheduler=None):
    model.train()
    correct = 0
    # train_loss = 0
    for img, label in train_loader:
        img, label = img.cuda(), label.cuda()
        img, label = mixup_fn(img, label)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        # train_loss += loss.item()
        out = out.data 
        out = out.argmax(dim=1, keepdim=True).t().squeeze()  
        target = label.data
        target = target.argmax(dim=1, keepdim=True).t().squeeze()
        correct += sum(out.eq(target)).item()
        # lr = optimizer.state_dict()['param_groups'][0]['lr']
    scheduler.step()
    return correct

@torch.no_grad()
def predict(val_loader, model):
    model.eval()
    val_acc = 0
    for img, label in val_loader:
        img, label = img.cuda(), label.cuda()
        out = model(img)
        out = out.data  
        out = out.argmax(dim=1, keepdim=True).t().squeeze() 
        target = label.data
        # true_data = out.eq(target)
        val_acc += sum(out.eq(target)).item()

    return val_acc


lr = args.lr
train_batch = args.bs
test_batch = train_batch
epoch = 20

fix_seed(args.seed)  # 固定seed
fold_acc = 0
model_save_path = os.path.join(args.path, 'model_save')

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
for fold in range(fold_num):
    train_name, train_label = img_name[train_fold[fold]], img_label[train_fold[fold]]
    val_name, val_label = img_name[val_fold[fold]], img_label[val_fold[fold]]
    train_data = Set_data(train_root, train_name, train_label, transform=train_transform, train=True)
    train_loader = data.DataLoader(train_data, batch_size=train_batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_data = Set_data(train_root, val_name, val_label, transform=test_transform, train=True)
    val_loader = data.DataLoader(val_data, batch_size=test_batch, shuffle=False, num_workers=4)

    model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=4)
    model = nn.DataParallel(model).cuda()
    if fold == 0:
        print(model)

    best_acc = 0.7  # acc为第一条件
    train_best = 0
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 30, eta_min=1e-6, last_epoch=-1)
    criterion = SoftTargetCrossEntropy().cuda()
    for ep in tqdm(range(epoch)):
        correct = train_one(train_loader, model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        acc = correct / len(train_data)
        val_acc = predict(val_loader, model)
        val_acc = val_acc / len(val_data)
        print('fold {} epoch: {}'.format(fold, ep))
        print('train acc : {}'.format(acc))
        print('val acc : ', val_acc)
        if val_acc > best_acc : 
            best_acc = val_acc
            torch.save(model.module.state_dict(), f'{model_save_path}/swin_{fold}.pth')
            print('save model')
    fold_acc += best_acc

    print('best val is : ', best_acc)

print('Conclusion : K-Fold Acc is : ', fold_acc/fold_num)


test_data = Set_data(test_root, test_name, transform=test_transform)
test_loader = data.DataLoader(test_data, batch_size=test_batch, shuffle=False, num_workers=4)

test_allout = torch.zeros(len(test_data), 4)
for fold in range(fold_num):
    model = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=4)
    checkpoint = torch.load(f'{model_save_path}/swin_{fold}.pth')
    model.load_state_dict(checkpoint)
    model = nn.DataParallel(model).cuda()

    model.eval()
    test_out = []
    with torch.no_grad():
        for img in test_loader:
            img = img.cuda()
            out = model(img)
            out = out.data.cpu()
            test_out += list(out.numpy())
    test_out = torch.as_tensor(test_out)
    # torch.save(test_allout, f'/pth/swin3_{fold}_ns.pth')  
    test_allout += test_out
test_allout /= fold_num     
test_allout = torch.nn.functional.softmax(test_allout, 1)
# torch.save(test_allout, '/pth/swin3.pth')  # 保存输出
out = test_allout.argmax(dim=1, keepdim=True).t().squeeze()
test_out = out.numpy()
save_file = f'{output_path}/swin.csv'  
class_name = ['image_id', 'category_id']
out_csv = {class_name[0]:test_name, class_name[1]:test_out}
df = pd.DataFrame(out_csv)
df.to_csv(save_file, index=False)




