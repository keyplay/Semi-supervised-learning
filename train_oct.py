from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer, BasicDataset
from semilearn.datasets.cv_datasets.imagenet import ImagenetDataset
from torchvision import transforms, datasets
import os
import torch
from randaugment import RandAugmentMC

config = {
    'algorithm': 'comatch',
    'net': 'resnet50', #'vit_small_patch16_224',
    'use_pretrain': True, 
    'pretrain_path': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth', #vit_small_patch16_224_mlp_im_1k_224.pth',
    'save_name': 'comatch',
    
    # optimization configs
    'epoch': 100,  # set to 100
    'num_train_iter': 1000,  # set to 102400
    'num_eval_iter': 10,   # set to 1024
    'num_log_iter': 10,    # set to 256
    'optim': 'AdamW',
    'lr': 5e-4,
    'layer_decay': 0.5,
    'batch_size': 16,
    'eval_batch_size': 16,


    # dataset configs
    'dataset': 'oct',
    'num_labels': 80,
    'num_classes': 4,
    'img_size': 32, #224,
    'crop_ratio': 0.875,
    'data_dir': './data',
    
    #'ulb_samples_per_class': None,

    # algorithm specific configs
    'hard_label': True,
    'uratio': 2,
    'ulb_loss_ratio': 1.0,

    # device configs
    'gpu': 0,
    'world_size': 1,
    'distributed': False,
    "num_workers": 0,
}

def process_item(item):
    image, label = item
    return image, label

def get_data_concurrent(dset):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_item, dset)
    return results

config = get_config(config)
print(config)

algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)
print(algorithm)

data_dir = './data/OCT2017'
TRAIN_LABEL = 'train_hard'
TRAIN_UNLABEL = 'train'
VAL = 'val'
TEST = 'test'

transform_weak = transforms.Compose([
    transforms.RandomResizedCrop(config.img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_strong = transforms.Compose([
    transforms.RandomResizedCrop(config.img_size),
    transforms.RandomHorizontalFlip(),
    RandAugmentMC(n=2, m=10),
    transforms.ToTensor(),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.img_size),
    transforms.ToTensor(),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

lb_dataset = ImagenetDataset(root=os.path.join(data_dir, TRAIN_LABEL), transform=transform_weak, ulb=False, alg=config.algorithm,strong_transform=transform_strong )
#dset = datasets.ImageFolder(os.path.join(data_dir, TRAIN_LABEL))
#lb_data, lb_targets = zip(*[(image, label) for image, label in dset])
#lb_data, lb_targets = zip(*get_data_concurrent(dset))
#lb_dataset = BasicDataset(config.algorithm, lb_data, lb_targets, config.num_classes, transform_weak, is_ulb=False)

ulb_dataset = ImagenetDataset(root=os.path.join(data_dir, TRAIN_UNLABEL), transform=transform_weak, ulb=True, alg=config.algorithm, strong_transform=transform_strong)
#dset = datasets.ImageFolder(os.path.join(data_dir, TRAIN_UNLABEL))
#lb_data, lb_targets = zip(*[(image, label) for image, label in dset])
#ulb_dataset = BasicDataset(config.algorithm, lb_data, lb_targets, config.num_classes, transform_weak, is_ulb=True, strong_transform=transform_strong)

eval_dataset = ImagenetDataset(root=os.path.join(data_dir, TEST), transform=transform_val, ulb=False, alg=config.algorithm)
#dset = datasets.ImageFolder(os.path.join(data_dir, TEST))
#lb_data, lb_targets = zip(*[(image, label) for image, label in dset])
#eval_dataset = BasicDataset(config.algorithm, lb_data, lb_targets, config.num_classes, transform_val, is_ulb=False)

train_lb_loader = get_data_loader(config, lb_dataset, config.batch_size)
train_ulb_loader = get_data_loader(config, ulb_dataset, int(config.batch_size * config.uratio))
eval_loader = get_data_loader(config, eval_dataset, config.eval_batch_size)

print('start training...')

trainer = Trainer(config, algorithm)
trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
trainer.evaluate(eval_loader)
