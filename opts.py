import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of action recognition models")
parser.add_argument('--exp_dir', type=str, default="experiments/exp_test",help='Path to experiment output directory')
parser.add_argument('--train_csv', type=str, default="",help='Path to experiment output directory')
parser.add_argument('--optim', type=str, default="sgd",help='')
parser.add_argument('--verification_csv', type=str, default="",help='Path to experiment output directory')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--save_every_n_epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--test_every_n_epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lora_rank', default=8, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--code_number', default=16, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume_main', default='', type=str, metavar='PATH',
                    help='path to latest resume_main checkpoint (default: none)')
parser.add_argument('--data_modality', default='image', type=str,
                    help='path to latest resume_main checkpoint (default: none)')
parser.add_argument('--backbone', default='clip_stadapt_imagenet', type=str,
                    help='path to latest resume_main checkpoint (default: none)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--wd', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--margin', default=0.4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument( '--only_eval_privacy', dest='only_eval_privacy', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--start_epoch', default=0, type=int, help='Batch size for training')
parser.add_argument('--text_use_method', type=str, default="none",help='Path to experiment output directory')
parser.add_argument('--relation_type', type=str, default="mlp",help='Path to experiment output directory')
parser.add_argument('--mimic_all', action='store_true', help='Enable verbose mode')
parser.add_argument('--no_codebook', action='store_true', help='Enable verbose mode')
parser.add_argument('--ft_all', action='store_true', help='Enable verbose mode')
parser.add_argument('--ft_backbone_lr_multi', default=1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--train_modality', type=str, default="all",help='')

parser.add_argument('--online_compute_id', action='store_true', help='Enable verbose mode')
parser.add_argument('--aug_color', action='store_true', help='Enable verbose mode')
parser.add_argument('--aug_resize', action='store_true', help='Enable verbose mode')

parser.add_argument('--no_randE', action='store_true', help='Enable verbose mode')
parser.add_argument('--no_randCrop', action='store_true', help='Enable verbose mode')
parser.add_argument('--clip_grad', default=0, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--local-rank', type=int, default="0")

