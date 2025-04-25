from utils import *

def parse_args():
    """Training Options for classification Experiments"""
    parser = argparse.ArgumentParser(description='ResNet101v2_Attention_Network_For_CD-AIVD')
    # choose model
    parser.add_argument('--model', type=str, default='ResNet101v2',
                        help='model name: ResNet101v2')
    # parameter for resnet101
    parser.add_argument('--channel_size', type=str, default='one',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='ResNet101v2',
                        help='ResNet101v2')
    parser.add_argument('--deep_supervision', type=str2bool, default=True, help='True or False (model==ResNet101v2)')


    # data and pre-process
    parser.add_argument('--dataset', type=str, default="CD-AIVD",
                        help='dataset name:  CD-AIVD')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when mode==Ratio')
    parser.add_argument('--root', type=str, default="dataset/")
    parser.add_argument('--suffix', type=str, default="")
    parser.add_argument('--split_method', type=str, default='80_20',
                        help='80_20')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='in_channel=1 for pre-process')
    parser.add_argument('--base_size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=50,
                        metavar='N', help='input batch size for \
                        training (default: 50)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help=' AdamW, Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: lr=1e-4)')
    # cuda and logging
    parser.add_argument('--gpus', type=str, default='1',
                        help='Training with GPUs, you can specify 1,3 for example.')


    args = parser.parse_args()
    # make dir for save result
    args.save_dir = make_dir(args.deep_supervision, args.dataset, args.model)
    # save training log
    save_train_log(args, args.save_dir)
    # the parser
    return args