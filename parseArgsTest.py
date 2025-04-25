from utils import *

def parse_args():
    """Training Options for classificaiton Experiments"""
    parser = argparse.ArgumentParser(description='ResNet101v2_for_AI voice detection')
    # choose model
    parser.add_argument('--model', type=str, default='ResNet101v2',
                        help='model name: ResNet101v2')

    # parameter for ResNet101
    parser.add_argument('--channel_size', type=str, default='one',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='ResNet101v2',
                        help='ResNet101v2')
    parser.add_argument('--deep_supervision', type=str2bool, default=True, help='True or False (model==ResNet101v2)')


    # data and pre-process
    parser.add_argument('--dataset', type=str, default='CD-AIVD',
                        help='dataset name: CD-AIVD')
    parser.add_argument('--st_model', type=str, default='CD-AIVD_ResNet_31_03_2025_wDS')
    parser.add_argument('--model_dir', type=str,
                        default = 'CD-AIVD_ResNet_31_03_2025_wDS/mIoU__ResNet_CD-AIVD_epoch.pth.tar',
                        help='CD-AIVD_ResNet_31_03_2025_wDS/mIoU__ResNet_CD-AIVD_epoch.pth.tar')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='in_channel=1 for pre-process')
    parser.add_argument('--base_size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--test_batch_size', type=int, default=50,
                        metavar='N', help='input batch size for \
                        testing (default: 50)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    
    # cuda and logging
    parser.add_argument('--gpus', type=str, default='1',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')


    args = parser.parse_args()

    # the parser
    return args