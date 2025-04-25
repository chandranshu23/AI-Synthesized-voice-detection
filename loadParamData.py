import os

def loadDataset(root, dataset):
    train_txt = os.path.join(root, dataset, "train.txt")
    test_txt = os.path.join(root, dataset, "test.txt")
    eval_txt = os.path.join(root, dataset, "eval.txt")  # New eval set

    train_img_ids, test_img_ids, eval_img_ids = [], [], []

    with open(train_txt, "r") as f:
        train_img_ids = [line.strip() for line in f]

    with open(test_txt, "r") as f:
        test_img_ids = [line.strip() for line in f]

    with open(eval_txt, "r") as f:
        eval_img_ids = [line.strip() for line in f]

    return train_img_ids, test_img_ids, eval_img_ids

def loadParams(channelSize, backbone):
    if channelSize == 'one':
        nb_filter =[4,8,16,32,64]
    elif channelSize == 'two':
        nb_filter = [8, 16, 32, 64, 128]
    elif channelSize == 'three':
        nb_filter = [16, 32, 64, 128, 256]
    elif channelSize == 'four':
        nb_filter = [32, 64, 128, 256, 512]

    if backbone == 'ResNet101v2':
        num_blocks = [3, 4, 23, 3]  # ResNet-101 architecture
    

    return nb_filter, num_blocks