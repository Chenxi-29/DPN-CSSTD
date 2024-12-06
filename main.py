import argparse
from DPNCSSTD_detection import DPNCSSTD_detection
from DPNCSSTD_finetune import DPNCSSTD_finetune
from DPNCSSTD_pretrain import DPNCSSTD_pretrain

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DPN-CSSTD")

    parser.add_argument("--windowsize", type=int, default=5)

    parser.add_argument("--pre_image_file", type=str, default='Sandiego')
    parser.add_argument("--pre_epoches", type=int, default=60)
    parser.add_argument("--pre_batchsize", type=int, default=256)
    parser.add_argument("--pre_lr", type=int, default=0.001)
    parser.add_argument("--pre_savepath", type=str, default='./pth/DPNCSSTD_pretrain/')

    parser.add_argument("--test_image", type=str, default='AVIRIS')
    parser.add_argument("--usepretrain", type=str, default=True)

    parser.add_argument("--method", type=str, default='Hyperpixel_segmentation')
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--show", type=str, default=False)
    parser.add_argument("--number", type=int, default=20)

    parser.add_argument("--finetune_epoches", type=int, default=200)
    parser.add_argument("--finetune_lr", type=int, default=0.001)
    parser.add_argument("--finetune_savepath", type=str, default='./pth/DPNCSSTD_finetune/')

    parser.add_argument("--detection_savepath", type=str, default='./result/')

    args = parser.parse_args(args=[])
    # DPNCSSTD_pretrain(args)
    DPNCSSTD_finetune(args)
    DPNCSSTD_detection(args)

