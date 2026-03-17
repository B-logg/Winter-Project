import argparse
from config_mf import MAX_EPOCHS, BATCH_SIZE,LR 


class BaseParams(argparse.ArgumentParser):
    def __init__(self):
        super(BaseParams, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument(
            "--net",
            type=str,
            default='UNet_carbon',
            help="network name"
        )
        self.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Fixed random seed"
        )
        self.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help='local rank for DistributedDataParallel(do not modify)')
        self.add_argument("--tensorboard", default=True, action="store_true")

        self.add_argument(
            "--num_workers",
            type=int,
            # default=2,
            default=0,
            help="num_workers"
        )


class TrainParser(BaseParams):
    def __init__(self):
        super(TrainParser, self).__init__()
        self.add_argument("--output-dir", type=str, help="output directory")        
        self.add_argument("--train_batch_size", default=BATCH_SIZE, type=int, help="train batch size")
        self.add_argument("--val_batch_size", default=BATCH_SIZE, type=int, help="validataion batch size")
        self.add_argument("--total-epoch", default=MAX_EPOCHS, type=int, help="total num epoch")
        self.add_argument("--eval-freq", default=5, type=int, help="evaluation frequency")
        self.add_argument("--save-freq", default=10, type=int, help="save frequency")
        self.add_argument("--learning-rate", default=LR, type=float, help="learning late") 
        self.add_argument('--pretrained', action="store_true", help='Start with pretrained model (if avail)')
        self.add_argument('--resume', action="store_true", help='resume from checkpoint')
        self.add_argument('--opt', default="adam", type=str, help="nadam, adam")
        self.add_argument('--lrs', default="cosinealr", type=str, help="cosinealr, steplr")
        self.add_argument('--enc_dropout', action="store_true", help='dropout for encoder')
        self.add_argument("--image_type", default="forest_SN_10", type=str, help="test image type")
        
class InferenceParser(BaseParams):
    def __init__(self):
        super(InferenceParser, self).__init__()        
        self.add_argument("--image_type", default="forest_SN_10", type=str, help="test image type")
        