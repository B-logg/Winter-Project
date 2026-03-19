import os
import torch
import torch.distributed as dist
import numpy as np
import torch.backends.cudnn as cudnn
from config.param_parser import TrainParser
from utils.utils import make_output_directory, load_checkpoint_files, save_checkpoint
from utils.logger import create_logger
from tensorboardX import SummaryWriter
from data.build import build_data_loader
from core.train_core import train_one_epoch
from core.validation_core import validation_one_epoch
from model.build import build_model
import torchsummary
from model.metrics import CarbonLoss
from utils.tensorboard import log_tensorboard
import time
import datetime
from config_mf import CONFIGURE, BATCH_SIZE

def main(args, world_size, rank):
    cfg = CONFIGURE(args.image_type)
    start_epoch = 0
    num_epochs = args.total_epoch
    best_loss = 9999
    best_acc = -9999

    log_dir, pth_dir, tensorb_dir = make_output_directory(args, cfg)
    logger = create_logger(log_dir, dist_rank=0, name='')

    #create tensorboard
    if args.tensorboard:
        writer_tb = SummaryWriter(log_dir=tensorb_dir)

    #load data
    train_loader = build_data_loader("datalists/", cfg.TRAIN_CSV,
                                     BATCH_SIZE, args.num_workers, args.local_rank, cfg)
    val_loader = build_data_loader( "datalists/", cfg.VAL_CSV,
                                     BATCH_SIZE, args.num_workers, args.local_rank, cfg, shuffle=False)

    #build model
    model = build_model(args.net, num_class=cfg.NUM_CLASSES, dropout= args.enc_dropout) 
    model.cuda()
    
    if rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True, broadcast_buffers=False)

    #create optimizer
    logger.info(f"optimizer = {args.opt} ......")
    if args.opt == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.learning_rate)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info(f"lr_scheduler = {args.lrs} ......")
    if args.lrs == "cosinealr":
        #create scheduler stochastic gradient descent with warm restarts(SGDR)
        tmax = np.ceil(args.total_epoch/args.train_batch_size) * 5
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= tmax, eta_min=1e-6)
    elif args.lrs == "steplr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    #pretrained
    ckpt_file_path = cfg.MODEL_PATH
    if args.pretrained and os.path.isfile(ckpt_file_path):
        _, best_loss, best_acc = load_checkpoint_files(ckpt_file_path, model, lr_scheduler, logger)    

    #resume
    ckpt_file_path = os.path.join(pth_dir, "checkpoints.pth")
    if args.resume and os.path.isfile(ckpt_file_path):
        start_epoch, best_loss, best_acc = load_checkpoint_files(ckpt_file_path, model, lr_scheduler, logger)

    # BCE & MSE loss
    criterion = CarbonLoss()


    logger.info(torchsummary.summary(model, (3,512,512)))


    best_loss_ckpt_file_path = os.path.join(pth_dir, "best_checkpoints_loss.pth")
    best_acc_c_ckpt_file_path = os.path.join(pth_dir, "best_checkpoints_acc_corr.pth")
    # best_acc_r_ckpt_file_path = os.path.join(pth_dir, "best_checkpoints_acc_r.pth")
    logger.info(">>>>>>>>>> Start training")
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        train_res = train_one_epoch(train_loader, model, optimizer, lr_scheduler, criterion, epoch, num_epochs, logger, rank)

        if rank in [-1, 0]:
            save_checkpoint(model, optimizer, lr_scheduler, epoch, ckpt_file_path, best_loss, best_acc)
            logger.info(f">>>>> {ckpt_file_path} saved......")
            if epoch % args.save_freq==0 or epoch==num_epochs-1:
                save_path = os.path.join(pth_dir, f"checkpoint_%06d.pth"%epoch)
                save_checkpoint(model, optimizer, lr_scheduler, epoch, save_path, best_loss, best_acc)
                logger.info(f">>>>> {save_path} saved......")


        if args.tensorboard:
            current_log = {'train_0_total_loss': train_res[0].avg,
                           'train_1_cls_loss': train_res[1].avg,
                           'train_2_reg_loss': train_res[2].avg,                           
                           'train_3_acc_corr': train_res[3].avg,
                           'train_4_acc_r': train_res[4].avg,
                           'train_5_lr_avg': train_res[5].avg}
            log_tensorboard(writer_tb, current_log, epoch)



        # evaluation
        if (epoch % args.eval_freq == 0) or (epoch == num_epochs - 1):
            val_res = validation_one_epoch(val_loader, model, criterion, epoch, logger, print_freq=100)

            if args.tensorboard:
                current_log = {'validation_0_total_loss': val_res[0].avg,
                               'validation_1_cls_loss': val_res[1].avg,
                               'validation_2_reg_loss': val_res[2].avg,
                               'validation_3_acc_corr': val_res[3].avg,
                               'validation_4_acc_r': val_res[4].avg}
                log_tensorboard(writer_tb, current_log, epoch)

            val_loss = val_res[0].avg
            val_acc_c = val_res[3].avg
            # val_acc_r = val_res[4].avg
            if rank in [-1, 0]:
                if val_loss < best_loss:
                    best_loss = val_loss
                    # save ckpt
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, best_loss_ckpt_file_path, best_loss=best_loss, best_acc=val_acc_c)                    
                    logger.info(f">>>>> best loss {best_loss_ckpt_file_path}_{epoch}_{best_loss} saved......")

                if val_acc_c > best_acc:
                    best_acc = val_acc_c
                    # save ckpt
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, best_acc_c_ckpt_file_path, best_loss=val_loss, best_acc=best_acc)
                    logger.info(f">>>>> best acc {best_acc_c_ckpt_file_path}_{epoch}_{best_acc} saved......")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('ALL Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = TrainParser()
    args = parser.parse_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    seed = args.seed
    if args.local_rank != -1:
      torch.cuda.set_device(args.local_rank)
      torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
      torch.distributed.barrier()
      seed = seed + dist.get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(args, world_size, rank)