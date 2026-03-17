import os
import torch

def is_parallel(model):
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)

def load_checkpoint_files(ckpt_path, model, lr_scheduler, logger):
    print(f">>>>>>>>>>>>> Resuming training from checkpoint: {ckpt_path}")
    if logger:
        logger.info(f">>>>>>>>>>>>> Resuming training from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["model"]
    if not is_parallel(model):
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.','')
            new_state_dict[new_key] = state_dict[key]
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
      
    start_epoch = ckpt["epoch"] + 1
    if lr_scheduler:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    best_loss = ckpt["best_loss"]
    best_acc = ckpt["best_acc"]

    del ckpt
    torch.cuda.empty_cache()

    return start_epoch, best_loss, best_acc

def save_checkpoint(model, optimizer, lr_scheduler, epoch, save_path,  best_loss=999, best_acc=-999):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'best_loss': best_loss,
                  'best_acc': best_acc}

    torch.save(save_state, save_path)


def make_directory(path):
    os.makedirs(path, exist_ok=True)

def make_output_directory(args, cfg):
    make_directory(cfg.OUTPUT_DIR)

    # create log file
    log_dir = os.path.join(cfg.OUTPUT_DIR, "logs")
    make_directory(log_dir)

    # create tensorboard
    tensorb_dir = None
    if args.tensorboard:
        tensorb_dir = os.path.join(cfg.OUTPUT_DIR, "tensorb")
        make_directory(tensorb_dir)

    #ckpt
    pth_dir = os.path.join(cfg.OUTPUT_DIR, "pths")
    make_directory(pth_dir)

    return log_dir, pth_dir, tensorb_dir