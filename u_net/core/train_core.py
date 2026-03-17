import time
from timm.utils import AverageMeter
import torch
import datetime
import math
from tqdm import tqdm

def train_one_epoch(data_loader, model, optimizer, lr_scheduler, criterion, epoch, num_epochs, logger, rank, print_freq=100):

    if rank != -1:
        data_loader.sampler.set_epoch(epoch)

    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()    
    acc_c_meter = AverageMeter()    
    acc_r_meter = AverageMeter()    
    lr_meter = AverageMeter()

    num_steps = len(data_loader)

    start = time.time()
    end = time.time()
    pbar = enumerate(data_loader)
    if rank in [-1, 0]:
        pbar = tqdm(pbar, total=num_steps)  # progress bar
        
    for idx, batch in pbar:
        image = batch["image"].cuda(non_blocking=True)
        label_cls = batch["label_cls"].cuda(non_blocking=True)
        label_reg = batch["label_reg"].cuda(non_blocking=True)
     
        optimizer.zero_grad()

        outputs = model(image)

        loss, cls_loss, reg_loss, acc_c, acc_r = criterion(outputs[0],outputs[1], label_cls, label_reg)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))


        loss.backward()
        # dice_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        #
        loss_meter.update(loss.item(), image.size(0))
        cls_loss_meter.update(cls_loss.item(), image.size(0))
        reg_loss_meter.update(reg_loss.item(), image.size(0))        
        acc_c_meter.update(acc_c, image.size(0))
        acc_r_meter.update(acc_r, image.size(0))
        lr_meter.update(optimizer.param_groups[0]["lr"])
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            logger.info(
                f'Train: [{epoch}/{num_epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'Time {batch_time.val:.8f} ({batch_time.avg:.8f})\t'
                f'loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'  
                f'cls_loss {cls_loss_meter.val:.8f} ({cls_loss_meter.avg:.8f})\t' 
                f'reg_loss {reg_loss_meter.val:.8f} ({reg_loss_meter.avg:.8f})\t' 
                f'train_acc_corr {acc_c_meter.val:.8f} ({acc_c_meter.avg:.8f})\t' 
                f'train_acc_r {acc_r_meter.val:.8f} ({acc_r_meter.avg:.8f})\t' 
                f'grad_norm(lr) {lr_meter.val:.8f} ({lr_meter.avg:.8f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return [loss_meter, cls_loss_meter, reg_loss_meter, acc_c_meter, acc_r_meter, lr_meter]