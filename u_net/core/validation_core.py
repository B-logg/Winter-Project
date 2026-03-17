import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time
from unittest import result
from timm.utils import AverageMeter
import torch
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from Map_dataset import MapDataset
from evaluate import corr, r_square


IMAGE_SIZE = 512

def validation_one_epoch(data_loader, model, criterion, epoch, logger, print_freq=100):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    acc_c_meter = AverageMeter()
    acc_r_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):

            # image = batch["image"].cuda(non_blocking=True)
            # mask = batch["mask"].cuda(non_blocking=True)
            image = batch["image"].cuda(non_blocking=True)
            label_cls = batch["label_cls"].cuda(non_blocking=True)
            label_reg = batch["label_reg"].cuda(non_blocking=True)

            outputs = model(image)

            # loss, bce_loss, dice_loss, dice_coef_mean = criterion(outputs, mask)
            loss, cls_loss, reg_loss, acc_c, acc_r = criterion(outputs[0],outputs[1], label_cls, label_reg)
            # Image.fromarray( (np.clip((input_reg[3,...].squeeze().detach().cpu().numpy()*255), 0,255)).astype(np.uint8)).save("/workspace/src/carbon/debug3.png")  #for debug

            batch_time.update(time.time() - end)
            loss_meter.update(loss.item(), image.size(0))
            cls_loss_meter.update(cls_loss.item(), image.size(0))
            reg_loss_meter.update(reg_loss.item(), image.size(0))
            acc_c_meter.update(acc_c, image.size(0))
            acc_r_meter.update(acc_r, image.size(0))

            if idx % print_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Evaluate: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t' 
                    f'cls_loss {cls_loss_meter.val:.8f} ({cls_loss_meter.avg:.8f})\t' 
                    f'reg_loss {reg_loss_meter.val:.8f} ({reg_loss_meter.avg:.8f})\t'   
                    f'eval_acc_corr {acc_c_meter.val:.8f} ({acc_c_meter.avg:.8f})\t'                   
                    f'eval_acc_r {acc_r_meter.val:.8f} ({acc_r_meter.avg:.8f})\t'                   
                    f'Mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} validataion takes {datetime.timedelta(seconds=int(epoch_time))}")

    return [loss_meter, cls_loss_meter, reg_loss_meter, acc_c_meter, acc_r_meter]

def test_one_epoch(data_loader, model, criterion, logger, output_dir):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    acc_c_meter = AverageMeter()
    acc_r_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    corr_avg = 0
    r_avg = 0
    index = 0

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):

            image = batch["image"].cuda(non_blocking=True)
            label_cls = batch["label_cls"].cuda(non_blocking=True)
            label_reg = batch["label_reg"].cuda(non_blocking=True)

            # i_i = batch["i_i_path"]
            # i_s = batch["i_s_path"]
            # l_c = batch["l_c_path"]
            # l_t = batch["l_t_path"]

            i_i = Image.open(batch["i_i_path"][0])
            i_s = Image.open(batch["i_s_path"][0])
            i_s = Image.fromarray(np.clip((np.array(i_s)*255/30).astype(np.uint8),0,255)).convert("RGB")
            l_c = Image.open(batch["l_c_path"][0])
            l_c = Image.fromarray(np.clip((np.array(l_c)*255/3000).astype(np.uint8),0,255)).convert("RGB")
            l_t = Image.open(batch["l_t_path"][0])

            outputs = model(image)


            image_margin = 5
            result_img = Image.new('RGB', (IMAGE_SIZE*4+image_margin*3, IMAGE_SIZE*2+image_margin))

            pred = F.softmax(outputs[0],dim=1)
            pred_max = pred.max(1)[1].cpu().numpy()[0]
            out_tif = Image.fromarray(MapDataset.decode_target_L(pred_max).astype('uint8'))
            
            out_c = (outputs[1].cpu().squeeze()).numpy()
            corr_val = corr(np.array(out_c), np.array(label_reg.cpu().detach()))
            r_square_val = r_square(np.array(out_c), np.array(label_reg.cpu().detach()))
            
            out_c = Image.fromarray(   np.clip((outputs[1].cpu().squeeze()*255).numpy(),0,255).astype('uint8')       )

            
            
            result_img.paste(i_i,(0,0))
            result_img.paste(i_s,(IMAGE_SIZE+image_margin,0))
            result_img.paste(l_t,(IMAGE_SIZE*2+image_margin*2,0))
            result_img.paste(l_c,(IMAGE_SIZE*3+image_margin*3,0))

            result_img.paste(out_tif,(IMAGE_SIZE*2+image_margin*2,IMAGE_SIZE+image_margin))
            result_img.paste(out_c,(IMAGE_SIZE*3+image_margin*3,IMAGE_SIZE+image_margin))

            result_img_draw = ImageDraw.Draw(result_img)
            font = ImageFont.truetype("ARIAL.TTF", 30)
            result_img_draw.text((IMAGE_SIZE/2,IMAGE_SIZE*3/2),    "correlation : " + "{0:.3f}".format(corr_val), (0,255,0), font=font)
            result_img_draw.text((IMAGE_SIZE/2,IMAGE_SIZE*3/2+35), " R_squre   : " + "{0:.3f}".format(r_square_val), (0,255,0), font=font)


            img_name = os.path.basename(batch["i_i_path"][0]).split('.')[0]
            result_img.save(os.path.join(output_dir, img_name+'_rst.png'))

            corr_avg = corr_avg+corr_val
            r_avg = r_avg+r_square_val
            index = index +1






            # # loss, bce_loss, dice_loss, dice_coef_mean = criterion(outputs, mask)
            # loss, cls_loss, reg_loss, acc_c, acc_r = criterion(outputs[0],outputs[1], label_cls, label_reg)
            # # Image.fromarray( (np.clip((input_reg[3,...].squeeze().detach().cpu().numpy()*255), 0,255)).astype(np.uint8)).save("/workspace/src/carbon/debug3.png")  #for debug
            # #Image.fromarray( (np.clip((outputs[1][0,...].squeeze().detach().cpu().numpy()*255), 0,255)).astype(np.uint8)).save("/workspace/src/carbon/debug3.png")
            
            # batch_time.update(time.time() - end)
            # loss_meter.update(loss.item(), image.size(0))
            # cls_loss_meter.update(cls_loss.item(), image.size(0))
            # reg_loss_meter.update(reg_loss.item(), image.size(0))
            # acc_c_meter.update(acc_c, image.size(0))
            # acc_r_meter.update(acc_r, image.size(0))

            
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            # logger.info(
            #     f'Evaluate: [{idx}/{len(data_loader)}]\t'
            #     f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #     f'loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t' 
            #     f'cls_loss {cls_loss_meter.val:.8f} ({cls_loss_meter.avg:.8f})\t' 
            #     f'reg_loss {reg_loss_meter.val:.8f} ({reg_loss_meter.avg:.8f})\t'   
            #     f'eval_acc_corr {acc_c_meter.val:.8f} ({acc_c_meter.avg:.8f})\t'                   
            #     f'eval_acc_r {acc_r_meter.val:.8f} ({acc_r_meter.avg:.8f})\t'                   
            #     f'Mem {memory_used:.0f}MB')
    
    corr_avg = corr_avg/index
    r_avg = r_avg/index

    epoch_time = time.time() - start
    logger.info(f"test takes {datetime.timedelta(seconds=int(epoch_time))}")

    return [loss_meter, cls_loss_meter, reg_loss_meter, acc_c_meter, acc_r_meter]