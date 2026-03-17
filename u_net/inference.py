from cmath import nan
import os
from datetime import datetime

import numpy as np
import csv 
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

from config.param_parser import InferenceParser
from Map_dataset import MapDataset
from evaluate import corr_wCla, r_square_wCla
from utils.utils import make_directory
from model.build import build_model
from data.carbon_dataset import CarbonDataset, Image

from config_mf import CONFIGURE

def main(args):
    cfg = CONFIGURE(args.image_type)
    make_directory(cfg.RESULT_OUT_DIR)
    
    model = build_model(args.net, num_class=cfg.NUM_CLASSES) 

    #device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('On which device we are on : {}'.format(device))
    model.to(device)    # move the model to GPU
    
    ckpt_file_path = cfg.MODEL_PATH
    print("loading model...")
    state_dict = torch.load(ckpt_file_path, map_location="cpu")
    new_state_dict = {}
    for key in state_dict['model']:
        new_key = key.replace('module.','')
        new_state_dict[new_key] = state_dict['model'][key]
    model.load_state_dict(new_state_dict)
    print(f" => loaded checkpoint {ckpt_file_path}")

    del new_state_dict
    torch.cuda.empty_cache()
    model.eval()

    print(f"권역: {args.image_type}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START")
    corr_sum=0.0
    r_sum=0.0
    cnt = 0
    
    with open(os.path.join("datalists", cfg.TEST_CSV), 'r') as f, open(cfg.OUTPUT_DIR+"results.txt", 'w') as d:
        reader = csv.reader(f, delimiter=',')
        for i, line in enumerate(reader):
            img_path = os.path.join(cfg.DATA_PATH, line[0])
            img_SGRST_HIGH_path = os.path.join(cfg.DATA_PATH, line[1])
            label_CRBN_QNTT_path = os.path.join(cfg.DATA_PATH, line[2])
            label_tif_path = os.path.join(cfg.DATA_PATH, line[3])

            img_name = os.path.basename(img_path).split('.')[0]

            i_i = Image.open(img_path)
            i_s = Image.open(img_SGRST_HIGH_path)
            l_c = Image.open(label_CRBN_QNTT_path)
            l_t = Image.open(label_tif_path)
            
            img, label_cls, label_reg_norm = CarbonDataset.normalize_concat(i_i, i_s, l_c, l_t, cfg)
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(img)

            pred = F.softmax(out[0],dim=1)
            pred_max = pred.max(1)[1].cpu().numpy()[0]
            out_tif = Image.fromarray(MapDataset.decode_target_L(pred_max, cfg.visible_mapping).astype('uint8'))

            out_c = (out[1].cpu().squeeze()).numpy()
            out_c_tif = Image.fromarray(CarbonDataset.denormalize(out_c, cfg))
            
            pred = np.array(CarbonDataset.denormalize(out_c, cfg))
            test = np.array(l_c)
            corr_val = corr_wCla(pred, test, np.array(label_cls.cpu().detach()))
            r_square_val = r_square_wCla(pred, test, np.array(label_cls.cpu().detach()))
            
            l_c_flatten = np.array(l_c).flatten()
            
            ##################################################
            cnt_exclude = False #탄소량 평균이 1 이하면 계산제외
            if np.average(l_c_flatten) < 1:
                cnt_exclude = True           
            ##################################################
            
            i_s = Image.fromarray(np.clip((np.array(i_s)*255/cfg.SGRST_CLIPPING).astype(np.uint8),0,255)).convert("RGB")
            l_c = Image.fromarray(np.clip(((np.array(l_c))*255/cfg.CARBON_CLIPPING[1]).astype(np.uint8),0,255)).convert("RGB")            
            out_c = Image.fromarray(np.clip((out[1].cpu().squeeze()*255).numpy(),0,255).astype('uint8'))

            image_margin = 5
            result_img = Image.new('RGB', (cfg.IMAGE_SIZE*4+image_margin*3, cfg.IMAGE_SIZE*2+image_margin))

            result_img.paste(i_i,(0,0))
            result_img.paste(i_s,(cfg.IMAGE_SIZE+image_margin,0))
            result_img.paste(l_t,(cfg.IMAGE_SIZE*2+image_margin*2,0))
            result_img.paste(l_c,(cfg.IMAGE_SIZE*3+image_margin*3,0))

            result_img.paste(out_tif,(cfg.IMAGE_SIZE*2+image_margin*2,cfg.IMAGE_SIZE+image_margin))
            result_img.paste(out_c,(cfg.IMAGE_SIZE*3+image_margin*3,cfg.IMAGE_SIZE+image_margin))

            result_img_draw = ImageDraw.Draw(result_img)
            font = ImageFont.truetype("ARIAL.TTF", 30)
            if np.isnan(corr_val) or np.isnan(r_square_val) or cnt_exclude:
                result_img_draw.text((cfg.IMAGE_SIZE/2,cfg.IMAGE_SIZE*3/2),    "carbon All NaN", (0,255,0), font=font)
            else:
                result_img_draw.text((cfg.IMAGE_SIZE/2,cfg.IMAGE_SIZE*3/2),    "correlation : " + "{0:.3f}".format(corr_val), (0,255,0), font=font)
                result_img_draw.text((cfg.IMAGE_SIZE/2,cfg.IMAGE_SIZE*3/2+35), " R_squred   : " + "{0:.3f}".format(r_square_val), (0,255,0), font=font)
                if corr_sum == nan: corr_sum = 0
                corr_sum = corr_sum + corr_val
                r_sum = r_sum + r_square_val
                cnt = cnt + 1
            
            vis_output_name = os.path.join(cfg.RESULT_OUT_DIR, img_name+'_vis.png')
            result_img.save(vis_output_name)
            out_tif.save(os.path.join(cfg.RESULT_OUT_DIR, img_name+'_cls.tif'))
            out_c_tif.save(os.path.join(cfg.RESULT_OUT_DIR, img_name+'_cq.tif'), compression='tiff_lzw')
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            d.write('[{}] {:4d}. {}:\tcorr:{:.3f}, R-squred:{:.3f},   {} saved\n'.format(timestamp, i, os.path.basename(img_path), corr_val, r_square_val, vis_output_name))
            print('[{}] {:4d}. {}:\tcorr:{:.3f}, R-squred:{:.3f},   {} saved'.format(timestamp, i, os.path.basename(img_path), corr_val, r_square_val, vis_output_name))
            
        
        corr_avg = corr_sum/cnt
        r_avg = r_sum/cnt
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] END")
        d.write('Overall correlation:{:.3f}, R-squred:{:.3f}'.format(corr_avg, r_avg))
        print('Overall correlation:{:.3f}, R-squred:{:.3f}'.format(corr_avg, r_avg))
        print('Result file: {}'.format(cfg.OUTPUT_DIR + "results.txt"))
    
if __name__ == '__main__':
    parser = InferenceParser()
    args = parser.parse_args()
    main(args)
    