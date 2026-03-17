BASE_DIR = "/workspace"
SRC_DIR = BASE_DIR+"/src"

CARBON_CLIPPING_DICT = {'city_AP_10_l': 40, 'city_AP_10_h': 640,          # 4000
                 'city_AP_25_l':200, 'city_AP_25_h':4000,                 # 4000
                 'city_AP_10_25_l':40, 'city_AP_10_25_h':4000,            # 8000
                 'city_NIR_10_l':50, 'city_NIR_10_h':640,                 # 4000
                 'forest_AP_10_l':0, 'forest_AP_10_h':930,                # 16000
                 'forest_AP_10_WIN_l':0, 'forest_AP_10_WIN_h':900,        # 5000
                 'forest_AP_25_l':40, 'forest_AP_25_h':5500,              # 16000
                 'forest_AP_10_25_l':0, 'forest_AP_10_25_h':5500,         # 32000
                 'forest_NIR_10_l':0, 'forest_NIR_10_h':930,              # 16000
                 'forest_SN_10_l':900, 'forest_SN_10_h':120000}           # 1000


MAX_EPOCHS = 30000000
BATCH_SIZE = 8
LR = 1e-4

class CONFIGURE:
  def __init__(self, image_type = "forest_SN_10"):
    if image_type == "forest_AP_10" or image_type == "forest_AP_10_WIN" or \
    image_type == "forest_AP_25" or image_type == "forest_AP_10_25" or image_type == "forest_NIR_10": 
      TARGET_DATA_TYPE = "Forest_ap_nir"
      self.IMAGE_SIZE = 512
    elif image_type == "forest_SN_10":
      self.IMAGE_SIZE = 256
      TARGET_DATA_TYPE = "forest_sn"
    elif image_type == "city_AP_10" or image_type == "city_AP_25" or image_type == "city_AP_10_25" or image_type == "city_NIR_10":
      self.IMAGE_SIZE = 512
      TARGET_DATA_TYPE = "city_ap_nir"

    
    self.DATA_PATH = ""
    self.OUTPUT_DIR = SRC_DIR+"/outputs/"+image_type+"/"
    self.RESULT_OUT_DIR = SRC_DIR+"/outputs/"+image_type+"/results"
    self.MODEL_PATH = SRC_DIR+"/weights/"+image_type+"/best_checkpoints_acc_corr.pth"
    self.TRAIN_CSV= "train_"+image_type+".csv"
    self.VAL_CSV= "val_"+image_type+".csv"
    self.TEST_CSV = "test_"+image_type+".csv"

    self.SGRST_CLIPPING = 30.0 #임분고 클리핑
    self.CARBON_CLIPPING = [CARBON_CLIPPING_DICT[image_type+"_l"], CARBON_CLIPPING_DICT[image_type+"_h"],CARBON_CLIPPING_DICT[image_type+"_h"]-CARBON_CLIPPING_DICT[image_type+"_l"] ]  #탄소량 클리핑
    ignore_label = 255

    if TARGET_DATA_TYPE == "city_ap_nir":
      self.label_mapping = {-1: ignore_label, 0: ignore_label,
                          110: 1, 
                          120: 2, 
                          130: 3, 
                          140: 4,
                          210: 5,
                          220: 6,
                          230: 7,
                          190: 0, 
                          255: ignore_label}
      
      self.visible_mapping = {0: 190,
        1: 110,
        2: 120, 
        3: 130,
        4: 140,
        5: 210,
        6: 220,
        7: 230,
        ignore_label : 0}

      self.NUM_CLASSES = 8
      self.NUM_CHANNELS = 3  
      
    elif TARGET_DATA_TYPE == "city_sn":
      self.label_mapping = {-1: ignore_label, 0: ignore_label,                      
                      140: 1,
                      150: 2,
                      190: 0, 
                      255: ignore_label}
      
      self.visible_mapping = {0: 190,
        1: 140,
        2: 150,     
        ignore_label : 0}

      self.NUM_CLASSES = 3
      self.NUM_CHANNELS = 3
      
    elif TARGET_DATA_TYPE == "Forest_ap_nir":
      self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              110: 1, 
                              120: 2, 
                              130: 3, 
                              140: 4,
                              190: 0, 255: ignore_label}
      
      self.visible_mapping = {0: 190,
        1: 110,
        2: 120, # GS, JL  
        3: 130,
        4: 140,
        ignore_label : 0}

      self.NUM_CLASSES = 5
      self.NUM_CHANNELS = 3

    elif TARGET_DATA_TYPE == "forest_sn":
      self.label_mapping = {-1: ignore_label, 0: ignore_label,                      
                  140: 1,
                  150: 2,
                  190: 0, 
                  255: ignore_label}
      
      self.visible_mapping = {0: 190,
        1: 140,
        2: 150,     
        ignore_label : 0}

      self.NUM_CLASSES = 3
      self.NUM_CHANNELS = 3


    else:
      print("TARGET_DATA_TYPE is wrong !!!!", TARGET_DATA_TYPE)      
