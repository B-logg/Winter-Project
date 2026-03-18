import os
import csv
import random
from xml.sax.handler import DTDHandler

os.chdir("/workspace/dataset/city")
# city_or_forest = "forest_"
city_or_forest = "city_"
name = "AP_10_25"
read_train = open("/workspace/dataset/city/"+city_or_forest+name+".txt", "r").readlines()
random.shuffle(read_train)


ratio = 10
index = 0
mode = 0  #"train_val_test"
# mode = 1 #  "train_test"

if mode == 0:
    with open("train_"+city_or_forest+name+".csv", mode="w", newline="") as train_file, open("val_"+city_or_forest+name+".csv", mode="w", newline="") as val_file, open("test_"+city_or_forest+name+".csv", mode="w", newline="") as test_file:
        for line in read_train:        
            image_image = line.replace("\n", "")
            cate=os.path.basename(image_image)[:5]
            image_SGRST_HIGH = image_image.replace("image/"+cate+"/IMAGE/"+cate, "image/"+cate+"/SGRST_HIGH/"+cate+"_SH")        
            label_CRBN_QNTT = image_image.replace("image/"+cate+"/IMAGE/"+cate, "label/"+cate+"/CRBN_QNTT/"+cate+"_CQ")
            label_tif = image_image.replace("image/"+cate+"/IMAGE/"+cate, "label/"+cate+"/GT/"+cate+"_GT")
            data = [image_image, image_SGRST_HIGH, label_CRBN_QNTT, label_tif]
            index = index +1;
            if index%ratio == 0:
                writer = csv.writer(val_file)
                writer.writerow(data)
            elif index%ratio == 1:
                writer = csv.writer(test_file)
                writer.writerow(data)
            else:
                writer = csv.writer(train_file)
                writer.writerow(data)
elif mode == 1:
    with open("train_"+city_or_forest+name+".csv", mode="w", newline="") as train_file, open("test_"+city_or_forest+name+".csv", mode="w", newline="") as test_file:
        for line in read_train:        
            image_image = line.replace("\n", "")
            image_SGRST_HIGH = image_image.replace("image/"+name+"/IMAGE/"+name, "image/"+name+"/SGRST_HIGH/"+name+"_SH")        
            label_CRBN_QNTT = image_image.replace("image/"+name+"/IMAGE/"+name, "label/"+name+"/CRBN_QNTT/"+name+"_CQ")
            label_tif = image_image.replace("image/"+name+"/IMAGE/"+name, "label/"+name+"/GT/"+name+"_GT")
            data = [image_image, image_SGRST_HIGH, label_CRBN_QNTT, label_tif]
            index = index +1;
            if index%ratio == 0:           
                writer = csv.writer(test_file)
                writer.writerow(data)
            else:
                writer = csv.writer(train_file)
                writer.writerow(data)

# forest/image/AP_10/IMAGE/AP_10_37709052_3517.tif
# forest/image/AP_10/SGRST_HIGH/AP_10_SH_37709052_3517.tif
# forest/label/AP_10/CRBN_QNTT/AP_10_CQ_37709052_3517.tif
# forest/label/AP_10/GT/AP_10_CQ_37709052_3517.tif


# with open("train.csv", mode="w", newline="") as train_file:
#     for line in read_train:        
#         image_file = line.split("/")[-1].replace("\n", "")
#         text_file = image_file.replace(".jpg", ".txt")
#         data = [image_file, text_file]
#         writer = csv.writer(train_file)
#         writer.writerow(data)

# read_train = open("test.txt", "r").readlines()

# with open("test.csv", mode="w", newline="") as train_file:
#     for line in read_train:
#         image_file = line.split("/")[-1].replace("\n", "")
#         text_file = image_file.replace(".jpg", ".txt")
#         data = [image_file, text_file]
#         writer = csv.writer(train_file)
#         writer.writerow(data)
