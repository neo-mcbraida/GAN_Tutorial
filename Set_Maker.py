import matplotlib
# from matplotlib import pyplot
from matplotlib.pyplot import *
import os
import numpy as np
from PIL import Image

height = 500
width = 500
channels = 3

current_image = 0

p1 = "G:/AI DataSets/wikiart_dataset/wikiart/Art_Nouveau_Modern"
p2 = "G:/AI DataSets/wikiart_dataset/wikiart/Baroque"
p3 = "G:/AI DataSets/wikiart_dataset/wikiart/Realism"
p4 = "G:/AI DataSets/wikiart_dataset/wikiart/New_Realism"

paths = [p3, p4]

save_dir = "G:/AI DataSets/wikiart_dataset/wikiart/500x500numpy1"

images = []

def get_data(direct, filename):
    global images
    try:
        path = direct + "/" + filename
        #rgb_im = imread(path)
        img = Image.open(path)
        #print(dim.shape)
        rgb_im = img.resize((width, height), Image.ANTIALIAS)
        rgb_im = np.array(rgb_im)
        #print(rgb_im.shape)
        images.append(rgb_im)
        img.close()

    except:
        print("not valid image")

def run():
    current_image = 0
    for direct in paths:
        for filename in os.listdir(direct):
            current_image += 1
            print("image number ", current_image)
            get_data(direct, filename)

def create_image(image):
    save_path = "C:/Users/Neo/Documents/gitrepos/GAN_Tutorial/testim.png"
    img = img = np.ascontiguousarray(image)
    img = Image.fromarray(img, 'RGB')
    img.save(save_path)

#get_data("Abstract_image_5.jpg")
run()
images = np.array(images)
#create_image(images[0])

with open(save_dir + "/" + "train_set2.npy", 'wb') as fp:
    np.save(fp, images)
print("saved")