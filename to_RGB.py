from PIL import Image
import numpy as np
import json, os



with open('./dataset/train/files_RGB.json', 'r') as f:
    # ファイルを読み込む
    RGB_images = json.load(f)




for file in RGB_images:
    # 画像ファイルを読み込む
    img = Image.open(file)
    # img_rgb = Image.open("./RGB1294c86b-eb4d-46fd-ab5e-be9388ed2be4/rgb_50.png")
    img_out = Image.new("RGB",img.size)
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r,g,b,a = img.getpixel((x,y))
            img_out.putpixel((x,y),(r,g,b))
    
    img_out.save(file)

with open('./dataset/test/files_RGB.json', 'r') as f:
    # ファイルを読み込む
    RGB_images = json.load(f)




for file in RGB_images:
    # 画像ファイルを読み込む
    img = Image.open(file)
    # img_rgb = Image.open("./RGB1294c86b-eb4d-46fd-ab5e-be9388ed2be4/rgb_50.png")
    img_out = Image.new("RGB",img.size)
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r,g,b,a = img.getpixel((x,y))
            img_out.putpixel((x,y),(r,g,b))
    
    img_out.save(file)
