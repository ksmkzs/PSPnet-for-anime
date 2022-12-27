import os, json
from PIL import Image
charas = os.listdir("./dataset/train")
files_seg_train = []
files_RGB_train = []
for chara in charas:
    seg_or_rgbs = os.listdir(f"./dataset/train/{chara}")
    for seg_or_rgb in seg_or_rgbs:
        if seg_or_rgb[0] == "R":
            rgb_imgs = os.listdir(f"./dataset/train/{chara}/{seg_or_rgb}")
            for rgb_img in rgb_imgs:
                files_RGB_train.append(f"./dataset/train/{chara}/{seg_or_rgb}/{rgb_img}")
        if seg_or_rgb[0] == "S":
            seg_imgs = os.listdir(f"./dataset/train/{chara}/{seg_or_rgb}")
            for seg_img in seg_imgs:
                files_seg_train.append(f"./dataset/train/{chara}/{seg_or_rgb}/{seg_img}")
files_seg_train = sorted(files_seg_train)
files_RGB_train = sorted(files_RGB_train)
seg_data_train = json.dumps(files_seg_train)
RGB_data_train = json.dumps(files_RGB_train)

    # JSON形式の文字列を保存
with open('./dataset/train/files_seg.json', 'w') as f:
    f.write(seg_data_train)
with open('./dataset/train/files_RGB.json', 'w') as f:
    f.write(RGB_data_train)


charas = os.listdir("./dataset/test")
files_seg_test = []
files_RGB_test = []
for chara in charas:
    seg_or_rgbs = os.listdir(f"./dataset/test/{chara}")
    for seg_or_rgb in seg_or_rgbs:
        if seg_or_rgb[0] == "R":
            rgb_imgs = os.listdir(f"./dataset/test/{chara}/{seg_or_rgb}")
            for rgb_img in rgb_imgs:
                files_RGB_test.append(f"./dataset/test/{chara}/{seg_or_rgb}/{rgb_img}")
        if seg_or_rgb[0] == "S":
            seg_imgs = os.listdir(f"./dataset/test/{chara}/{seg_or_rgb}")
            for seg_img in seg_imgs:
                files_seg_test.append(f"./dataset/test/{chara}/{seg_or_rgb}/{seg_img}")
files_seg_test = sorted(files_seg_test)
files_RGB_test = sorted(files_RGB_test)
seg_data_test = json.dumps(files_seg_test)
RGB_data_test = json.dumps(files_RGB_test)

    # JSON形式の文字列を保存
with open('./dataset/test/files_seg.json', 'w') as f:
    f.write(seg_data_test)
with open('./dataset/test/files_RGB.json', 'w') as f:
    f.write(RGB_data_test)