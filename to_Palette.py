from PIL import Image
import numpy as np
import json, os



with open('./annotation.json', 'r') as f:
    # ファイルを読み込む
    annotation = json.load(f)
data = {}
for part in annotation:
    data[part['label_name']] = []
palette = []
palette.append(0)
palette.append(0)
palette.append(0)
for part in annotation:
    palette.append(part['pixel_value']['r'])
    palette.append(part['pixel_value']['g'])
    palette.append(part['pixel_value']['b'])


with open('./dataset/train/files_seg.json', 'r') as f:
    # ファイルを読み込む
    seg_images = json.load(f)



for file in seg_images:
    # 画像ファイルを読み込む
    img = Image.open(file)
    # img_rgb = Image.open("./RGB1294c86b-eb4d-46fd-ab5e-be9388ed2be4/rgb_50.png")
    data = {}
    for part in annotation:
        data[part['label_name']] = []


    w, h  = img.size

    for y in range(h):
        for x in range(w):
            p = img.getpixel((x,y))
            for color_list in annotation:
                if p[0] == color_list['pixel_value']['r'] and p[1] == color_list['pixel_value']['g'] and p[2] == color_list['pixel_value']['b'] and p[3] == color_list['pixel_value']['a']:
                    data[color_list['label_name']].append((x,y))
    """
    for color in annotation:
        img_out = Image.new("RGBA",img.size)
        for p in data[color['label_name']]:
            p_rgb = img_rgb.getpixel(p)
            img_out.putpixel(p,p_rgb)
        img_out.save(f"sample/sample_{color['label_name']}.png")
    """

    img_correct = Image.new("P",img.size)
    img_correct.putpalette(palette)
    for i, color in enumerate(annotation):
        for p in data[color['label_name']]:
            img_correct.putpixel(p,i+1)
    img_correct.save(file)

with open('./dataset/test/files_seg.json', 'r') as f:
    # ファイルを読み込む
    seg_images = json.load(f)



for file in seg_images:
    # 画像ファイルを読み込む
    img = Image.open(file)
    # img_rgb = Image.open("./RGB1294c86b-eb4d-46fd-ab5e-be9388ed2be4/rgb_50.png")
    data = {}
    for part in annotation:
        data[part['label_name']] = []


    w, h  = img.size

    for y in range(h):
        for x in range(w):
            p = img.getpixel((x,y))
            for color_list in annotation:
                if p[0] == color_list['pixel_value']['r'] and p[1] == color_list['pixel_value']['g'] and p[2] == color_list['pixel_value']['b'] and p[3] == color_list['pixel_value']['a']:
                    data[color_list['label_name']].append((x,y))
    """
    for color in annotation:
        img_out = Image.new("RGBA",img.size)
        for p in data[color['label_name']]:
            p_rgb = img_rgb.getpixel(p)
            img_out.putpixel(p,p_rgb)
        img_out.save(f"sample/sample_{color['label_name']}.png")
    """

    img_correct = Image.new("P",img.size)
    img_correct.putpalette(palette)
    for i, color in enumerate(annotation):
        for p in data[color['label_name']]:
            img_correct.putpixel(p,i+1)
    img_correct.save(file)
