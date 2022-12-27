import torch
import torch.nn as nn
from resnet import ResNet
import torch.nn.functional as F
from collections import OrderedDict
import os, json, math
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch.utils.data as data
from torchvision import transforms
from torch import optim 


class PyramidPoolingModule(nn.Module):
    def __init__(self):
        super(PyramidPoolingModule, self).__init__()

        # 1x1スケール
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        # 2×2スケール
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=2)
        self.conv2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        # 3×3スケール
        self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=3)
        self.conv3 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        
        # 6x6スケール
        self.avgpool6 = nn.AdaptiveAvgPool2d(output_size=6)
        self.conv6 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out1 = out1 = self.avgpool1(x)
        out1 = self.conv1(out1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = F.interpolate(out1, (60, 60), mode='bilinear', align_corners=True)
        
        # 2×2スケール
        out2 = self.avgpool2(x)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        out2 = F.interpolate(out2, (60, 60), mode='bilinear', align_corners=True)
        
        # 3×3スケール
        out3 = self.avgpool3(x)
        out3 = self.conv3(out3)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)
        out3 = F.interpolate(out3, (60, 60), mode='bilinear', align_corners=True)
        
        # 6×6スケール
        out6 = self.avgpool6(x)
        out6 = self.conv6(out6)
        out6 = self.bn6(out6)
        out6 = self.relu(out6)
        out6 = F.interpolate(out6, (60, 60), mode='bilinear', align_corners=True)
        
        # 元の入力と各スケールの特徴量を結合させる
        out = torch.cat([x, out1, out2, out3, out6], dim=1)
        
        return out













class PSPNet(nn.Module):
    
    def __init__(self,n_classes):
        super(PSPNet, self).__init__()

        # ResNetから最初の畳み込みフィルタとlayer1からlayer4を取得する
        resnet = ResNet()
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, 
            resnet.conv2, resnet.bn2, resnet.relu, 
            resnet.conv3, resnet.bn3, resnet.relu, 
            resnet.maxpool
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # layer3とlayer4の畳み込みフィルタのパラメータを変更する
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)

            elif 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        self.ppm = PyramidPoolingModule()

        self.cls = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, n_classes, kernel_size=1)
        )

        if self.training is True:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, n_classes, kernel_size=1)
            )
            
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # AuxLossのためにlayer3から出力を抜き出しておく
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        x = self.ppm(x)
        x = self.cls(x)
        x = F.interpolate(x, size=(475,475), mode="bilinear", align_corners=True)
        # 学習時にのみAuxLossモジュールを使用するように設定
        if self.training is True:
            aux = self.aux(x_tmp)
            aux = F.interpolate(aux, size=(475, 475), mode='bilinear', align_corners=True)
            return x, aux
        return x

class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight
    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs[0],targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')
        return loss + self.aux_weight * loss_aux


class Compose(object):
    """引数transformに格納された変形を順番に実行するクラス
       対象画像とアノテーション画像を同時に変換させます。 
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):

        width = img.size[0]  # img.size=[幅][高さ]
        height = img.size[1]  # img.size=[幅][高さ]

        # 拡大倍率をランダムに設定
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)  # img.size=[幅][高さ]
        scaled_h = int(height * scale)  # img.size=[幅][高さ]

        # 画像のリサイズ
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        # アノテーションのリサイズ
        anno_class_img = anno_class_img.resize(
            (scaled_w, scaled_h), Image.NEAREST)

        # 画像を元の大きさに
        # 切り出し位置を求める
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop(
                (left, top, left+width, top+height))

        else:
            # input_sizeよりも短い辺はpaddingする
            p_palette = anno_class_img.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0, 0, 0))
            img.paste(img_original, (pad_width_left, pad_height_top))

            anno_class_img = Image.new(
                anno_class_img.mode, (width, height), (0))
            anno_class_img.paste(anno_class_img_original,
                                 (pad_width_left, pad_height_top))
            anno_class_img.putpalette(p_palette)

        return img, anno_class_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):

        # 回転角度を決める
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # 回転
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


class RandomMirror(object):
    """50%の確率で左右反転させるクラス"""

    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img


class Resize(object):
    """引数input_sizeに大きさを変形するクラス"""

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):

        # width = img.size[0]  # img.size=[幅][高さ]
        # height = img.size[1]  # img.size=[幅][高さ]

        img = img.resize((self.input_size, self.input_size),
                         Image.BICUBIC)
        anno_class_img = anno_class_img.resize(
            (self.input_size, self.input_size), Image.NEAREST)

        return img, anno_class_img


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):

        # PIL画像をTensorに。大きさは最大1に規格化される
        img = transforms.functional.to_tensor(img)

        # 色情報の標準化
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        # アノテーション画像をNumpyに変換
        anno_class_img = np.array(anno_class_img)  # [高さ][幅]

        # 'ambigious'には255が格納されているので、0の背景にしておく
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # アノテーション画像をTensorに
        anno_class_img = torch.from_numpy(anno_class_img)

        return img, anno_class_img


class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  # 画像の拡大
                RandomRotation(angle=[-10, 10]),  # 回転
                RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)

class MyDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # [高さ][幅]

        # 3. 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img

### 関数定義
def make_datapath_list():
    """
    学習用と検証用の画像データとアノテーションデータのファイルパスを格納したリストを取得する
    
    Args:
        rootpath(str): データフォルダへのパス
    
    Returns:
        train_img_list(list): 学習用の画像データへのファイルパス
        train_anno_list(list): 学習用のアノテーションデータへのファイルパス
        valid_img_list(list): 検証用の画像データへのファイルパス
        valid_anno_list(list): 検証用のアノテーションデータへのファイルパス
    """
    """
    # 学習用画像の一覧を取得
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

    """
    with open('./dataset/train/files_RGB.json', 'r') as f:
    # ファイルを読み込む
        files_RGB_train = json.load(f)

    with open('./dataset/test/files_RGB.json', 'r') as f:
    # ファイルを読み込む
        files_RGB_test = json.load(f)
    
    with open('./dataset/train/files_seg.json', 'r') as f:
    # ファイルを読み込む
        files_seg_train = json.load(f)
    
    with open('./dataset/test/files_seg.json', 'r') as f:
    # ファイルを読み込む
        files_seg_test = json.load(f)
    
  

        


    
    
    # 学習用データのリストを作成
    train_img_list = files_RGB_train
    train_anno_list = files_seg_train
    
    # 検証用データのリストを作成
    valid_img_list = files_RGB_test
    valid_anno_list = files_seg_test
    
    return train_img_list, train_anno_list, valid_img_list, valid_anno_list


# 動作確認
if __name__ == '__main__':

    
    ### 1. ファイルパスのリストを作成

    # 学習用と検証用の画像データとアノテーションデータのファイルパスを格納したリストを取得する

    train_img_list, train_anno_list, valid_img_list, valid_anno_list = make_datapath_list()

    # 3. Datasetクラスの作成

    # (RGB)の色の平均値と標準偏差
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    # データセット作成
    train_dataset = MyDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
    input_size=475, color_mean=color_mean, color_std=color_std))

    val_dataset = MyDataset(valid_img_list, valid_anno_list, phase="val", transform=DataTransform(
    input_size=475, color_mean=color_mean, color_std=color_std))    
    


   

    
    ### 4. DataLoaderの作成

    batch_size = 4

    # 学習データのDataLoader
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # 検証データのDataLoader
    valid_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


model = PSPNet(n_classes=150)
state_dict = torch.load("./trained_model/pspnet_base.pth")
model.load_state_dict(state_dict)
n_classes = 12
model.aux[4] = nn.Conv2d(256, n_classes, kernel_size=1)
model.cls[4] = nn.Conv2d(512, n_classes, kernel_size=1)
model.cuda()  # GPU対応

# 交差エントロピー誤差関数
loss_fnc = nn.CrossEntropyLoss()

# 最適化アルゴリズム
optimizer = optim.SGD([
    {'params': model.layer0.parameters(), 'lr': 1e-3},
    {'params': model.layer1.parameters(), 'lr': 1e-3},
    {'params': model.layer2.parameters(), 'lr': 1e-3},
    {'params': model.layer3.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-3},
    {'params': model.ppm.parameters(), 'lr': 1e-3},
    {'params': model.cls.parameters(), 'lr': 1e-2},
    {'params': model.aux.parameters(), 'lr': 1e-2},
], momentum=0.9, weight_decay=0.0001)

def lambda_epoch(epoch):
    max_epoch = 50
    return math.pow((1-(epoch/max_epoch)), 0.9)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)
criterion = PSPLoss(aux_weight=0.4)


# 損失のログ
record_loss_train = []
record_loss_test = []
# 学習
for i in range(50):  # 6エポック学習
    epoch = i+1
    model.train()  # 訓練モード
    scheduler.step()
    optimizer.zero_grad()
    loss_train = 0
    for j, (x, t) in enumerate(train_dataloader):  # ミニバッチ（x, t）を取り出す
        x, t = x.cuda(), t.cuda()  # GPU対応
        y = model(x)
        loss = criterion(y,t.long())
        loss_train += loss.item()
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        if (j+1) % 10 == 0:

            print(f"iters:{j+1}")

        
    loss_train /= j+1
    record_loss_train.append(loss_train)

    model.eval()  # 評価モード
    loss_test = 0
    for j, (x, t) in enumerate(valid_dataloader):  # ミニバッチ（x, t）を取り出す
        x, t = x.cuda(), t.cuda()
        y = model(x)
        loss = F.cross_entropy(y, t.long(), reduction='mean')
        loss_test += loss.item()
    loss_test /= j+1
    record_loss_test.append(loss_test)

    if i%1 == 0:
        print("Epoch:", i+1, "Loss_Train:", loss_train, "Loss_Test:", loss_test)
    if (i+1) % 5 == 0:
        torch.save(model.state_dict(), f'./trained_model/model_{i+1}.pt')



import matplotlib.pyplot as plt

plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()




model.eval()
for x, t in train_dataloader:
    x = x.cuda()
    t = t[0]
    y = model(x)
    y = y.cpu()
    y = y.argmax(1)
    
 
    with open('./annotation.json', 'r') as f:
        # ファイルを読み込む
        annotation = json.load(f)
    palette = []
    palette.append(0)
    palette.append(0)
    palette.append(0)
    for part in annotation:
        palette.append(part['pixel_value']['r'])
        palette.append(part['pixel_value']['g'])
        palette.append(part['pixel_value']['b'])
    img_y = np.array(y[0])
    img_a = Image.new(mode="P",size = (475,475))
    for y in range(475):
        for x in range(475):
            a = int(img_y[y][x])    
            img_a.putpixel((x,y), a) 
    img_a.putpalette(palette)
    img_a.save("sample_y.png")
    img_t_array = np.array(t)
    img_t = Image.fromarray(img_t_array,mode="P")
    img_t.putpalette(palette)
    img_t.save("sample_t.png")
    break



    