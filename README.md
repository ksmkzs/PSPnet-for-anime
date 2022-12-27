# 概要
PSPnetを11のクラス分類にファインチューニングしたもの
**./dataset**ディレクトリに**train**、**test**が存在し、その中にデータセットとデータのリストをJSON形式で保存していることを前提としている。リストの作成方法は**listmaker.py**を参照
### Requirement
・Pytorch  
・CUDA  
・RGBデータセットはRGB、segmentationデータセットはpaletteモードである必要がある  
・RGBA画像のモード変更方法は**to_RGB.py**、**to_Palette.py**を参照

### 更新情報
2022/12/26 モデルの完成　過学習が起きたためデータセットを再用意中