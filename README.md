## 1. 环境说明
运行以下代码自动安装所需库
```
pip requirements.txt
```
若运行时，库从github上自动下载模型失败，则需根据报错信息与链接手动操作,放在库中的相应位置


## 2. 数据集构建

### 2.1 初始数据集构建
数据集选择从[ ImageNet 网站](https://image-net.org/)下载，下载链接为[ILSVRC2012_img_val.tar](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)，并放在正确位置，采用python构建初始数据集，运行：
```
py data_set.py
```


### 2.2 DCT系数构建(C++)
之后会得到所需数量的256*256图片集和对应的yuv420格式文件，此时构建的数据集的UV通道均不为全0。接着将yuv文件地址填写到H.265项目根目录下main.cpp文件中，相应参数如下：
```
input_yuv            # yuv文件输入地址（需要适配）
TOTAL_ITERATIONS     # 循环编码的yuv文件数量（需要适配）
frame_dir            # 帧号配置文件地址
out_pic_dir          # H265编码图片输出地址
bppout               # bpp输出配置，布尔型变量，true时输出每帧bpp与平均bpp
```
此外还需要将H.265项目下\source\encoder\encoder.h中的相应参数进行改动，如下：
```
outputdatas         # 用于判断是否提取编码端DCT系数，布尔型变量，true时执行提取
frame_dir           # 帧号文件路径，与上述main.cpp文件中路径相同
coeff_dir           # DCT系数存储地址
tu_dir              # TU规则的存储地址
```
之后调试运行H.265项目代码（过信道？）（删除Y通道系数？），得到DCT系数数据集，与H.265编码图片


### 2.3 Y通道像素数据集构建
将H.265编码图片地址填入y_txt.py文件中，设置Y通道系数输出地址，并运行该文件，得到H.265编码后的Y通道像素数据集。


## 3. 模型训练与测试


## 4. 模型重建数据集导入


### 4.1 UV重建数据后处理
更改res_uvdata.py中相应参数进行更改，并运行。
```
sum_poc             # 总帧数
coeff_path          # DCT系数的存储地址
tu_path             # TU规则的存储地址
output_dir          # 重建UV系数主目录
```


### 4.2 UV通道数据导入H.265（C++）
首先将项目DCT系数提取功能关闭（将encoder.h中的outputdatas变量设置false），接着定位到项目根目录下\end\libde265\slice.cc文件中的residual_coding函数中，更改相应参数：
```
inputdatas          # 判断是否导入外部DCT系数，true时导入后处理数据
inputcoeff_dir      # DCT系数后处理数据的地址
frame_dir           # 帧号文件路径（路径同上）
```
更改main.cpp中的编码输出图片地址（out_pic_dir），此时是NTC混合方案的重建结果，运行H.265项目，得到UV数据重建图像。


### 4.3 Y通道重建数据导入
更改y_input.py中的参数，如下，并运行该文件。
```
sun_poc             # 总帧数
ydata_dir           # y通道重建数据地址
uvimg_dir           # uv通道重建图片地址
out_pic             # 方案最终重建结果图
```


## 5. 实验结果评估
更改eval.py中的参数,如下，并运行该文件。
```
final_nums              # 总共测试帧
BATCH_SIZE              # 每批评价计算的样本数
original_dir            # 原始图像地址
NTC_dir                 # 重建图像目录
H265_dir                # H265重建图像目录
out_eval                # 实验结果存储地址
index_H265              # 判断H265PSNR是否输出，为1时输出
index_NTC               # 判断NTCPSNR是否输出，为1时输出
```