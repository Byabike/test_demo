## 1. 环境说明
按照requirements.txt文件安装所需库



## 2. 数据集构建

### 2.1 初始数据集构建
数据集选择从[ ImageNet 网站](https://image-net.org/)下载，下载链接为[ILSVRC2012_img_val.tar](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)，并放在data_set.py代码所设置的位置，采用python构建初始数据集，运行：
```
python data_set.py
```
会得到所需数量的256*256图片集和对应的yuv420格式文件。


### 2.2 DCT系数构建(C++)
接着将yuv文件地址填写到H.265项目根目录下main.cpp文件中，相应参数如下：
```
TOTAL_ITERATIONS     # 循环编码的yuv文件数量（需要适配）
original_yuv         # yuv文件输入地址（需要适配）
inputcoeff_dir       # 神经网络后处理系数的地址
frame_dir            # 帧号配置文件地址
out_pic_dir          # H265编码图片输出地址
wrongbits            # 误比特率配置，模拟信道传输
bppout               # bpp输出配置，布尔型变量，true时输出每帧bpp与平均bpp
```
在DCT系数提取阶段，误比特率设置无影响，可设为0。此外还需要将H.265项目下\source\encoder\encoder.h中的相应参数进行改动，如下：
```
outputdatas         # 用于判断是否提取编码端DCT系数，布尔型变量，true时执行提取
frame_dir           # 帧号文件路径，与上述main.cpp文件中路径相同
coeff_dir           # DCT系数存储地址
tu_dir              # TU规则的存储地址
```
之后调试运行H.265项目代码，得到DCT系数数据集。



### 2.3 Y通道像素数据集构建
将H.265编码图片地址填入y_txt.py文件中，设置Y通道系数输出地址，并运行该文件，得到H.265编码后的Y通道像素数据集。


## 3. 模型训练与测试


## 4. 模型重建数据集导入

### 4.1 H.265编解码图片导出（C++）
首先将H.265项目DCT系数提取功能关闭（将H.265项目下\source\encoder\encoder.h中的outputdatas变量设置false），定位到H.265项目下的main.cpp文件，同时更改变量wrongbits与编码解码图片输出路径out_pic_dir，得到不同信噪比下的H.265编码图片。本方案未将信道调制加入本次代码中，故采用误比特率模拟不同信噪比下传输方式，如下表：

| SNR  |             wrongbits              |
|:----:|:----------------------------------:|
| 10.5 |             0.0000906              |
| 10.6 |             0.0000302              |
| 10.7 |             0.0000246              |
| 10.8 |             0.0000042              |
| 10.9 |             0.0000012              |
再运行完所有信噪比条件下的H.265编解码图片后，定位到H.265项目根目录下\end\libde265\slice.cc文件中的residual_coding函数中将变量input_NTCcoeff更改为true，
```
input_NTCcoeff         # 判断是否导入后处理DCT系数，true时导入后处理数据
```

### 4.2 UV重建数据后处理
更改res_uvdata.py中相应参数进行更改，
```
sum_poc             # 总帧数
coeff_path          # DCT系数的存储地址
tu_path             # TU规则的存储地址
output_dir          # 重建UV系数主目录
```
运行res_uvdata.py，得到后处理系数数据。

### 4.3 UV通道数据导入H.265（C++）
定位到H.265项目根目录下\end\libde265\slice.cc文件中的residual_coding函数中，确定变量input_NTCcoeff为true。
再根据后处理系数地址，
```
1、更改根目录下main.cpp中的inputcoeff_dir；
2、更改对应的out_pic_dir，此时编码解码图片是NTC混合方案的重建结果；
3、当前信噪比对应的误比特率wrongbits。
```
运行H.265项目代码，得到UV数据重建图像。


### 4.4 Y通道重建数据导入
更改y_input.py中的参数，如下，并运行该文件。
```
sun_poc             # 总帧数
ydata_dir           # y通道重建数据地址
uvimg_dir           # uv通道重建图片地址
out_pic             # 方案最终重建结果图
```
最后得到的本次设计方案的最终图片out_pic。


## 5. 实验结果评估
更改eval.py中的参数,如下，并运行该文件。
```
final_nums              # 总共测试帧
BATCH_SIZE              # 每批评价计算的样本数
original_dir            # 原始图像地址
NTC_dir                 # 混合方案最终重建图像目录
H265_dir                # H265重建图像目录
out_eval                # 实验结果存储地址
index_H265              # 判断H.265结果是否计算，为1时输出
index_NTC               # 判断混合方案结果是否计算，为1时输出
```
运行时，库需要从github上自动下载模型，若无法连接github，则需根据报错信息手动下载相应模型，放在库中的相应位置
