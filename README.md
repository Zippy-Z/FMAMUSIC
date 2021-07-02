Music genre classification  
=====  

#项目背景  
    本项目通过对音乐的音频文件提取特征，使用卷积神经网络学习，实现对音乐的风格/流派进行预测/分类。  

#安装  
##1.数据集获取  
    点击链接https://os.unil.cloud.switch.ch/fma/fma_small.zip 下载所用的数据集(7.2G)，将其中的MP3文件全部提取放到同一个文件夹中。  
    手动删除编号为098565，098567，098569，099134，108925，133297的文件(时长为0或严重小于30s)。接着使用格式工厂工具将数据从MP3转为flac格式（需要十分钟以上）。  
    最终数据集歌曲采样率为44.1KHz,数据集大小为40GB。

##2.代码文件获取  
    从本项目中直接下载FMA_Music_CNN_Mel.py文件即可。  
    FMA_Music_SVM_Mfcc.py为自对比试验中的测试代码，并非本项目的最终版本。  

#使用  
##1.数据集说明  
    本项目训练模型时从总集中随机提取1000个音频文件作为测试集，剩余音频作为训练集；实际训练中使用的训练集样本个数为7005，使用的测试集样本个数为945。实际读取数据时仅读取时长为29s的文 
    件，并截取从第一秒至第五秒的长度为4s的连续音频、从中提取特征用于训练。
    附件中的track_v2.csv文件中存储了本项目所使用的所有音频数据样本的风格流派标签；  
    test_flac_genre.csv、train_flac_genre.csv中分别存储测试集、训练集音频样本的编号与对应所属的风格流派编号  

##2.代码文件使用说明
    将代码文件FMA_Music_CNN_Mel.py与数据集放在同一个文件夹，根据代码中的注释修改代码中的数据集路径后即可运行程序开始模型训练。  
    FMA_Music_CNN_Mel.py中所调用的库声明：  
    import librosa  
    import numpy  
    import glob  
    import csv  
    import pydub  
    import codecs  
    import torch  
    import random  
    import sklearn 

#主要项目负责人  
   本项目主要负责人为冯滨涛，冯江锴，付旭煜，朱泽宇。

