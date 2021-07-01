# FMAMUSIC  
Music genre classification  
点击链接https://os.unil.cloud.switch.ch/fma/fma_small.zip 下载所用的数据集(7.2G)，将其中的MP3文件全部提取放到同一个文件夹中。  
手动删除编号为098565，098567，098569，099134，108925，133297的文件(时长为0或严重小于30s)。接着使用格式工厂工具将数据从MP3转为flac格式（需要十分钟以上）。  
最终数据集歌曲采样率为44.1KHz,数据集大小为40GB。  
实际读取数据时仅读取时长为29s的文件，并取中间4s用于训练。  
附件中的csv文件有对剩下的歌曲的流派的标注。  
将代码与数据集放在同一个文件夹，修改代码中的数据集路径。  
将数据集分为训练集和测试集(本小组随机抽取了200个文件作为测试集)，开始训练,其中的两个py文件分别使用了svm及cnn网络。  
调用的库声明：  
import librosa  
import numpy  
import glob  
import csv  
import pydub  
import codecs  
import torch  
import random  
import sklearn  
