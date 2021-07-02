import librosa
import numpy as np
import glob
import csv
import torch
import random
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import librosa.display

#   运行本文件前请阅读主函数中关于路径变量的修改的注释！


#   获取flac文件时长
def get_flac_duration(audio_path):
    duration = librosa.get_duration(filename=audio_path)
    return duration


#   读取所有音频文件的路径和编号并存入列表输出
def load_flac(path, split):
    f_path = path + split
    flac_path = []
    flac_NO = []
    for ind, file in enumerate(glob.glob(path + '/' + split + '/*.flac')):
        flac_path.append(file)
        s1 = file.split('\\')
        s2 = s1[-1].split('.')
        s3 = s2[0]
        flac_NO.append(s3)
    return flac_NO, flac_path


#   从表格中读取音频对应的流派label_dict（音频编号-流派编号）（流派编号是8行1列的矩阵，1表示对应流派，
#   顺序是Rock, Electronic, Experimental, Hip-Hop, Folk, Instrumental, Pop, International）
def load_genre(path, split):
    with open(path + split + '_genre.csv') as f:
        reader = csv.reader(f)
        genre_no = [row[1] for row in reader]
        length = len(genre_no)
        label = np.ndarray((length, ))
        for i in range(0, length):
            label[i] = int(genre_no[i]) - 1
        label = torch.from_numpy(label)
        return label


#   对表格中的音频编号和音频文件求交集，筛选出可用的音频样本
def data_processing(flac_NO, flac_path, csv_path, path, split):
    with open(csv_path, encoding='ANSI') as f:
        reader = csv.reader(f)
        flac_no_csv = [row[0] for row in reader]
        flac_no_csv = flac_no_csv[3:]
        for i in range(0, len(flac_no_csv)):
            t_str = flac_no_csv[i]
            if len(flac_no_csv[i]) < 6:
                for j in range(0, 6 - len(flac_no_csv[i])):
                    flac_no_csv[i] = '0' + flac_no_csv[i]
    with open(csv_path, encoding='ANSI') as f:
        reader = csv.reader(f)
        genre_no = [row[1] for row in reader]
        genre_no = genre_no[3:]
    final_flac_no = [a for a in flac_NO if a in flac_no_csv]
    final_flac_path = []
    final_genre_no = []
    for i in final_flac_no:
        final_flac_path.append(flac_path[flac_NO.index(i)])
        final_genre_no.append(genre_no[flac_no_csv.index(i)])
    with open(path + split + '_genre.csv', 'w', newline='') as f1: #手动新建
        csv_writer = csv.writer(f1, dialect='excel')
        for i in range(0, len(final_genre_no)):
            csv_writer.writerow([final_flac_no[i], final_genre_no[i]])
    return final_flac_no, final_flac_path


#   音频长度判断，剔除小于29秒的音频样本
def duration_check(flac_NO, flac_path):
    for i in range(0,len(flac_NO)):
        j = len(flac_NO) - i - 1
        dura = int(get_flac_duration(flac_path[j]))
        if dura < 29:
            del flac_NO[j]
            del flac_path[j]
    return flac_NO, flac_path


#   梅尔谱图变换，4秒时长变换所得MELS矩阵大小为128*345
def MELS_trans(flac_path, path, split,flac_no):
    MELS = np.ndarray((len(flac_path), 128, 345))
    for i in range(0, len(flac_path)):
        y, sr = librosa.load(flac_path[i], sr=44100, offset=1.0, duration=4.0)
        melspec = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=1024, hop_length=512, n_mels=128)
        # mfcc = librosa.feature.mfcc(y=y, sr=sr,n_fft=1024, n_mfcc=128)
        # logmelspec = librosa.power_to_db(melspec)
        # plt.figure()
        # librosa.display.specshow(logmelspec, sr=sr)
        # plt.savefig(path + split + "_jpg/" + flac_no[i] + ".jpg", bbox_inches="tight")
        # print(melspec.shape)
        print(flac_path[i])
        # m = np.mean(melspec)
        # mx = np.max(melspec)
        # mn = np.min(melspec)
        # im = Image.open(path + split + "_jpg/" + flac_no[i] + ".jpg")
        # im = im.convert('L')
        # y_s = 256
        # x_s = 256
        # im1 = im.resize((x_s, y_s), Image.ANTIALIAS)
        # im1 = np.array(im1)
        # im1 = im1/256
        # for j in range(3):
        #     MFCCS[i, j, :, :] = im1[:, :, j]
        # plt.close()
        # print(melspec)
        melspec =1 / (1.0 + np.exp(-melspec))
        MELS[i, :, :] = melspec
    MELS = np.expand_dims(MELS, 1)
    MELS = torch.from_numpy(MELS)
    MELS = MELS.double()
    return MELS


# 使用pytorch搭建卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(  # 创建时序容器
            nn.Conv2d(1, 4, kernel_size=(3, 4)),
            nn.BatchNorm2d(4),  # 归一化
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=3)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * 20 * 56, 8),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#   计算预测结果的正确率
def Get_Acc(out, label):
    num_correct = 0
    for i in range(label.shape[0]):
        max_index1 = np.argmax(out[i, :])
        print("第" + str(i) + "个样本流派与预测值：")
        print(label[i])
        print(np.argmax(out[i, :]))
        if max_index1 == label[i]:
            num_correct += 1
    Acc = num_correct / out.shape[0]
    return Acc


if __name__ == '__main__':

    # 定义一些超参数
    batch_size = 64
    learning_rate = 1e-2
    num_epoches = 10

    # 去除随机性
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchamark = False
    torch.set_default_tensor_type(torch.DoubleTensor)

    #   修改路径变量：path路径下应创建文件夹train_flac用于存储训练集音频文件(flac格式)，以及创建文件夹test_flac用于存储测试集音频文件(flac格式)，
    #   代码运行后将会在path路径下生成两个csv表格文件，train_flac_genre.csv中存储训练集中音频文件编号及所属流派编号，test_flac_genre.csv中存储测试集中音频文件编号及所属流派编号
    path = "D:\锴子\深度学习\python文件"
    csv_path = "D:\锴子\深度学习\python文件/tracks_v2.csv"
    
    #   训练集数据预处理
    split = '/train_flac'
    train_flac_NO, train_flac_path = load_flac(path, split)
    train_flac_NO, train_flac_path = duration_check(train_flac_NO, train_flac_path)
    train_final_flac_no, train_final_flac_path = data_processing(train_flac_NO, train_flac_path, csv_path, path, split)
    train_x = MELS_trans(train_final_flac_path, path, split, train_final_flac_no)
    train_y = load_genre(path, split)
    print(train_y.shape)

    #   测试集数据预处理
    split = '/test_flac'
    test_flac_NO, test_flac_path = load_flac(path, split)
    test_flac_NO, test_flac_path = duration_check(test_flac_NO, test_flac_path)
    test_final_flac_no, test_final_flac_path = data_processing(test_flac_NO, test_flac_path, csv_path, path, split)
    test_x = MELS_trans(test_final_flac_path, path, split, test_final_flac_no)
    test_y = load_genre(path, split)

    #   数据集封装
    Train_data = TensorDataset(train_x, train_y)
    Train_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=False)
    Test_data = TensorDataset(test_x, test_y)
    Test_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False)

    # 选择模型
    Model = CNN()
    if torch.cuda.is_available():
        Model = Model.cuda()

    # 定义损失函数和优化器
    Criterion = nn.CrossEntropyLoss()
    Optimizer = optim.SGD(Model.parameters(), lr=learning_rate)

    # 训练模型
    for count in range(num_epoches):
        print("回合：" + str(count))
        for data in Train_loader:
            mfcc, label = data
            mfcc = Variable(mfcc)
            if torch.cuda.is_available():
                mfcc = mfcc.cuda()
                label = label.cuda()
            else:
                mfcc = Variable(mfcc)
                label = Variable(label)
            label = label.long()
            out = Model(mfcc)
            loss = Criterion(out, label)
            print_loss = loss.data.item()
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

    # 模型评估
    Pred = np.zeros([945, 8])
    count = 0
    Model.eval()
    for data in Test_loader:
        mfcc, label = data
        mfcc = Variable(mfcc)
        # print(mfcc)
        if torch.cuda.is_available():
            mfcc = mfcc.cuda()
            label = label.cuda()
        label = label.long()
        out = Model(mfcc)
        print(out)
        for xy in range(list(out.size())[0]):
            for i in range(8):
                Pred[count, i] = out[xy, i]
            count += 1
        loss = Criterion(out, label)
    Acc = Get_Acc(Pred, test_y)
    # print(Pred)
    print('测试集正确率: ' + str(100 * Acc) + '%')


















































































