import librosa
import numpy as np
import glob
import csv
import pydub
import codecs
import torch
import random
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


#   获取flac文件时长
def get_mp3_duration(audio_path):
    duration = librosa.get_duration(filename=audio_path)
    # duration = len('1.mp3')audio_path
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
    # print(flac_NO)
    return flac_NO, flac_path


#   从表格中读取音频对应的流派label_dict（音频编号-流派编号）（流派编号是8行1列的矩阵，1表示对应流派，
#   顺序是Rock, Electronic, Experimental, Hip-Hop, Folk, Instrumental, Pop, International）
def load_genre(path, split):
    # with open(path + split + '_genre.csv') as f:
    #     reader = csv.reader(f)
    #     flac_no_csv = [row[0] for row in reader]
    with open(path + split + '_genre.csv') as f:
        reader = csv.reader(f)
        genre_no = [row[1] for row in reader]
        # flac_no_csv = flac_no_csv[1:]
        # genre_no = genre_no[1:]
        length = len(genre_no)
        label = np.ndarray((length, ))
        for i in range(0, length):
            label[i] = int(genre_no[i]) - 1
        label = torch.from_numpy(label)
        # label = label.double()
        return label
        # empty_genre = []
        # not_target_genre = []
        # temp_label = []
        # target_genre = ['1', '2', '3', '4', '5', '6', '7', '8']
            # if genre_no[i] == "[]":
            #     empty_genre.append[i]
            # else:
            #     text = genre_no[i]
            #     t1 = text.split(']')[0].split('[')[1]
            #     t2 = t1.split(', ')
            #     signal = 0
            #     temp_l = np.zeros((8,))
            #     for no in t2:
            #         if no in target_genre:
            #             signal = 1
            #             temp_l[target_genre.index(no)] = 1
            #         else:
            #             continue
            #     if signal == 0:
            #         not_target_genre.append(i)
            #     else:
            #         temp_label.append(temp_l)
                # if len(t2) == 1:
                #     if t2 in target_genre:
                #         genre_no[i] = t2
                #     else:
                #         not_target_genre.append(i)
                # else:
                #     for g_no in t2:
                #         if g_no in target_genre:
                #             genre_no[i] = t2
                #             continue
                #     not_target_genre.append(i)
        # not_target_genre.extend(empty_genre)
        # not_target_genre.sort(reverse=True)
        # for i in not_target_genre:
        #     del flac_no[i]
        # for i in range(0, len(flac_no)):
        #     label_dict[flac_no[i]] = temp_label


#   对表格中的音频编号和音频文件求交集，筛选出可用的音频样本
def data_processing(flac_NO, flac_path, csv_path, path, split):
    with open(csv_path, encoding='ANSI') as f:
        reader = csv.reader(f)
        flac_no_csv = [row[0] for row in reader]
        # genre_no = [row[1] for row in reader]
        flac_no_csv = flac_no_csv[3:]
        # genre_no = genre_no[3:]
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
    # print(len(final_flac_no))
    # print(flac_no_csv)
    final_flac_path = []
    final_genre_no = []
    for i in final_flac_no:
        final_flac_path.append(flac_path[flac_NO.index(i)])
        final_genre_no.append(genre_no[flac_no_csv.index(i)])
        # print(flac_NO.index(i))
    # print(final_flac_no)
    # print(final_genre_no)
    with open(path + split + '_genre.csv', 'w', newline='') as f1: #手动新建
        csv_writer = csv.writer(f1, dialect='excel')
        # csv_writer.writerow(["track", "genre top"])
        for i in range(0, len(final_genre_no)):
            csv_writer.writerow([final_flac_no[i], final_genre_no[i]])
        # f1.close()
    return final_flac_no, final_flac_path


#   音频长度判断，剔除小于29秒的音频样本
def duration_check(flac_NO, flac_path):
    for i in range(0,len(flac_NO)):
        j = len(flac_NO) - i - 1
        dura = int(get_mp3_duration(flac_path[j]))
        if dura < 29:
            del flac_NO[j]
            del flac_path[j]
    return flac_NO, flac_path


#   mfcc变换，29秒时长变换所得MFCC矩阵大小为128*345
def MFCC_trans(flac_path):
    MFCCS = np.ndarray((len(flac_path), 128, 345))
    for i in range(0, len(flac_path)):
        y, sr = librosa.load(flac_path[i], sr=44100, offset=1.0, duration=4.0)
        melspec = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=1024, hop_length=512, n_mels=128)
        # print(melspec.shape)
        print(flac_path[i])
        # m = np.mean(melspec)
        # mx = np.max(melspec)
        # mn = np.min(melspec)
        melspec = 1.0 / (1 +np.exp(-melspec))
        MFCCS[i, :, :] = melspec
        # print(melspec)
    MFCCS = np.expand_dims(MFCCS, 1)
    MFCCS = torch.from_numpy(MFCCS)
    MFCCS = MFCCS.double()
    return MFCCS

# =============================================================================
# 使用pytorch搭建卷积神经网络
# =============================================================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(  # 创建时序容器
            nn.Conv2d(1, 4, kernel_size=(3,4)),
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
            nn.Linear(8 * 20 * 56, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 8),
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


def Get_Acc(out, label):
    num_correct = 0
    for i in range(out.shape[0]):
        max_index1 = np.argmax(out[i, :])
        print("第" + str(i) + "次：")
        print(label[i])
        print(np.argmax(out[i, :]))
        if max_index1 == label[i]:
            num_correct += 1
    Acc = num_correct / out.shape[0]
    return Acc


if __name__ == '__main__':

   # 定义一些超参数
    batch_size = 10
    learning_rate = 1e-6   # adam
    # num_epoches = 20

   # 去除随机性
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchamark = False
    torch.set_default_tensor_type(torch.DoubleTensor)

    path = "F:\学习资料\大学资料\大二下资料\深度学习\音乐分类"
    csv_path = "F:\学习资料\大学资料\大二下资料\深度学习\音乐分类/tracks_v2.csv"
    split = '/train_flac'

    train_flac_NO, train_flac_path = load_flac(path, split)
    train_flac_NO, train_flac_path = duration_check(train_flac_NO, train_flac_path)
    train_final_flac_no, train_final_flac_path = data_processing(train_flac_NO, train_flac_path, csv_path, path, split)
    train_x = MFCC_trans(train_final_flac_path)
    train_y = load_genre(path, split)
    print(train_y.shape)

    split = '/test_flac'

    test_flac_NO, test_flac_path = load_flac(path, split)
    test_flac_NO, test_flac_path = duration_check(test_flac_NO, test_flac_path)
    test_final_flac_no, test_final_flac_path = data_processing(test_flac_NO, test_flac_path, csv_path, path, split)
    test_x = MFCC_trans(test_final_flac_path)
    test_y = load_genre(path, split)

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
    Optimizer = optim.Adam(Model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)

    # 训练模型
    Epoch = 0
    for data in Train_loader:
        mfcc, label = data
        mfcc = Variable(mfcc)
        if torch.cuda.is_available():
            mfcc = mfcc.cuda()
            label = label.cuda()
        else:
            mfcc = Variable(mfcc)
            label = Variable(label)
        print("训练" + str(Epoch) + ":")
        print(label)
        label = label.long()
        out = Model(mfcc)
        # print(type(label[0]))
        loss = Criterion(out, label)
        print_loss = loss.data.item()
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
        Epoch += 1

    # 模型评估
    Pred = np.zeros([200, 8])
    count = 0
    Model.eval()
    # eval_loss = 0                                                               
    # eval_acc = 0
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
    print('正确率: ' + str(100 * Acc) + '%')





