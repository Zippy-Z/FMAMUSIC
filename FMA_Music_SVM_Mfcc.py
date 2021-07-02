import librosa
import numpy as np
import glob
import csv
import torch
import random
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA


# =============================================================================
#   获取flac文件时长
# =============================================================================
def get_flac_duration(audio_path):
    duration = librosa.get_duration(filename=audio_path)
    return duration


# =============================================================================
#   读取所有音频文件的路径和编号并存入列表输出
# =============================================================================
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
    print(flac_NO)
    return flac_NO, flac_path


# =============================================================================
#   从表格中读取音频对应的流派label_dict（音频编号-流派编号）（流派编号是0-7的整数，分别对应
#   Rock, Electronic, Experimental, Hip-Hop, Folk, Instrumental, Pop, International）
# =============================================================================
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


# =============================================================================
#   对表格中的音频编号和音频文件求交集，筛选出可用的音频样本
# =============================================================================
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
    print(final_genre_no)
    with open(path + split + '_genre.csv', 'w', newline='') as f1: #手动新建
        csv_writer = csv.writer(f1, dialect='excel')
        # csv_writer.writerow(["track", "genre top"])
        for i in range(0, len(final_genre_no)):
            csv_writer.writerow([final_flac_no[i], final_genre_no[i]])
    return final_flac_no, final_flac_path



# =============================================================================
#   音频长度判断，剔除小于29秒的音频样本
# =============================================================================
def duration_check(flac_NO, flac_path):
    for i in range(0,len(flac_NO)):
        j = len(flac_NO) - i - 1
        dura = int(get_flac_duration(flac_path[j]))
        if dura < 29:
            del flac_NO[j]
            del flac_path[j]
    return flac_NO, flac_path


# =============================================================================
#   mel频谱变换，4秒时长变换所得MEL矩阵大小为128*345
# =============================================================================
def MEL_trans(flac_path):
    MELS = np.ndarray((len(flac_path), 128, 345))
    for i in range(0, len(flac_path)):  #记得改
        y, sr = librosa.load(flac_path[i], sr=44100, offset=14.0, duration=4.0)
        ps = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=1024, hop_length=512, n_mels=128)
        print(flac_path[i])   #006517、006519、009887、009550、009962采样率是48000不能用098567是0秒
        MELS[i, :, :] = ps
    # MELS = np.expand_dims(MELS, 1)
    MELS = torch.from_numpy(MELS)
    MELS = MELS.double()
    return MELS


# =============================================================================
#   SVM分类器训练
# =============================================================================
def classifier_train(X_train, Y_train):
    print('start training classifier...')
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    print('finish training classifier!')
    return clf


# =============================================================================
#   SVM分类器测试
# =============================================================================
def classifier_test(clf, X_test, Y_test):
    print('start testing classifier...')
    Y_predicted = clf.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum() * 100
    print('finish testing classifier!')
    return confusion_matrix, acc


# =============================================================================
#   主成分分析PCA
# =============================================================================
def process(X_train, X_test, isSubMean, n_components):
    print('start processing data...')
    if isSubMean:
        X_train_mean = X_train.mean(0)
        X_train -= X_train_mean
        X_test -= X_train_mean
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.fit_transform(X_test)
    print('finish processing data!')
    return X_train_pca, X_test_pca


if __name__ == '__main__':
    # 去除随机性
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchamark = False
    torch.set_default_tensor_type(torch.DoubleTensor)


    path = "D:\锴子\深度学习\python文件"
    csv_path = "D:\锴子\深度学习\python文件/tracks_v2.csv"
    split = '/train_flac'

    train_flac_NO, train_flac_path = load_flac(path, split)
    train_flac_NO, train_flac_path = duration_check(train_flac_NO, train_flac_path)
    train_final_flac_no, train_final_flac_path = data_processing(train_flac_NO, train_flac_path, csv_path, path, split)
    train_x = MEL_trans(train_final_flac_path)
    train_x1 = np.expand_dims(train_x, 1)
    train_y = load_genre(path, split)
    print(train_y.shape)

    split = '/test_flac'

    test_flac_NO, test_flac_path = load_flac(path, split)
    test_flac_NO, test_flac_path = duration_check(test_flac_NO, test_flac_path)
    test_final_flac_no, test_final_flac_path = data_processing(test_flac_NO, test_flac_path, csv_path, path, split)
    test_x = MEL_trans(test_final_flac_path)
    test_x1 = np.expand_dims(test_x, 1)
    test_y = load_genre(path, split)

    train_xx = np.ndarray((train_x.shape[0], 128 * 345))
    test_xx = np.ndarray((test_x.shape[0], 128 * 345))
    for i in range(train_x.shape[0]):
        print("正在拉伸向量:" + str(i))
        train_xx[i, :] = train_x[i].reshape(128*345, )
    for i in range(test_x.shape[0]):
        print("正在拉伸向量:" + str(i))
        test_xx[i, :] = test_x[i].reshape(128*345, )
    isSubMean = False
    n_components = 200
    X_train, X_test = process(train_xx, test_xx, isSubMean, n_components)
    clf = classifier_train(train_xx, train_y)
    confusion_matrix, acc = classifier_test(clf, test_xx, test_y)
    print('accuracy: ' + str(acc) + '%')
    print(confusion_matrix)
