import numpy as np
import torch
import operator
from typing import Union,Literal
from sklearn.model_selection import train_test_split

class KNeighborsClassifier:
    def __init__(self, n_neighbors: int = 5, 
                weights: Union[Literal['uniform', 'distance'], None] = "uniform", 
                # algorithm: Union[Literal['auto', 'ball_tree', 'kd_tree', 'brute'], None] = "auto",
                p:Literal[1, 2] = 2):
        self._n_neighbors = n_neighbors
        self._weights = weights
        # self._algorithm = algorithm
        self._p = p
        
    def _getResponse(self,test_each:torch.Tensor) -> list[int]:
        # 列表里较小的前k项
        distances = []
        for index in range(self._x_train.shape[0]):
            train_each = self._x_train[index]
            dis = 0.
            if self._p == 2:
                # 计算两个张量之间的欧氏距离
                dis = torch.norm(test_each - train_each)
            else:
                # 曼哈顿距离
                dis = torch.sum(torch.abs(test_each - train_each))
            distances.append([self._y_train[index].item(), dis])
        distances.sort(key=operator.itemgetter(1))  # 按欧氏距离排序
        distances = distances[:self._n_neighbors]

        #  获取距离最近的K个实例中占比例较大的分类
        classVotes = {}

        for index, item in enumerate(distances):
            uns = item[0]
            weight = (1. if self._weights == 'uniform' else item[1]) / self._label_num[uns]
            if uns in classVotes:
                classVotes[uns] += weight
            else:
                classVotes[uns] = weight
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

        return sortedVotes[0][0]
    
    def fit(self, X:torch.Tensor, Y:torch.Tensor):
        X = X.reshape(X.shape[0],-1)
        Y = Y.reshape(-1)
        print("X_train Y_train的shape:",X.shape,Y.shape)
        if X.shape[0] != Y.shape[0]:
            print("X和Y的shape不符合")
            return
        if X.shape[0] < self._n_neighbors:
            print("训练集的数量小于n_neighbors")
            return
        self._x_train = X
        self._y_train = Y
        self._features = self._x_train.shape[1]
        
        # 计算每一个标签的个数
        self._label_num = {}
        for item in self._y_train.tolist():
            if item in self._label_num:
                self._label_num[item] += 1
            else:
                self._label_num[item] = 1

    def predict(self, x_test:torch.Tensor) -> list[int]:
        predictions = []
        x_test = x_test.reshape(x_test.shape[0],-1)
        if x_test.shape[1] != self._features:
            print("x_test的feature数量与X_train不符")

        for index in range(x_test.shape[0]):
            result = self._getResponse(x_test[index])
            predictions.append(result)
        return predictions
    
    def score(self, x_test:torch.Tensor, y_test:torch.Tensor) -> float:
        x_test = x_test.reshape(x_test.shape[0],-1)
        y_test = y_test.reshape(-1)
        if x_test.shape[1] != self._features:
            print("X_test的feature数量与X_train不符")
        if x_test.shape[0]!=y_test.shape[0]:
            print("X_test和Y_test的shape不符合")
        
        y_predict = self.predict(x_test)

        # 计算准确率
        correct = 0
        for index,y_pre in enumerate(y_predict):
            print('predicted=' + repr(y_pre) + ',actual=' + repr(y_test[index].item()))
            if y_test[index].item() == y_predict[index]:
                correct += 1
        return (correct / float(len(y_predict)))




def loadCifarDataset(filename:str) -> list:
    import pickle
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    label = dict[b'labels']
    data = dict[b'data']
    return [data, label]

if __name__ == "__main__":    
    # 使用鸢尾花卉数据集进行分类
    dataSet = loadIrisDataset(r'Iris_database\iris.txt')
    data = torch.FloatTensor([x[0:-1] for x in dataSet])
    print(data.shape)
    targetstr = [x[-1] for x in dataSet]
    unique_labels = list(set(targetstr))
    print("labels:",unique_labels)
    target = torch.IntTensor([unique_labels.index(x) for x in targetstr])
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.4,random_state=0)
    print("X_train shape:{}".format(X_train.shape),"X_test shape:{}".format(X_test.shape))
    print("y_train shape:{}".format(Y_train.shape),"y_test shape:{}".format(Y_test.shape))

    # # cifar
    # from torchvision.datasets import CIFAR10
    # train_dataset = CIFAR10(root='./data', train=True, download=True)
    # test_dataset = CIFAR10(root='./data', train=False, download=True)

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
    knn.fit(X_train,Y_train)
    print(knn.predict(X_test))
    print('Accuracy: ' + repr(knn.score(X_test,Y_test)*100) + '%')
