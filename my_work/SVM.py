import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Union,Literal

class SVM:
    def __init__(self, X, y, C:float=1.0, kernel:Literal['linear', 'rbf'] = 'rbf', gamma: Union[float,Literal['auto']]='auto', tol:float = 1e-3, max_iter:int = 10):
        self._X:np.ndarray = X  # 样本特征 m*n m个样本 n个特征
        self._y:np.ndarray = y  # 样本标签 m*1
        self._C:float = C  # 惩罚因子, 用于控制松弛变量的影响
        self._kernel:str = kernel  # 核函数
        self._m, self._n = X.shape
        self._gamma:float = 1./self._n if gamma == 'auto' else float(gamma)
        self._tol:float = tol # 容忍度
        self._max_iter:int = max_iter  # 最大迭代次数
        self._alpha:np.ndarray[float] = np.zeros(self._m) # 初始化alpha = [0,0,0,...,0]
        self._b:float = 0. # 初始化b = 0
        self._w:np.ndarray[float] = np.zeros(self._n) # 初始化w = [0,0,0,...,0]

    # 计算核函数
    def _K(self, i:int, j:int) -> np.ndarray:
        if self._kernel == 'linear':
            return np.dot(self._X[i].T, self._X[j])
        elif self._kernel == 'rbf':
            return np.exp(-self._gamma * np.linalg.norm(self._X[i] - self._X[j]) ** 2)
        else:
            raise ValueError('Invalid kernel specified')

    def predict(self, x_test) -> np.ndarray:
        pred = np.zeros_like(x_test[:, 0])
        pred = np.dot(x_test, self._w) + self._b
        return np.sign(pred)

    def _calculate_error(self, i:int):
        E_i = 0
        for ii in range(self._m):
            E_i += self._alpha[ii] * self._y[ii] * self._K(ii, i)
            E_i += self._b - self._y[i]
        return E_i
    
    def _is_violating_KKT(self, i:int, E_i:float):
        return (self._y[i] * E_i < -self._tol and self._alpha[i] < self._C) or (self._y[i] * E_i > self._tol and self._alpha[i] > 0)

    def _update_alphas(self, i:int, j:int, E_i:float, E_j:float):
        alpha_i_old = self._alpha[i].copy()
        alpha_j_old = self._alpha[j].copy()

        # L和H用于将alpha[j]调整到[0, C]之间
        if self._y[i] != self._y[j]:
            L = max(0, self._alpha[j] - self._alpha[i])
            H = min(self._C, self._C + self._alpha[j] - self._alpha[i])
        else:
            L = max(0, self._alpha[i] + self._alpha[j] - self._C)
            H = min(self._C, self._alpha[i] + self._alpha[j])

        # 如果L == H，则不需要更新alpha[j]
        if L == H:
            return False
        
        # eta: alpha[j]的最优修改量
        eta = 2 * self._K(i, j) - self._K(i, i) - self._K(j, j)
        # 如果eta >= 0, 则不需要更新alpha[j]
        if eta >= 0:
            return False
        
        # 更新alpha[j]
        self._alpha[j] -= (self._y[j] * (E_i - E_j)) / eta
        self._alpha[j] = np.clip(self._alpha[j], L, H) # 根据取值范围修剪alpha[j]

        # 检查alpha[j]是否只有轻微改变，如果是则退出for循环
        if abs(self._alpha[j] - alpha_j_old) < 1e-5:
            return False
        
        # 更新alpha[i]
        self._alpha[i] += self._y[i] * self._y[j] * (alpha_j_old - self._alpha[j])
        # 更新b1和b2
        b1 = self._b - E_i - self._y[i] * (self._alpha[i] - alpha_i_old) * self._K(i, i) \
             - self._y[j] * (self._alpha[j] - alpha_j_old) * self._K(i, j)
        b2 = self._b - E_j - self._y[i] * (self._alpha[i] - alpha_i_old) * self._K(i, j) \
             - self._y[j] * (self._alpha[j] - alpha_j_old) * self._K(j, j)
        # 根据b1和b2更新b
        if 0 < self._alpha[i] and self._alpha[i] < self._C:
            self._b = b1
        elif 0 < self._alpha[j] and self._alpha[j] < self._C:
            self._b = b2
        else:
            self._b = (b1 + b2) / 2
        return True
                                               
    def fit(self):
        """
        训练模型
        :return:
        """
        passes = 0
        while passes < self._max_iter:
            num_changed_alphas = 0
            for i in range(self._m):
                E_i = self._calculate_error(i)
                if self._is_violating_KKT(i, E_i):
                    # 随机选择样本x_j
                    j = np.random.choice(list(range(i)) + list(range(i + 1, self._m)), size=1)[0]
                    E_j = self._calculate_error(j)
                    num_changed_alphas += self._update_alphas(i, j, E_i, E_j)
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        # 提取支持向量和对应的参数
        idx = self._alpha > 0  # 支持向量的索引
        # SVs = X[idx]
        selected_idx = np.where(idx)[0]
        SVs = self._X[selected_idx]
        SV_labels = self._y[selected_idx]
        SV_alphas = self._alpha[selected_idx]
        
        # 计算权重向量和截距
        self._w = np.sum(SV_alphas[:, None] * SV_labels[:, None] * SVs, axis=0)
        self._b = np.mean(SV_labels - np.dot(SVs, self._w))
        # print("w", self._w)
        # print("b", self._b)
        return

    def score(self, x_test:np.ndarray, y_test:np.ndarray) -> np.ndarray:
        y_predict = self.predict(x_test)
        # 计算准确率
        # for index,y_pre in enumerate(y_predict):
        #     print('predicted=' + repr(y_pre) + ',actual=' + repr(y_test[index].item()))
        return np.mean(y_predict == y_test)


if __name__=='__main__':
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    print(iris.keys())
    X:np.ndarray = iris['data']
    y:np.ndarray = iris['target']
    print(X.shape, y.shape)
    # print(y)
    y[y != 0] = -1
    y[y == 0] = 1
    
    # 为了方便可视化，只取前两个特征，并且只取两类
    # plt.scatter(X[y != 1, 0], X[y != 1, 1], color='red')
    # plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
    # plt.show()
    
    # 数据预处理，将特征进行标准化，并将数据划分为训练集和测试集
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)#, random_state=40)
    # X_train_std = scaler.fit_transform(X_train)
    # X_test_std = scaler.fit_transform(X_test)
    
    # 创建SVM对象并训练模型
    svm = SVM(X_train, y_train, C=0.1, kernel='linear', tol=0.001, gamma=0.01, max_iter=15)
    svm.fit()
    
    # 预测测试集的结果并计算准确率
    accuracy = svm.score(X_test, y_test)
    print('正确率: {:.2%}'.format(accuracy))
    
    
    gammas = np.linspace(0, 0.5, 100)
    acc = list()
    for i in gammas:
        # 创建SVM对象并训练模型
        svm = SVM(X_train, y_train, C=15, kernel='linear', tol=0.001, gamma=0.01, max_iter=15)
        svm.fit()
        # 预测测试集的结果并计算准确率
        accuracy = svm.score(X_test, y_test)
        acc.append(accuracy)
        print('gamma:{},正确率: {:.2%}'.format(i, accuracy))
    
    plt.plot(gammas.tolist(), acc, marker='o', linestyle='-', color='b', label='曲线图')  # 设置标记、线型、颜色和标签
    plt.xlabel('X轴')  # 设置X轴标签
    plt.ylabel('Y轴')  # 设置Y轴标签
    plt.title('曲线图示例')  # 设置标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.show()