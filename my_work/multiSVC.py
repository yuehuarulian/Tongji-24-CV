import sys
import numpy as np
from typing import Union,Literal
import random

'''
implemented classes of classifiers.
'''
class ClsModel:
    def __init__(self,model_name='anonymous',num_cls=10):
        self.model_name = model_name
        self.num_cls = num_cls
        self.param_lst = []

    def fit(self,data,labels):
        '''
            fit cls model with train data.
        :param data: ndarray[n × m], n - samples, m - dimension
        :param labels: ndarray[1 × m], m - dimension
        :return:
        '''
        raise NotImplementedError

    def predict(self,data):
        '''
            predict test samples of their labels.
        :param data: ndarray[n × m], n - samples, m - dimension
        :return: ndarray[1 × m] - labels
        '''
        raise NotImplementedError

    def get_size_of(self):
        return sum([sys.getsizeof(ele) for ele in self.param_lst])

    def __str__(self):
        return 'ClsModel-' + self.model_name if self.model_name == 'anonymous' else 'ClsModel'


class SMC:
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

class MultiSVM(SVM):
    def __init__(self,**kwargs):
        super(MultiSVM, self).__init__(**kwargs)
        self.param_dict = kwargs
        self.svm_dict = {}
    def fit(self,data,labels):
        # init sub-SVM model.
        for k,cdata in self._gen_keys(data,labels):
            rnd_idx = list(range(cdata.shape[0]))
            random.shuffle(rnd_idx)
            svm = SMC(**self.param_dict)
            svm.fit(cdata[rnd_idx][:,:-1],cdata[rnd_idx][:,-1].reshape(-1))
            self.svm_dict[k] = svm

    def predict(self,data):
        preds = []
        for row_data in data:
            lb2cnt = {}
            max_lb = None
            max_cnt = 0
            for label in range(self.num_cls):
                for k,inf_lb in zip(*self._sel_pos(label)):
                    pred = self.svm_dict[k].predict(row_data.reshape(1, -1))
                    pred = 0 if pred > 0 else 1 # lb2idx.
                    if inf_lb[pred] is not None:
                        lb2cnt[inf_lb[pred]] = lb2cnt.get(inf_lb[pred],0) + 1
                        if lb2cnt[inf_lb[pred]] > max_cnt:
                            max_cnt = lb2cnt[inf_lb[pred]]
                            max_lb = inf_lb[pred]
            # assert max_lb is not None
            preds.append(max_lb if max_lb is not None else random.choice(range(self.num_cls)))
        return preds

    def _gen_keys(self,data,labels):
        raise NotImplementedError

    def _sel_pos(self,label):
        raise NotImplementedError

    def get_size_of(self):
        return sum([v.get_size_of() for k,v in self.svm_dict.items()])

    def __str__(self):
        return 'MultiSVM-' + self.model_name if self.model_name == 'anonymous' else 'MultiSVM'

class OVRSVM(MultiSVM):
    def _gen_keys(self,data,labels):
        cdata = np.concatenate([data, labels.reshape(-1, 1)], axis=1)
        for pos_cls in range(self.num_cls):
            pos_data = cdata[cdata[:, -1] == pos_cls]
            neg_data = cdata[cdata[:, -1] != pos_cls]
            pos_data[:, -1] = 1
            neg_data[:, -1] = -1
            yield pos_cls, np.concatenate([pos_data, neg_data], axis=0)

    def _sel_pos(self,label):
        return [label],[(label,None)]

    def __str__(self):
        return 'OVRSVM-' + self.model_name if self.model_name == 'anonymous' else 'OVRSVM'


class OVOSVM(MultiSVM):
    def _gen_keys(self, data, labels):
        cdata = np.concatenate([data, labels.reshape(-1, 1)], axis=1)
        for pos_cls in range(self.num_cls):
            for neg_cls in range(pos_cls+1,self.num_cls):
                pos_data = cdata[cdata[:, -1] == pos_cls]
                neg_data = cdata[cdata[:, -1] == neg_cls]
                pos_data[:, -1] = 1
                neg_data[:, -1] = -1
                yield '{}-{}'.format(pos_cls,neg_cls), np.concatenate([pos_data, neg_data], axis=0)

    def _sel_pos(self, label):
        return ['{}-{}'.format(label,num) for num in range(label+1,self.num_cls)], [(label,num) for num in range(label+1,self.num_cls)]

    def __str__(self):
        return 'OVOSVM-' + self.model_name if self.model_name == 'anonymous' else 'OVOSVM'