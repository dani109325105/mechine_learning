import random
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix

#初始化沒問題

class _particle(object):

    def __init__(self, weight, speed):
        self.weight = weight
        self.speed = speed
        self.r = []
        self.fitness = (2**31)-1
        self.local_best_weight = []

    def update_r(self, r):
        self.r = r
    
    
    def record_local_best(self, fitness, weight):
        self.fitness = fitness
        #assert(len(weight)==540)
        self.local_best_weight = weight
        
    
def init_particle(amount, total_weight):
    particle = []
    for i in range(amount):
        weight = []
        speed = []
        for j in range(total_weight):
            weight.append(np.random.uniform(-2,2))#常態分佈初始化權重
            speed.append(np.random.uniform(-2,2))#常態分佈初始化速度
        p = _particle(weight, speed)#initial partical status
        #assert(len(p.weight)==540)
        particle.append(p)

    return particle


class _model(object):

    def __init__(self, input_x, input_y, particle_amount, loss_fn):
        self.x = input_x
        self.y = input_y
        self.particle_amount = particle_amount
        self.w = 0
        self.layers = []
        self.global_MSE = (2**31)-1
        self.global_weight = []
        #self.global_index = []
        self.loss_fn = loss_fn
        self.confuse_matrix = [[0] * 4 for i in range(3)]
    
    def sigmoid(self, x):
        x = x*-1 
        s = 1 / (1 + np.exp(x))
        return s

    def show_confuse(self):#算percision、recall
        precision = 0
        recall = 0
        f1score = 0
        accuracy = 0
        total_accuracy = 0
        total_f1score = 0
        total_recall = 0
        total_precision = 0
        r = []
        d_sample = [0,0,0]
        total_sample = len(self.y)
        #各類分別多少
        for i in range(len(self.y)):
            d_sample[self.y[i].index(max(self.y[i]))] += 1
        #各類比例
        for i in range(3):#Weighted-average
            r.append(d_sample[i]/total_sample)

        for i in range(len(self.confuse_matrix)):#0:TP,1:FN,2:FP,3:TN
            #precision
            if (self.confuse_matrix[i][0]+self.confuse_matrix[i][2]) > 0:
                precision = self.confuse_matrix[i][0]/(self.confuse_matrix[i][0]+self.confuse_matrix[i][2])
            else:
                precision = 0
            #recall
            if (self.confuse_matrix[i][0]+self.confuse_matrix[i][1]) > 0:
                recall = self.confuse_matrix[i][0]/(self.confuse_matrix[i][0]+self.confuse_matrix[i][1])
            else:
                recall = 0
            #f1score
            if (precision+recall) > 0:
                f1score = 2*precision*recall / (precision+recall)
            else:
                f1score = 0
            #accuracy
            if(self.confuse_matrix[i][0]+self.confuse_matrix[i][1]+self.confuse_matrix[i][2]+self.confuse_matrix[i][3]) > 0:
                accuracy = (self.confuse_matrix[i][0]+self.confuse_matrix[i][3])/(self.confuse_matrix[i][0]+self.confuse_matrix[i][1]+self.confuse_matrix[i][2]+self.confuse_matrix[i][3])
            else:
                accuracy = 0
            total_accuracy += accuracy*r[i]
            total_precision += precision*r[i]
            total_recall += recall*r[i]
            print("%d:accuracy%f,precision%f,recall%f,f1score%f" % (i, accuracy, precision, recall, f1score))
        total_f1score = 2*total_precision*total_recall/(total_precision+total_recall)
        print("total_accuracy%f,total_precision%f,total_recall%f,total_f1score%f" % (total_accuracy, total_precision, total_recall, total_f1score))
    
    def reset_confuse(self):#
        self.confuse_matrix = [[0] * 4 for i in range(3)]

    def add_layer(self, layer_unit):#新增網路層數
        self.layers.append(layer_unit)#[8,8,3]

    def update_r(self, p):#更新r
        r = []
        for i in range(2):
            r.append(random.random())
        p.r = r

    def update_w(self, T_max, t):#線性遞減公式
        self.w = 0.9-(0.9-0.4)/T_max*t

    def update_weight(self, p):
        for i in range(len(p)):
            v_local = []
            v_global = []
            v=[]
            assert(len(self.global_weight)==len(p[i].local_best_weight)==len(p[i].weight))
            for x in range(len(self.global_weight)):
                v_local.append(2*p[i].r[0]*(p[i].local_best_weight[x]-p[i].weight[x]))
                v_global.append(3*p[i].r[1]*(self.global_weight[x]-p[i].weight[x]))
            #速度更新
            for y in range(len(p[i].speed)):
                v.append(self.w*p[i].speed[y]+v_local[y]+v_global[y])#公式三
            #速度+-4
            for j in range(len(v)):
                if v[j] > 1:
                    v[j] = 1
                elif v[j] < -1:
                    v[j] = -1
            #更新位置
            for z in range(len(p[i].weight)):
                p[i].weight[z] += v[z]
            for k in range(len(p[i].weight)):
                if p[i].weight[k]<-2:
                    p[i].weight[k] = -2
                elif p[i].weight[k]>2:
                    p[i].weight[k] = 2
        return p
        


    #main
    def fit(self, iteration):
        p = self._create_particle(self.particle_amount)#p is that have five particle
        iteration_ = 0
        while iteration_ != iteration:#迭代次數
            print("第%d次iteration"%(iteration_))
            for h in range(len(p)):#跑每個粒子 
                total_MSE = 0
                self.update_r(p[h])#更新r1,r2的值, r:[0, 1]的隨機值
                self.update_w(iteration, iteration_)
                self.loss(p[h], h)#算適應值
            p = self.update_weight(p)
            iteration_ += 1
        return p
    
    def _create_particle(self, particle_amount=5):
        input_amount = len(self.x[0])
        total_weight=0
        for layer in self.layers:#how many weight in model ?
            total_weight += layer*input_amount
            input_amount = layer
        p = init_particle(particle_amount, total_weight)
        return p

    def loss(self, p, h):
        total_mse = 0
        for i in range(len(self.x)):
            total_mse += self.calculation(p, i)#each data calculate mse loss
        if p.fitness >= total_mse:
            p.record_local_best(total_mse, p.weight)#local_best
            if self.global_MSE > total_mse:
                print("當前global_MSE:%f"%(self.global_MSE/len(self.x)))
                self.global_MSE = total_mse#global_best
                self.global_weight = p.weight
    '''
    def CCE(self, input_x, input_y):#y_pred : input_x, y_true : self.y[index]
        
        if type(y_true) == pd.DataFrame:
            ground_truth = y_true.values.tolist()
        else:
            ground_truth = y_true
        total_CE = 0
        prob = []
        for i_target, i_pred in zip(input_y, input_x):#true:y, pred:x
            if i_target == 1:
                # required a proper smoothing operation
                i_pred = i_pred if i_pred > 1e-7 else 1e-7
                i_pred = i_pred if i_pred < 1 - 1e-7 else 1 - 1e-7
                prob.append(i_pred)

        prob = np.array(prob, dtype='float32')
        prob_tensor = tf.constant(prob)
        log_tensor = tf.math.log(prob_tensor)
        loss = tf.reduce_sum(log_tensor).numpy()

        total_CE = -1 * loss / len(input_x)
        return total_CE
    '''
    def MSE(self, input_x, input_y):#loss_function
        MSE = 0
        for i in range(len(input_x)):
            MSE += pow(input_y[i]-input_x[i],2)
        return MSE
    
    def confuse(self, predict, actual):#0:TP,1:FN,2:FP,3:TN

        if predict.index(max(predict)) == actual.index(max(actual)):#TP
            self.confuse_matrix[predict.index(max(predict))][0]+=1
            for i in range(len(predict)):
                if(i!=predict.index(max(predict))):
                    self.confuse_matrix[i][3]+=1#TN
        else:
            self.confuse_matrix[predict.index(max(predict))][1]+=1#FN
            self.confuse_matrix[actual.index(max(actual))][2]+=1#FP
            for i in range(len(predict)):
                if (i!=predict.index(max(predict))&i!=actual.index(max(actual))):
                    self.confuse_matrix[i][3]+=1#TN
        '''
        tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
        '''


    #要加activation_function
    def calculation(self, p, data_index):#caculate every data loss with one particle
        input_y = self.y[data_index]
        input_x = np.array(self.x[data_index])#one row data
        unit_value = 0
        count = 0
        bias = 0
        for layer in self.layers:#forward propagation 
            next_input_data = []     
            for i in range(layer):
                for j in range(len(input_x)):
                    bias = random.random()
                    unit_value += input_x[j]*p.weight[count]+bias
                    count+=1
                unit_value = self.sigmoid(unit_value)#activation:sigmoid
                next_input_data.append(unit_value)
            input_x = np.array(next_input_data)#最後有三個值
        predict = []
        for i in range(len(input_x)):
            predict.append(input_x[i]/sum(input_x))#normalization
        self.confuse(predict, input_y)
        if self.loss_fn == "MSE":
            return self.MSE(input_x, input_y)
        else:
            return self.CCE(input_x, input_y)
    
    #test
    def _evaluate(self, test_x, test_y):
        total_mse = 0
        self.x = test_x
        self.y = test_y
        for i in range(len(test_x)):
            total_mse += self._calculation(i)#every data calculate mse loss
        print("test_loss:%f"%(total_mse/len(self.x)))
    
    def _calculation(self, data_index):
        input_x = np.array(self.x[data_index])#one row data
        input_y = self.y[data_index]
        unit_value = 0
        count = 0
        bias = 0
        for layer in self.layers:#forward propagation 
            next_input_data = []     
            for i in range(layer):
                for j in range(len(input_x)):
                    bias = random.random()
                    unit_value += input_x[j]*self.global_weight[count]+bias
                    count+=1
                unit_value = self.sigmoid(unit_value)#activation:sigmoid
                next_input_data.append(unit_value)#下一層輸入可以加activation_function
            input_x = np.array(next_input_data)#最後有三個值
        predict = []
        for i in range(len(input_x)):
            predict.append(input_x[i]/sum(input_x))#normalization
        assert len(predict)==len(input_y)
        self.confuse(predict, input_y)
        if self.loss_fn == "MSE":
            return self.MSE(input_x, input_y)
        else:
            return self.CCE(input_x, input_y)

#資料前處理
def data_process(path, file):
    input_data = pd.DataFrame()
    position = path + "/" + file 
    df = pd.read_table(position, header=None)
    input_data = input_data.append(df, ignore_index=True)
    x = []
    y = []
    #label
    input_data_5 = pd.get_dummies(input_data[5])
    input_data_5.columns = [0,1,2]
    #feature
    for i in range(len(input_data[1])):
        temp_x = []
        for j in range(1,5):
            temp_x.append(input_data[j][i]-input_data[j].min()/input_data[j].max()-input_data[j].min())
        x.append(temp_x)
        y.append(input_data_5.loc[i].tolist())
    return x,y
            
if __name__ == '__main__':
    #讀資料
    test_path = "C:/Users/TCU-2373-NB1/all_data/machine_learning/final_project/testing"
    files = os.listdir
    

    train_path = "C:/Users/TCU-2373-NB1/all_data/machine_learning/final_project/training"
    files = os.listdir(train_path)
    input_data = pd.DataFrame()
    count=1
    x=[]
    y=[]
    
    for file in files:
        x,y = data_process(train_path,file)

        #training_position = train_path + "/" + file 
        #df = pd.read_table(training_position, header=None)
        #建立模型
        model = _model(x, y, 100, loss_fn="MSE")#1000
        model.add_layer(8)
        model.add_layer(8)
        model.add_layer(3)
        #model.fit
        print("以下為第%d次訓練更改global_weight"%(count))
        p = model.fit(30)
        #印當次的TP、FN、FP之後要更新
        print("第%d個檔案training:"%(count))
        model.show_confuse()

        model.reset_confuse()#model confuse matrix reset

        #testing部分
        test_file = file.replace("training","testing")
        x,y = data_process(test_path, test_file)
        #evaluate()q
        model._evaluate(x,y)
        print("第%d個檔案testing:"%(count))
        model.show_confuse()
        model.reset_confuse()#model confuse matrix reset
        count+=1
