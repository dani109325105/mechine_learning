import random
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score

#初始化沒問題

class _gene(object):

    def __init__(self, weight):
        self.weight = weight
        self.fitness = (2**31)-1
        
def init_gene(amount, total_weight):
    gene = []
    for i in range(amount):
        weight = []
        for j in range(total_weight):
            weight.append(np.random.uniform(-3,3))#常態分佈初始化權重
        g = _gene(weight)#initial partical status
        gene.append(g)
        
    return gene


class _model(object):

    def __init__(self, input_x, input_y, gene_amount, loss_fn):
        self.x = input_x
        self.y = input_y
        self.gene_amount = gene_amount
        self.layers = []
        self.global_loss = (2**31)-1
        self.global_weight = []
        self.loss_fn = loss_fn
        self.confuse_matrix = [[0] * 4 for i in range(3)]
        self.y_true = []
        self.y_pred = []
    
    def sigmoid(self, x):
        x = x*-1 
        s = 1 / (1 + np.exp(x))
        return s

    def calculate_confuse(self):
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
            #print("%d:accuracy%f,precision%f,recall%f,f1score%f" % (i, accuracy, precision, recall, f1score))每一個類別分別為多少
        if (total_precision+total_recall)>0:
            total_f1score = 2*total_precision*total_recall/(total_precision+total_recall)
        else:
            total_f1score = 0

        return self.global_loss, total_accuracy, total_precision, total_recall, total_f1score
    
    def reset_confuse(self):#
        self.confuse_matrix = [[0] * 4 for i in range(3)]

    def add_layer(self, layer_unit):#新增網路層數
        self.layers.append(layer_unit)

    #新增mask    
    def create_mask(self, length):
        weight = []
        for j in range(length):
            weight.append(random.randint(0,1))#0,1初始化mask
        return weight
    #選擇父代
    def crossover(self, g, repeat):
        parent1 = _gene([])
        parent2 = _gene([])
        rand = 0 
        for i in range(len(g)):
            odd = random.random()
            #希望越前面的染色體最高機率被選到,盡量不要重複選
            if odd < 0.8*(100-i*10)/100*10/abs(10-repeat[i]*100):
                parent1 = g[i]
                repeat[i]+=1
                break
        #如果都沒選到就隨機選一條
        if len(parent1.weight)==0:
            rand = random.randint(0,len(g)-1)
            parent1 = g[rand]
            repeat[rand]+=1

        for i in range(len(g)):
            odd = random.random()
            if odd < 0.8*(100-i*10)/100*10/abs(10-repeat[i]*100):
                parent2 = g[i]
                repeat[i]+=1
                break
        if len(parent2.weight)==0:
            rand = random.randint(0,len(g)-1)
            parent2 = g[rand]
            repeat[rand]+=1

        return parent1,parent2

    def update_weight(self, g):#g=所有母代
        offspring = []
        count=0
        yn=1
        parent1 = []
        parent2 = []
        loss = []
        temp=0
        mask = []
        g_loss = []
        repeat = [0 for i in range(len(g))]
        position = 0
        pick_up = 0
        new_loss = 0

        #判斷母體是否有重複的權重
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                if g[i].fitness==g[j].fitness:
                    new_g = _gene([])
                    for k in range(len(g[i].weight)):
                        new_g.weight.append(np.random.uniform(-3,3))
                    g[j] = new_g
                    new_loss = self.offspring_loss(g[j])
                    g[j].fitness = new_loss
        
        #根據g的適應值排序
        for i in range(len(g)):
            g_loss.append(g[i].fitness)

        quicksort(g_loss, g, 0, len(g_loss)-1)#sort:loss、g
        for i in range(len(g)):
            g[i].fitness = g_loss[i]
            #print("上面%d"%(i+1),g[i].fitness)
        
        #隨機挑兩個交配，使用均勻交配
        rand=0
        count=0
        while yn == 1:
            if len(offspring) < len(g):
                #產生父代1,父代2
                parent1, parent2 = self.crossover(g, repeat) 
                
                #產生mask
                mask = self.create_mask(len(g[0].weight))
                for i in range(len(mask)):
                    if mask[i]==0:#如果mask[i]==0兩條作互換產生兩條子代
                        temp = parent1.weight[i]
                        parent1.weight[i] = parent2.weight[i]
                        parent2.weight[i] = temp
                offspring.append(parent1)
                offspring.append(parent2)
                #交配突變
                repeat = [0 for i in range(len(g))]
                parent1, parent2 = self.crossover(g, repeat)
                if random.random() > 0.5:
                    for i in range(int(len(parent1.weight)/20)):
                        position = random.randint(0,len(parent1.weight)-1)
                        item = random.randint(0,2)
                        if item == 1:
                            parent1.weight[position] = np.random.uniform(-3,3)#parent1.weight[position]+random.uniform(-0.01,0.01)#移動
                        else:
                            parent2.weight[position] = np.random.uniform(-3,3)#parent2.weight[position]+random.uniform(-0.01,0.01)
                offspring.append(parent1)
                offspring.append(parent2)
            else:#離開迴圈
                yn=0
        #print("repeat",repeat)
        #評估子代適應值
        for h in range(len(offspring)):#跑每一條基因
            loss.append(self.offspring_loss(offspring[h]))#算適應值
        
        quicksort(loss, offspring, 0, len(loss)-1)#sort:loss、offspring

        #選擇1/2子代如果loss值更低則取代母體offspring
        for i in range(len(g)-int(len(g)/2),len(g)):
            if(g[i].fitness>loss[i-int(len(offspring)/2)+1]):
                g[i].weight = offspring[i-int(len(offspring)/2)+1].weight
                g[i].fitness = loss[i-int(len(offspring)/2)+1]
        
        #判斷最後替代完的母體是否有重複的權重
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                if g[i].fitness==g[j].fitness:
                    new_g = _gene([])
                    for k in range(len(g[i].weight)):
                        new_g.weight.append(np.random.uniform(-3,3))
                    g[j] = new_g
                    new_loss = self.offspring_loss(g[j])
                    g[j].fitness = new_loss
    
        return g
        
    def offspring_loss(self, offspring):
        total_loss = 0
        self.y_pred=[]
        self.y_true=[]
        for i in range(len(self.x)):
            self.calculation(offspring, i)#each data calculate mse loss
        
        if(self.loss_fn=="MSE"):
            total_loss = self.MSE()
        else:
            total_loss = self.CCE()
        return total_loss

    #main
    def fit(self, iteration):
        g = self._create_gene(self.gene_amount)#g is that have five gene
        iteration_ = 0
        while iteration_ != iteration:#迭代次數
            print("第%d次iteration"%(iteration_))
            for h in range(len(g)):#跑每一條基因
                self.loss(g[h])#算適應值
            g = self.update_weight(g)
            iteration_ += 1
        #global_weight做confuse_matrix
        self.y_pred=[]
        self.y_true=[]
        for i in range(len(self.x)):
            self._calculation(i)
        self.confuse()
    
    def _create_gene(self, gene_amount=5):
        input_amount = len(self.x[0])
        total_weight=0
        for layer in self.layers:#how many weight in model ?
            total_weight += layer*input_amount
            total_weight += layer
            input_amount = layer
        g = init_gene(gene_amount, total_weight)
        return g

    def loss(self, g):#g:某一條基因
        total_loss = 0
        self.y_pred=[]
        self.y_true=[]
        for i in range(len(self.x)):
            self.calculation(g, i)#each data calculate mse loss
        
        if(self.loss_fn=="MSE"):
            total_loss = self.MSE()
        else:
            total_loss = self.CCE()

        if g.fitness > total_loss:
            g.fitness = total_loss
            if self.global_loss > total_loss:
                self.global_loss = total_loss#global_best
                self.global_weight = g.weight
                print("當前global_loss:%f"%(self.global_loss))
        else:
            g.fitness = total_loss

    def CCE(self):#y_pred : input_x, y_true : self.y[index]
        
        total_CE = 0
        prob = []
        for idx in range(len(self.y_pred)):
            true_units = self.y_true[idx]
            pred_units = self.y_pred[idx]
            for i_target, i_pred in zip(true_units, pred_units):#true:y, pred:x
                if i_target == 1:
                # required a proper smoothing operation
                    i_pred = i_pred if i_pred > 1e-7 else 1e-7
                    i_pred = i_pred if i_pred < 1 - 1e-7 else 1 - 1e-7
                    prob.append(i_pred)

        prob = np.array(prob, dtype='float32')
        prob_tensor = tf.constant(prob)
        log_tensor = tf.math.log(prob_tensor)
        loss = tf.reduce_sum(log_tensor).numpy()

        total_CE = -1 * loss / len(self.y_pred)
        return total_CE

    def MSE(self):#loss_function
        MSE = 0
        for i in range(len(self.y_pred)):
            for j in range(len(self.y_pred[i])):
                MSE += pow(self.y_true[i][j]-self.y_pred[i][j],2)
        
        MSE = MSE/len(self.x)
        return MSE
    #confuse_matrix
    def confuse(self):#0:TP,1:FN,2:FP,3:TN

        for i in range(len(self.y_pred)):
            predict =self.y_pred[i]
            actual = self.y_true[i]
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

    def calculation(self, g, data_index):#caculate every data loss with one gene
        input_y = self.y[data_index]
        input_x = np.array(self.x[data_index])
        predict = []
        exp_output = []
        sum_exp = 0
        unit_value = 0
        count = 0
        bias = 0
        for layer in self.layers:#forward propagation
            next_input_data = []     
            for i in range(layer):
                bias = g.weight[count]
                count+=1
                for j in range(len(input_x)):
                    unit_value += input_x[j]*g.weight[count]
                    count+=1
                unit_value += bias
                unit_value = self.sigmoid(unit_value)#activation:sigmoid
                next_input_data.append(unit_value)
            #預測結果
            input_x = np.array(next_input_data)
        for i in range(len(input_x)):
            exp_output.append(np.exp(input_x[i]))
            sum_exp +=exp_output[i]
        for i in range(len(exp_output)):
            predict.append(exp_output[i]/sum_exp)

        self.y_pred.append(predict)
        self.y_true.append(input_y)
    
    #test
    def _evaluate(self, test_x, test_y):
        self.y_pred = []
        self.y_true = []
        total_loss = 0
        self.x = test_x
        self.y = test_y
        for i in range(len(self.x)):
            self._calculation(i)#every data calculate loss
        
        if(self.loss_fn=="MSE"):
            total_loss = self.MSE()
        else:
            total_loss = self.CCE()

        print("test_loss:%f"%(total_loss))
        self.confuse()
    
    def _calculation(self, data_index):
        input_x = np.array(self.x[data_index])#one row data
        input_y = self.y[data_index]
        unit_value = 0
        predict = []
        exp_output = []
        sum_exp = 0
        count = 0
        bias = 0
        for layer in self.layers:#forward propagation 
            next_input_data = []     
            for i in range(layer):
                bias = self.global_weight[count]
                count+=1
                for j in range(len(input_x)):
                    unit_value += input_x[j]*self.global_weight[count]
                    count+=1
                unit_value += bias
                unit_value = self.sigmoid(unit_value)#activation:sigmoid
                next_input_data.append(unit_value)
            input_x = np.array(next_input_data)#最後有三個值
        #softmax
        for i in range(len(input_x)):
            exp_output.append(np.exp(input_x[i]))
            sum_exp +=exp_output[i]
        for i in range(len(exp_output)):
            predict.append(exp_output[i]/sum_exp)
        
        self.y_pred.append(predict)
        self.y_true.append(input_y)

def quicksort(loss, offspring, left, right):
    if left >= right:
        return
    i = left
    j = right
    key = loss[left]

    while i != j:                  
        while loss[j] > key and i < j:
            j -= 1
        while loss[i] <= key and i < j:
            i += 1
        if i < j:                       
            loss[i], loss[j] = loss[j], loss[i]
            offspring[i].weight, offspring[j].weight = offspring[j].weight, offspring[i].weight

    loss[left] = loss[i] 
    loss[i] = key
    offspring[left].weight = offspring[i].weight
    offspring[i].weight = offspring[left].weight

    quicksort(loss, offspring, left, i-1)
    quicksort(loss, offspring, i+1, right)




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
    total_global_loss = []
    total_accuracy = []
    total_precision = []
    total_recall = []
    total_f1score = []
    loss = 0
    accuracy = 0
    precision = 0
    recall = 0
    f1score = 0
    x=[]
    y=[]
    
    for file in files:
        print("訓練%d個檔案\n"%(count))
        x,y = data_process(train_path,file)

        #training_position = train_path + "/" + file 
        #df = pd.read_table(training_position, header=None)
        #建立模型
        model = _model(x, y, 30, loss_fn="MSE")#1000
        model.add_layer(6)
        model.add_layer(3)
        #model.fit
        
        model.fit(200)
        #得到train的結果
        loss, accuracy, precision, recall, f1score = model.calculate_confuse()
        #把檔案結果全部放到最後顯示
        total_global_loss.append(loss)
        total_accuracy.append(accuracy)
        total_precision.append(precision)
        total_recall.append(recall)
        total_f1score.append(f1score)
        model.reset_confuse()#model confuse matrix reset

        #testing部分
        test_file = file.replace("training","testing")
        x,y = data_process(test_path, test_file)
        #evaluate()
        model._evaluate(x,y)
        #得到test的結果
        loss, accuracy, precision, recall, f1score = model.calculate_confuse()
        #把檔案結果全部放到最後顯示
        total_global_loss.append(loss)
        total_accuracy.append(accuracy)
        total_precision.append(precision)
        total_recall.append(recall)
        total_f1score.append(f1score)
        count+=1
    count=1
    for i in range(len(total_global_loss)):
        if i%2 ==0:
            print("第%d個檔案"%(count))
            count+=1
            print("train\nglobal_loss%f, accuracy%f, precision%f, recall%f, f1score%f"%(total_global_loss[i], total_accuracy[i], total_precision[i], total_recall[i], total_f1score[i]))
        else:
            print("test\nglobal_loss%f, accuracy%f, precision%f, recall%f, f1score%f"%(total_global_loss[i], total_accuracy[i], total_precision[i], total_recall[i], total_f1score[i]))
