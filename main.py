import pickle
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from bilstm_crf import Model


def load_data():
    path="./dataMSRA.pkl"
    with open(path,'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
        print('word2id:',len(word2id))
        print('tag2id:',len(tag2id))
        print('训练集:',x_train.shape)
        print('测试集:',x_test.shape)
        print('验证集:',x_valid.shape)
    return word2id,id2word,tag2id,id2tag,x_train,y_train,x_test,y_test,x_valid,y_valid

def embedding_pretrained(word2id):
    path="./vec.txt"
    word_embedding={}
    with open(path,'r',encoding='utf8') as inp:
        for line in inp:
            line=line.strip().split()
            word_embedding[line[0]]=list(map(float,line[1:]))
    #未知词向量
    unknow_pre=[]
    unknow_pre.extend([1.0]*100)
    pre_embedding=[]
    pre_embedding.append(unknow_pre)

    for word in word2id.index:
        if word in word_embedding.keys():
            pre_embedding.append(word_embedding[word])
        else:
            pre_embedding.append(unknow_pre)
    pre_embedding=np.asarray(pre_embedding,dtype=np.float32)
    print("pre_embedding:",pre_embedding.shape)
    return pre_embedding

def get_batch(X,y,batch_size=64,shuffle=True):
    if shuffle:
        shuffle_index=np.random.permutation(X.shape[0])
        X=X[shuffle_index]
        y=y[shuffle_index]
    batch_num=int(X.shape[0]/batch_size)
    batch=[]
    for i in range(batch_num):
        x_batch=X[i*batch_size:(i+1)*batch_size]
        y_batch=y[i*batch_size:(i+1)*batch_size]
        one_batch=(x_batch,y_batch)
        batch.append(one_batch)
    if X.shape[0]%batch_size!=0:
        x_batch=X[X.shape[0]-batch_size:X.shape[0]]
        y_batch=y[X.shape[0]-batch_size:X.shape[0]]
        one_batch = (x_batch, y_batch)
        batch.append(one_batch)
    return batch

def caculate(x,y,id2word,id2tag,res=[]):
    entity=[]
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j]==0 or y[i][j]==0:
                continue
            if id2tag[y[i][j]][0]=='B':
                entity.append(id2word[x[i][j]]+"/"+id2tag[y[i][j]])
            elif id2tag[y[i][j]][0]=="M" and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
            elif id2tag[y[i][j]][0]=='E' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]]+'/'+id2tag[y[i][j]])
                entity.append(str(i))
                entity.append(str(j))
                res.append(entity)
                entity=[]
            else:
                entity=[]
    return res

def metrics(y_true,y_pred):
    jiaoji = [i for i in y_pred if i in y_true]
    if len(jiaoji) != 0:
        p = float(len(jiaoji)) / len(y_pred)  # 精确率，预测正确的样本占预测总样本的比例
        r = float(len(jiaoji)) / len(y_true)  # 召回率，预测正确的样本占真正为正样本的比例
        f=2 * p * r/(r + p)
    else:
        p=0
        r=0
        f=0
    return p,r,f



def train(x_train,y_train,x_valid,y_valid,config,pre_embedding,id2word,id2tag,keep_prob=0.5):
    lr=config["lr"]
    batch_size=config["batch_size"]
    vocab_size=config["vocab_size"]
    embedding_dim=config["embedding_dim"]
    max_len=config["max_len"]
    n_classes=config["n_classes"]
    pretrained=config["pretrained"]
    epochs=config["epochs"]
    model=Model(lr,batch_size,vocab_size,embedding_dim,max_len,
                 n_classes,pretrained,pre_embedding)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        for epoch in range(epochs):
            j=0
            t_p,t_r,t_f=0,0,0
            for batch_x,batch_y in get_batch(x_train,y_train,batch_size=batch_size):
                j+=1
                feed_dict={model.input_data:batch_x,
                           model.labels:batch_y,
                           model.keep_prob:keep_prob}
                pre,_=sess.run([model.viterbi_sequence,model.train_op],feed_dict=feed_dict)
                print("当前epoch:{},当前循环:{}".format(epoch,j))
                if epoch%3==0:
                    pred=caculate(batch_x,pre,id2word,id2tag)
                    res=caculate(batch_x,batch_y,id2word,id2tag)
                    p, r, f = metrics(res, pred)
                    t_p+=p
                    t_r+=r
                    t_f+=f
            # 计算精确率、召回率、F1值
            print("训练集：精确率={:.5f}，召回率={:.5f}，f1={:.5f}".format(t_p/j, t_r/j, t_f/j))

            if epoch%3==0:
                path_name = "./model/model" + str(epoch) + ".ckpt"
                saver.save(sess,path_name)
                print("model has been saved!!!\n")
                feed_dict={model.input_data:x_valid,
                           model.labels:y_valid,
                           model.keep_prob:1.0}
                pre=sess.run(model.viterbi_sequence,feed_dict=feed_dict)

                entitypre=caculate(x_valid,pre[0],id2word,id2tag) #预测结果
                entityres=caculate(x_valid,y_valid,id2word,id2tag) #真实结果
                #计算精确率、召回率、F1值
                p,r,f=metrics(entityres,entitypre)
                print("验证集：精确率={:.5f}，召回率={:.5f}，f1={:.5f}".format(p,r,f))


def test_input():
    print("开始测试>>>>")
    model=Model()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        ckpt=tf.train.get_checkpoint_state("./model")
        if ckpt==None:
            print("模型未找到，请先训练....")
            return None
        else:
            path=ckpt.model_checkpoint_path
            saver.restore(sess,path)

        for batch_x,batch_y in get_batch(x_test,y_test,batch_size=batch_size):
            feed_dict={model.input_data:batch_x,
                       model.labels:batch_y,
                       model.keep_prob:1.0}
            pre=sess.run(model.viterbi_sequence,feed_dict=feed_dict)











if __name__ == '__main__':
    print("开始加载数据>>>\n")
    word2id, id2word, tag2id, id2tag, x_train, y_train, x_test, y_test, x_valid, y_valid=load_data()
    print("\n开始加载词向量>>>\n")
    pre_embedding=embedding_pretrained(word2id)

    #定义超参数
    config = {}
    config["lr"] = 0.001  # 学习率大小为0.001
    config["embedding_dim"] = 100  # 词向量维度为100
    config["max_len"] = len(x_train[0])  # 序列长度
    config["batch_size"] = 128  # 参与训练样本数
    config["vocab_size"] = len(word2id) + 1  # 词嵌入大小
    config["n_classes"] = len(tag2id)  # 标记集合的大小
    config["epochs"]=20 #迭代次数
    config["pretrained"] = True  # 是否预训练
    config["test"]=True

    print("\n开始训练>>>>\n")
    train(x_train,y_train,x_valid,y_valid,config,pre_embedding,id2word,id2tag,keep_prob=0.5)