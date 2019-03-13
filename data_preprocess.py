"""
数据预处理
BOME标记
ns 地点
nt 组织机构
nr 人名
"""
import re
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


def corpusHandle():
    path1="./MSRA/train1.txt"
    path2="./MSRA/word2tag.txt"
    with open(path1,'r',encoding='utf8') as inp,open(path2,'w',encoding='utf8') as outp:
        for line in inp:
            line=line.strip().split()
            if len(line)==0:
                continue
            for word in line:
                word=word.split('/')
                """如果词性为ns/nt/nr，则为每个字添加BME标记，否则添加O标记"""
                if word[1]!='o':
                    if len(word[0])==1:
                        outp.write(word[0]+'/B_'+word[1]+' ')
                    elif len(word[0])==2:
                        outp.write(word[0][1]+"/B_"+word[1]+" ")
                        outp.write(word[0][1]+"/E_"+word[1]+" ")
                    else:
                        outp.write(word[0][0]+"/B_"+word[1]+" ")
                        for j in word[0][1:-1]:
                            outp.write(j+"/M_"+word[1]+" ")
                        outp.write(word[0][-1]+"/E_"+word[1]+" ")
                else:
                    for j in word[0]:
                        outp.write(j+"/o"+" ")
            outp.write("\n")

def sentence2split():
    path1="./MSRA/word2tag.txt"
    path2="./MSRA/train2.txt"
    with open(path1,'r',encoding="utf8") as inp,open(path2,'w',encoding='utf8') as outp:
        texts=inp.read()
        sentences=re.split("[，。！？、‘’“”：；]/[o]",texts)
        for sentence in sentences:
            if sentence!=" ":
                outp.write(sentence.strip()+'\n')


def save():
    path="./MSRA/train2.txt"
    datas=[]
    labels=[]
    with open(path,'r',encoding='utf8') as inp:
        for line in inp:
            line=line.strip().split()
            linedata = []
            linelabel = []
            num_not_o=0
            for word in line:
                word=word.split('/')
                if len(word[0])!=0:
                    linedata.append(word[0])
                    linelabel.append(word[1])
                    if word[1]!='o':
                        num_not_o+=1
            if num_not_o!=0:
                datas.append(linedata)
                labels.append(linelabel)

    #构建字词典和标签词典
    all_words=[word for line in datas for word in line]
    all_words=pd.Series(all_words)
    word_count=all_words.value_counts()
    set_words=word_count.index
    set_ids=range(1,len(set_words)+1)
    id2word=pd.Series(set_words,index=set_ids)
    word2id=pd.Series(set_ids,index=set_words)
    word2id['unknow']=len(word2id)+1
    id2word[len(word2id)]='unknow'

    tag2id = {'': 0,
              'B_ns': 1,
              'B_nr': 2,
              'B_nt': 3,
              'M_nt': 4,
              'M_nr': 5,
              'M_ns': 6,
              'E_nt': 7,
              'E_nr': 8,
              'E_ns': 9,
              'o': 10}
    id2tag = {0: '',
              1: 'B_ns',
              2: 'B_nr',
              3: 'B_nt',
              4: 'M_nt',
              5: 'M_nr',
              6: 'M_ns',
              7: 'E_nt',
              8: 'E_nr',
              9: 'E_ns',
              10: 'o'}

    df_data=pd.DataFrame({"word":datas,"tag":labels},index=range(len(datas)))

    max_len=50
    def X_padding(words):
        ids=list(word2id[words])
        if len(ids)>max_len:
            ids=ids[:max_len] #长则舍弃
        else:
            ids.extend([0]*(max_len-len(ids))) #短则补全
        return ids

    def y_padding(labels):
        ids=[]
        for label in labels:
            ids.append(tag2id[label])
        if len(ids)>max_len:
            return ids[:max_len]
        ids.extend([0]*(max_len-len(labels)))
        return ids

    df_data['x']=df_data['word'].apply(X_padding)
    df_data['y']=df_data['tag'].apply(y_padding)

    x=np.asarray(list(df_data['x'].values))
    y=np.asarray(list(df_data['y'].values))
    print(x.shape)
    print(y)

    #划分训练集、测试集、验证集
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2019)
    x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.2,random_state=2019)

    print("数据处理完毕，开始保存>>>>")
    with open("./dataMSRA.pkl",'wb') as outp:
        pickle.dump(word2id,outp)
        pickle.dump(id2word,outp)
        pickle.dump(tag2id,outp)
        pickle.dump(id2tag,outp)
        pickle.dump(x_train,outp)
        pickle.dump(y_train,outp)
        pickle.dump(x_test,outp)
        pickle.dump(y_test,outp)
        pickle.dump(x_valid,outp)
        pickle.dump(y_valid,outp)

    print("<<<<数据保存完毕....")



if __name__ == '__main__':
    #将词粒度变为字粒度，并标记
    # corpusHandle()
    #划分句子
    # sentence2split()
    save()