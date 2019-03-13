#构建BiLSTM+CRF网络
import numpy as np
import tensorflow as tf

class Model():
    def __init__(self,lr,batch_size,vocab_size,embedding_dim,max_len,
                 n_classes,pretrained,embedding_pretrained):
        self.lr=lr #学习率大小
        self.batch_size=batch_size #一个epoch参与训练样本数
        self.vocab_size=vocab_size #词典大小
        self.embedding_dim=embedding_dim #词向量维度
        self.max_len=max_len #序列长度
        self.n_classes=n_classes #标签个数
        self.pretrained=pretrained #是否预训练词向量嵌入
        self.embedding_pretrained=embedding_pretrained #预训练的词向量

        self.input_data=tf.placeholder(tf.int32,shape=[self.batch_size,self.max_len],name='input_data')
        self.labels=tf.placeholder(tf.int32,shape=[self.batch_size,self.max_len],name='labels')
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        with tf.name_scope("bilstm_crf"):
            self._build_net()

    def _build_net(self):
        word_embedding=tf.get_variable(shape=[self.vocab_size,self.embedding_dim],dtype=tf.float32,name="word_embedding")
        if self.pretrained:
            embedding_init=word_embedding.assign(self.embedding_pretrained) #将词向量指定为embedding_pretrained
        input_embed=tf.nn.embedding_lookup(word_embedding,self.input_data)
        input_embed=tf.nn.dropout(input_embed,keep_prob=self.keep_prob) #输入增加dropout

        lstm_cell_fw=tf.nn.rnn_cell.LSTMCell(self.embedding_dim,forget_bias=1.0,state_is_tuple=True)
        lstm_cell_bw=tf.nn.rnn_cell.LSTMCell(self.embedding_dim,forget_bias=1.0,state_is_tuple=True)
        (output_fw,output_bw),states=tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,lstm_cell_bw,input_embed,dtype=tf.float32)
        # print('output_fw', output_fw.shape) #(128, 50, 100)
        # print('output_bw', output_bw.shape) #(128, 50, 100)
        bilstm_out=tf.concat([output_fw,output_bw],-1)
        # print('bilstm_out', bilstm_out.shape) #(128, 50, 200)

        w=tf.get_variable(name='w',shape=[self.batch_size,2*self.embedding_dim,self.n_classes],initializer=tf.random_normal_initializer,dtype=tf.float32)
        b=tf.get_variable(name='b',shape=[self.n_classes],initializer=tf.zeros_initializer,dtype=tf.float32)

        bilstm_out=tf.tanh(tf.matmul(bilstm_out,w)+b)

        #线性CRF层
        log_likelihood,self.transition_param=tf.contrib.crf.crf_log_likelihood(bilstm_out,self.labels,tf.tile(np.array([self.max_len]),np.array([self.batch_size])))
        #log_likelihood 是真实路径的概率取对数，损失值则加负号
        loss=tf.reduce_mean(-log_likelihood)

        #解码
        self.viterbi_sequence,viterbi_score=tf.contrib.crf.crf_decode(bilstm_out,self.transition_param,tf.tile(np.array([self.max_len]),np.array([self.batch_size])))

        optimizer=tf.train.AdamOptimizer(self.lr)
        self.train_op=optimizer.minimize(loss)


