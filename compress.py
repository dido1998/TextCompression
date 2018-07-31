import tensorflow as tf
import tensorflow_hub as hub
import re
import nltk
import pickle
from tqdm import tqdm
import numpy as np
import  os
import tensorflow_hub as hub
import random
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


minibatch_size=100
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map
words_to_index, index_to_words, word_to_vec_map = read_glove_vecs('content/glove.6B.300d.txt')
"""
The pretrained glove vectors are obtained as follows
wget http://nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip -d content
"""

with open('dict.pickle','rb') as f :
	dic1=pickle.load(f)
dic1['<start>']=len(dic1)
e_t=np.zeros([minibatch_size])
s_t=np.zeros([minibatch_size])
for q in range(minibatch_size):
	s_t[q]=len(dic1)-1
vocab_size=len(dic1)
#print(vocab_size)
embed_matrix1 = np.zeros((vocab_size, 300))
for word,index in dic1.items():
  try:
    embed_matrix1[index, :] = word_to_vec_map[word]
  except:
    embed_matrix1[index, :] = np.random.uniform(-1, 1, 300)
str_placeholder=tf.placeholder(tf.string,shape=[None])
inputs=tf.placeholder(tf.int32,shape=[None,None])
sq=tf.placeholder(tf.int32,shape=[None])
with tf.name_scope('embed'):
	with tf.device('/device:GPU:0'):
		W = tf.get_variable(name = 'W', shape = embed_matrix1.shape, initializer = tf.constant_initializer(embed_matrix1), trainable = True)
		embeddings_out = tf.nn.embedding_lookup(W, inputs)
#Iam using the tensorflow universal senetence encoder for encoding the sentence
with tf.device('/device:CPU:0'):
	module_url = "https://tfhub.dev/google/universal-sentence-encoder/1" 
	embed = hub.Module(module_url)
	emb = embed(str_placeholder)
keep_prob=tf.placeholder(tf.float32)
labels=tf.placeholder(tf.int32,shape=[None,None])
starttoken=tf.placeholder(tf.int32,shape=[None])
endtoken=tf.constant(0)
training=tf.placeholder(tf.int32)
weights=tf.placeholder(tf.float32,shape=[None,None])
#For each sentence the universal sentence encoder gives a 512 size vector representation.
#this representation will be further compressed to 5 size vector representation
#the 5 size vector representation will be expanded to 512 dimensions to reproduce the original embedding
#the new 512 size vectors are provided to the decoder RNN as the initial hidden state
reduce_emb1=tf.layers.dense(emb,256,activation=tf.nn.relu)
reduce_emb2=tf.layers.dense(reduce_emb1,5,activation=tf.nn.relu)
increase_emb1=tf.layers.dense(reduce_emb2,256,activation=tf.nn.relu)
increase_emb3=tf.layers.dense(increase_emb1,512,activation=tf.nn.relu)

state=tf.contrib.rnn.LSTMStateTuple(increase_emb3,increase_emb3)
state=tuple((state,state,state,state))
train_helper = tf.contrib.seq2seq.TrainingHelper(embeddings_out, sq)

pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(W, start_tokens=starttoken, end_token=endtoken)

def decode(helper, scope,state, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        cell =[tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell (size,activation=tf.nn.relu),state_keep_prob=keep_prob) for size in [512,512,512,512]]  
        cell = tf.nn.rnn_cell.MultiRNNCell(cell, state_is_tuple=True)
       	out_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.OutputProjectionWrapper(
                cell,1024,activation=tf.nn.relu,reuse=reuse),output_keep_prob=keep_prob)
       	out_cell =  tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.OutputProjectionWrapper(
                out_cell,1024,activation=tf.nn.relu,reuse=reuse),output_keep_prob=keep_prob)
       	out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                out_cell,vocab_size,activation=None,reuse=reuse)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=out_cell, helper=helper,
            initial_state=state)
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,impute_finished=True, output_time_major=False,maximum_iterations=40
             )
        return outputs[0]
train_outputs = decode(train_helper, 'decode',state)
pred_outputs = decode(pred_helper, 'decode',state, reuse=True)
train_outputs=train_outputs[0]

pred_outputs=pred_outputs[0]
#the total loss is a sum of two losses
# 1-regression loss between increase_emb3 and emb
# 2-negative log likelihood for the decoder output
loss_reg=tf.multiply(tf.subtract(increase_emb3,emb),tf.subtract(increase_emb3,emb))
loss_reg=tf.reduce_mean(loss_reg,1)
loss_reg=tf.reduce_mean(loss_reg)
loss=tf.contrib.seq2seq.sequence_loss(logits=train_outputs,targets=labels,weights=weights)
loss=tf.add(tf.multiply(1.0,loss),tf.multiply(1.0,loss_reg))
optimize=tf.train.AdamOptimizer(0.0005)
grads=optimize.compute_gradients(loss)
opt=optimize.apply_gradients(grads)

saver=tf.train.Saver()
def clean_str(string):
    string = string.strip().lower()

    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r',', '', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    if 'let\'s' in string:
        string = re.sub(r'let\'s', 'let us', string)
    if 'lets' in string:
        string = re.sub(r'lets', 'let us', string)

    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'ve", " have", string)
    if 'wont ' in string:
        string = re.sub(r"won\'?t", "will not", string)
    if 'won\'t ' in string:
        string = re.sub(r"won\'?t", "will not", string)

    if 'cant ' in string:
        string = re.sub(r"n\'?t", " can not", string)
    if 'can\'t ' in string:
        string = re.sub(r"n\'?t", " can not", string)

    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    # string = re.sub(r"\'", '', string)

    return string.strip()
def preprocess(sentence):
	sentence = sentence.lower()
	sentence=clean_str(sentence)

	tokenizer = nltk.RegexpTokenizer(r'\w+')

	tokens = tokenizer.tokenize(sentence)
	m=len(tokens)
	tokens=" ".join(tokens)
	return m,tokens


f=open('image_coco.txt','r')
text=f.read();

sentence=text.split('\n')
psent=[]
"""
###This code is used for creating the dictionary and writing it to dict.pickle
dic={}
dic['<end>']=0
for s in sentence:
	wds=s.split()
	for w in wds:
		if w not in dic:
			dic[w]=len(dic)
with open('dict.pickle','wb') as f:
	pickle.dump(dic,f)
"""

max=0
for s in sentence:
	m,sent=preprocess(s)
	if m>max:
		max=m
	psent.append(sent)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
os.mkdir('compmodel')
with tf.Session(config=config) as sess:
	#saver.restore(sess,'')
	sess.run(tf.global_variables_initializer())
	
	for u in range(1,5000):
		l=0
		for i in tqdm(range(int(10000/minibatch_size))):
			cur_ip=psent[i*minibatch_size:i*minibatch_size+minibatch_size]
			mkx=0
			for c in cur_ip:
				h11=c.split()
				if len(h11)>mkx:
					mkx=len(h11)
			lbs=np.zeros([minibatch_size,mkx+1])
			inps=np.zeros([minibatch_size,mkx+1])
			ws=np.zeros([minibatch_size,mkx+1])
			for x in range(minibatch_size):
				inps[x,0]=dic1['<start>']
				ws[x,0]=1

			c1=0

			sqlen=np.zeros([minibatch_size])
			for c in cur_ip:
				h=c.split()
				c2=1
				c3=0
				sqlen[c1]=len(h)+1
				for j in h:
					ws[c1,c2]=1
			
					lbs[c1,c3]=dic1[j]
					inps[c1,c2]=dic1[j]
					c2+=1
					c3+=1
				c1+=1

			ls,_=sess.run([loss,opt],feed_dict={sq:sqlen,str_placeholder:cur_ip,keep_prob:0.5,weights:ws,starttoken:np.array([len(dic1)-1]),training:1,inputs:inps,labels:lbs})
			l+=ls
		q=random.randint(1,10000)
		d=[psent[q]]
	
		f=sess.run(pred_outputs,feed_dict={str_placeholder:d,starttoken:np.array([len(dic1)-1]),keep_prob:1.0})
		a1=f[0,:,:]
		print(d)

		print(a1)
		print('---------------')
		choice=[i for i in range(vocab_size)]
		choice=np.array(choice)
		preds=''
		for i in range(a1.shape[0]):
			w1=np.argmax(a1[i,:])
			
			for q,t in dic1.items():
			
				if t==w1:
					preds=preds+" "+q
		print(preds)
		cur_ip=[]

		l=l/(10000/minibatch_size)

		print(str(u)+':'+str(l))
		if u%20==0:
			try:
				os.mkdir('compmodel_'+str(u))
			except:
				pass
			saver.save(sess,'compmodel_'+str(u)+'/dec_model')

