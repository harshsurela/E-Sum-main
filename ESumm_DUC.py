# import commands
import nltk
import string
import textmining
import os
import csv
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
import codecs
import knapsack

summary_length = 100.0 ## Number of words

    
## Removes Punctuation from text
def strip_punctuation(s):
        table = str.maketrans({key: None for key in string.punctuation})
        return s.translate(table)

## Preprocessing text
def preprocess(inputFile,f_name):

        ## Open and read text file
        file = open(inputFile, 'r')
        text = file.read()      

##        print(text)
        text = text.replace('\n',' ')
        
        text = text.replace("U.S.","US")
        text = text.replace("U.N.","UN")
        text = text.replace("Gov.","government")
        
        ## Counts number of words in the text after removing punctuations
        words_count = len(word_tokenize(strip_punctuation(text)))
        
        ## split text into sentences and store the sentences in a list
        sentences = tokenize.sent_tokenize(text)

        ## Original Sentences backup
        sentences_backup = list(sentences)
        lengths =[]
        for i in range(len(sentences_backup)):
                lengths.append(len(word_tokenize(strip_punctuation(sentences_backup[i].replace('\n',' ')))))
                
##        print("Sentence Lengths",lengths)
        return(words_count,len(sentences_backup),sentences)

## Function for NMF decomposition of term-sentence matrix A
def find_WH(inputFile,f_name,k1):
        
        ## Reads term-sentence frequency matrix from .csv file and stores in matrix A
        termSentFile = ".\\Pre_Processed\\"+f_name.replace('.txt','')+".csv"
        data = np.genfromtxt(termSentFile, dtype='float64', delimiter=',', names=None)
        A = np.asarray(data,dtype = 'float64')
                        
        ## Convert A into binary-incidence matrix
        for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                        if(A[i][j]!=0):
                                A[i][j]=1.0

        ## Calls method preprocess
        [words_count,no_of_sentences,sentences] = preprocess(inputFile,f_name)

        ## average sentence length = total words in the document/ number of sentences
        avg_sentence_length = math.ceil(words_count/no_of_sentences)

        ## Number of latent topics = summary length (100 words for DUC data-sets)/ average sentence length
        k2 = math.ceil(summary_length/avg_sentence_length)

        k = k1
        
        if(k<=3):
                k= k2

        if(k>no_of_sentences):
                k=no_of_sentences

        model = NMF(n_components=k, init='nndsvd',max_iter=4000)
        W = model.fit_transform(A)
        H= model.components_
        np.set_printoptions(suppress=True)
        return(W,H)

## function to compute log base 2
def log2(x):
        if(x==0):
                return 0
        else:
                return math.log(x,2)


## function to compute topic entropy using H matrix
## Rows in H matrix represent topics
def top_ent_H(H):
        
        ## find sum of elements of each row in H
        H_sum_rows = np.sum(H,axis=1)

        ## variable for storing probability distribution of topics in sentences
        topic_prob = np.zeros((H.shape[0],H.shape[1]),dtype = 'float64')

        ## Compute the probability distribution of topics in sentences
        for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                        if(H_sum_rows[i]!=0):
                                topic_prob[i][j]=H[i][j]/H_sum_rows[i]

        self_info = np.zeros((H.shape[0],H.shape[1]),dtype = 'float64')
        for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                        self_info[i][j]=-log2(topic_prob[i][j])

        
        ## variable for topics entropy
        topic_entropy = np.zeros(H.shape[0],dtype = 'float64')

        ## Compute topic entropy
        for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                        topic_entropy[i] -= topic_prob[i][j]*log2(topic_prob[i][j])

        np.set_printoptions(suppress=True)
##        print("Topic Entropy",topic_entropy)
        
        return topic_entropy


## function to compute sentence entropy using H matrix
## Columns in H matrix represent sentences
def sent_ent_H(H):

        ## find sum of elements of each column in H
        H_sum_cols = np.sum(H,axis=0)

        ## variable for storing probability distribution of sentences in topics
        sent_prob = np.zeros((H.shape[0],H.shape[1]),dtype = 'float64')

        ## Compute the probability distribution of sentences in topics
        for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                        if(H_sum_cols[j]!=0):
                                sent_prob[i][j]=H[i][j]/H_sum_cols[j]

        ## variable for sentences entropy
        sent_entropy = np.zeros(H.shape[1],dtype = 'float64')

        ## Compute sentence entropy
        for i in range(H.shape[1]):
                for j in range(H.shape[0]):
                        sent_entropy[i] -= sent_prob[j][i]*log2(sent_prob[j][i])

        np.set_printoptions(suppress=True)
##        print("Sentence Entropy",sent_entropy)
        return sent_entropy


def E_Summ(H, sentence_entropy, topic_entropy,sentences):
        
        sentence_score = np.zeros(len(sentences), dtype = 'float64')

        ## Rank topics - Rank 1 means topic of highest entropy
        x = topic_entropy.argsort()
        temp_ranks = np.zeros(len(topic_entropy),dtype = int)
        
        for j in range(len(topic_entropy)):
                temp_ranks[x[j]]=len(topic_entropy)-j

        ranks_list = temp_ranks.tolist()
        
        for i in range(1,len(topic_entropy)+1):                
                j = ranks_list.index(i)
                selected_index = np.argmax(H[j,:])
                H[j][selected_index]=0.0
                sentence_score[selected_index] = sentence_entropy[selected_index]+topic_entropy[j]

        indices =[]
        scores=[]
        lengths=[]

        for i in range(len(sentence_score)):
                if(sentence_score[i]!=0):
                        indices.append(i)
                        scores.append(sentence_score[i])
                        lengths.append(len(word_tokenize(strip_punctuation(sentences[i]))))

        capacity = summary_length
        [a,selected]=knapsack.knapsack(lengths, scores).solve(capacity)
        
        selected_sentences=[]
        for i in range(len(selected)):
                selected_sentences.append(indices[selected[i]])
        for j in range(len(selected)):
                indices[selected[j]]=-1
                scores[selected[j]] =-1
##        print("knapsack sentences",selected_sentences)
        return indices, scores, lengths, selected_sentences
                       
def write_summary(indices, scores, lengths, selected_sentences,f_name,original_sentences):
        words = 0
        for i in range(len(selected_sentences)):
                words += len(word_tokenize(strip_punctuation(sentences[selected_sentences[i]].replace('\n',' '))))
                
        while(words<summary_length):
                max_score_sent = np.argmax(scores)
                if(scores[max_score_sent]<=0 or words>=summary_length):
                        break
                selected_sentences.append(indices[max_score_sent])
                words+= len(word_tokenize(strip_punctuation(sentences[indices[max_score_sent]].replace('\n',' '))))

                indices[max_score_sent]=-1
                scores[max_score_sent]=-1
               
##        print(selected_sentences)
        sorted_sentences = np.sort(selected_sentences)
        name = ".\\Summaries\\"+"system3_"+f_name
        file_object = codecs.open(name,"w",encoding="utf-8")
        
        for i in range(len(selected_sentences)):
                file_object.write(sentences[sorted_sentences[i]].replace('\n',' '))
                file_object.write('\n')
        file_object.close()
        

os.chdir("H:\WORKSPACE\Python\E-Summ-main")
termSentFile = ".\\Sample Doc_terms.csv"
data = np.genfromtxt(termSentFile, dtype='int32', delimiter=',', names=None)
topics = np.asarray(data,dtype = 'int32')

i=0
for f in os.listdir(".\\Documents"):
    inputFile = ".\\Documents\\"+f
    
    [words_count,no_of_sentences,sentences]=preprocess(inputFile,f)
    [W,H] = find_WH(inputFile,f,topics[i])
    top_ent=top_ent_H(H)
    sent_ent = sent_ent_H(H)
    [indices, scores,lengths,selected_sentences] = E_Summ(H, sent_ent, top_ent,sentences)
    write_summary(indices, scores,lengths,selected_sentences,f,sentences)
    
    i+=1

