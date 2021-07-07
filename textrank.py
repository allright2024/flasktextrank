import json
import os
import sys
import warnings 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns 

from newspaper import Article
from konlpy.tag import Kkma
from konlpy.tag import Twitter 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import normalize 
import numpy as np

class SentenceTokenizer(object):
    def __init__(self):
        self.kkma=Kkma()
        self.twitter=Twitter()
        self.stopwords=['중인' ,'만큼', '마찬가지', '꼬집었', "연합뉴스", "데일리", "동아일보", "중앙일보", "조선일보", "기자"
,"아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가",
]
    def url2sentences(self, url):
        article = Article(url, language='ko')
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)

        for idx in range(0,len(sentences)):
            if len(sentences[idx])<=10:
                sentences[idx-1]+=(' '+sentences[idx])
                sentences[idx]=''

        return sentences 

    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx])<=10:
                sentences[idx-1]+=(' '+sentences[idx])
                sentences[idx]=''
        return sentences

    def get_nouns(self, sentences):
        nouns=[]
        for sentence in sentences:
            if sentence != ' ':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence))
                                      if noun not in self.stopwords and len(noun)>1]))
        return nouns 

class GraphMatrix(object):
    def __init__(self):
        self.tfidf=TfidfVectorizer()
        self.cnt_vec=CountVectorizer()
        self.graph_sentence=[]

    def build_sent_graph(self, sentence):
        tfidf_mat=self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence=np.dot(tfidf_mat,tfidf_mat.T)
        return self.graph_sentence
    
    def build_words_graph(self, sentence):
        cnt_vec_mat=normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab=self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word]:word for word in vocab}#dic 형태의 {idx, word} 리턴

class Rank(object):
    def get_ranks(self,graph, d=0.85):
        A=graph
        matrix_size=A.shape[0]
        for id in range(matrix_size):
            A[id:id]=0
            link_sum=np.sum(A[:,id])
            if link_sum!=0:
                A[:,id]/=link_sum
        pr=list(range(0,matrix_size))
        
        for iter in range(100):
            pr=0.15+0.85*np.dot(A,pr)
    
        
        #pr2=[int (i) for i in pr]
        
        dictionary = {i:pr[i] for i in range(len(pr))}
        
        return dictionary  
        
class TextRank(object):
    def __init__(self,text):
        self.sent_tokenize=SentenceTokenizer()
        
        if text[:5] in ('http:', 'https'): #text로 url을 주는 경우
            self.sentences = self.sent_tokenize.url2sentences(text)
        else: #text로 text를 주는 경우
            self.sentences = self.sent_tokenize.text2sentences(text)
            
        self.nouns = self.sent_tokenize.get_nouns(self.sentences) #문장단위로 나눠준 self.sentences에서 명사만을 추출하기
        
        self.graph_matrix = GraphMatrix() # graph_matrix 생성자 (?)
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns) #tfidf 매트릭스 만들기 
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns) #cnt_vet_mat의 내적을 words_graph에, dic형태의 {idx, word}를 idx2word에 할당
        
        
        self.rank=Rank()
        self.sent_rank_idx=self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx=sorted(self.sent_rank_idx, key=lambda k:self.sent_rank_idx[k], reverse=True)
        
        self.word_rank_idx=self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx=sorted(self.word_rank_idx, key=lambda k:self.word_rank_idx[k], reverse=True)
        
    def summarize(self,sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        
        
        index.sort()
        
        print(index)
        for idx in index:
            summary.append(self.sentences[idx])

       
                
        return summary
    
    def summarizeidx(self,sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        
        
        index.sort()
        
        return index
        
    def keywords(self, word_num=10):
        rank=Rank()
        rank_idx=rank.get_ranks(self.words_graph)
        sorted_rank_idx=sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
            
        keywords=[]
        index=[]
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)
            
        
        for idx in index:
            keywords.append(self.idx2word[idx])
                
        return keywords
'''def newssum(url):
    textrank=TextRank(url)
    sum=[]
    for row in textrank.summarize(3):
        sum+=row+'\n'
    
    return sum'''

def newssum(url):
    textrank=TextRank(url)
    sum=textrank.summarize(3)
    return sum