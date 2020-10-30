import re
import pandas as pd
import numpy as np
from transformers import pipeline
classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')



df_1=pd.read_csv('BankiaV1.csv')
df_2=pd.read_csv('BBVAV2.csv')
df_3=pd.read_csv('SantanderV1.csv')
df_4=pd.read_csv('INGV1.csv')
df_5=pd.read_csv('SabadellV1.csv')
df_6=pd.read_csv('CaixaV1.csv')

tweets_df = pd.concat([df_1,df_2,df_3,df_4,df_5,df_6],axis=0 ,join='outer').sort_index()

tweets_df
tweets_df['Tweet_Content']=tweets_df['Tweet_Content'].astype(str)
tweets_df = tweets_df[tweets_df['tweet_id'].notna()]


def cleantxt (text):
  text=re.sub(r'@','',text)
  return text

tweets_df['Tweet_Content']=tweets_df['Tweet_Content'].apply(cleantxt)

import ast

def dict_trans(x):
    return ast.literal_eval(x)

tweets_df['entities']=tweets_df['entities'].apply(dict_trans)

tweets_df=tweets_df.reset_index(drop=True)

s=list(tweets_df['Tweet_Content'].replace(np.nan, '', regex=True))

label_1=[]
for i in s:
  label_1.append(i.get('label'))

score_1=[]
for i in s:
  score_1.append(i.get('score'))

tweets_df['labels_t'] = pd.Series((label_1), index=tweets_df.index)
tweets_df['score_t'] = pd.Series((score_1), index=tweets_df.index)
tweets_df['datetime']=tweets_df['datetime'].values.astype('datetime64[M]')

tweets_df=tweets_df.reset_index(drop=True)

def rem_stars(x):
    m=re.sub('stars|star','',x)
    return int(m)

tweets_df['stars_t']=tweets_df['labels_t'].apply(rem_stars)

def hashtag(x):
    n=x.get('hashtags')
    for i in n:
        for key,value in i.items():
            return value

tweets_df['hashtag']=tweets_df['entities'].apply(hashtag)


def split_url(x):
    n=x.get('urls')
    for i in n:
        for key,value in i.items():
            if key == 'expanded_url':
                return value

def del_url(x):
    return re.sub(r'http\S+', '', x)

tweets_df['Comment_Content']=tweets_df['Comment_Content'].astype(str)
tweets_df['labels_r']=tweets_df['labels_r'].astype(str)
tweets_df['score_r']=tweets_df['score_r'].astype(float)

tweets_df.to_csv('Data/DataBankV8', index=False)



