import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from spacy.lang.es import Spanish
import webbrowser
from PIL import Image
from collections import Counter


def main():
    img=Image.open('Img/bank1.jpg')
    st.image(img,width=600)
    st.title("Twitter Sentiment Analysis on Bank")
    img2 = Image.open('Img/twitter_logo.png')
    st.sidebar.image(img2,use_column_width=True)

    st.sidebar.title("Twitter Sentiment Analysis on Bank")


    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("Data/DataBankV8.csv")
        return data

    data = load_data()

    bank_group=data.groupby(['tweet_id', 'name', 'Tweet_Content', 'Tweet_Website', 'Tweet_Number_of_Likes', \
                      'Tweet_Number_of_Retweets', 'Tweet_Number_of_share', 'datetime', 'stars_t']) \
        .agg({'Comment_Content': 'count', 'stars_r': 'mean'}) \
        .rename(columns={'Comment_Content': 'count_replies', 'sent': 'mean_stars'}) \
        .reset_index()


    def follow(x):
        followers = {'Bankia': 31300, 'Santander': 77200, 'Banco Sabadell': 49200, 'BBVA': 2500, 'ING España': 44000,
                     'CaixaBank': 54000}
        for key, value in followers.items():
            if x == key:
                return value

    bank_group['followers'] = bank_group['name'].apply(follow)

    def eng_rate(x):
        i = ((x['Tweet_Number_of_Likes'] + x['Tweet_Number_of_Retweets']) / x[
            'followers']) * 100
        return i

    bank_group['eng_rate'] = bank_group.apply(eng_rate, axis=1).round(2)
    bank_group['year_month'] = bank_group['datetime'].values.astype('datetime64[M]')

    bank_s = st.multiselect('Bank Selection:',
                            ('Bankia', 'CaixaBank', 'Banco Sabadell', 'BBVA', 'Santander',
       'ING España'))
    data = data[data['name'].isin(bank_s)]
    bank_group=bank_group[bank_group['name'].isin(bank_s)]

    #st.write(bank_group)

    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    start_date = st.date_input('Start date', datetime.date(2020, 1, 1))
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = st.date_input('End date', tomorrow)
    end_date = end_date.strftime("%Y-%m-%d")

    stars = st.slider('Sentiment replies rate', 1, 5, 3)
    st.write(stars, 'stars')
    data_s = data[data['stars_r']==stars]
    data_t=data[data['stars_t']==stars]


    # Tweets count historic
    st.sidebar.subheader("Tweets count per month")
    # select = st.sidebar.selectbox("Visualization Type", ["Bar Plot"])
    if st.sidebar.checkbox("Show", False, key='0'):

        if start_date < end_date:
            # st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
            tweets_count = data_s[['name', 'tweet_id', 'year_month', 'Tweet_Content']].groupby(
                ['tweet_id','year_month','name']).agg(['nunique']).reset_index(drop=False)
            tweets_count2 = tweets_count[
                (tweets_count['year_month'] >= start_date) & (tweets_count['year_month'] < end_date)]
            fig = px.bar(tweets_count2, x="year_month", y="Tweet_Content", color="name", barmode='group')
            st.subheader('Tweets count per month')
            st.plotly_chart(fig)

    #Tweets replies count historic
    st.sidebar.subheader("Tweets total replies per month")
    if st.sidebar.checkbox("Show", False, key='1'):

        if start_date < end_date:
            #st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
            tweets_count = data_s[['name', 'tweet_id', 'year_month', 'Comment_Content']].groupby(
                ['tweet_id', 'name', 'year_month']).count().reset_index()
            tweets_count2=tweets_count[(tweets_count['year_month'] >= start_date) & (tweets_count['year_month'] < end_date)]
            fig = px.bar(tweets_count2, x="year_month", y="Comment_Content", color="name")
            st.subheader('Tweets replies per month')
            st.plotly_chart(fig)

    #eng_rate
    st.sidebar.subheader("Engagement Rate")
    if st.sidebar.checkbox("Show", False, key='2'):
        if start_date < end_date:
            bank_group = bank_group[['name', 'tweet_id', 'year_month', 'Tweet_Number_of_Likes', 'Tweet_Number_of_Retweets',
                         'eng_rate']].groupby(
                ['name', 'year_month']).mean().reset_index()
        # tweets_count=tweets_count[(tweets_count['year_month'] >= start_date) & (tweets_count['year_month'] < end_date)]
        bank_group = bank_group[(bank_group['year_month'] >= start_date) & (bank_group['year_month'] < end_date)]
        fig = px.line(bank_group, x="year_month", y="eng_rate", color="name")
        st.subheader('Engagement Rate')
        st.plotly_chart(fig)

    #Keywords
    st.sidebar.subheader("Tweets Wordcloud")
    if st.sidebar.checkbox("Show", False, key='3'):
        if start_date < end_date:

            def spacy_tokenizer(sentence):
                parser = Spanish()
                tokens = parser(sentence)
                filtered_tokens = []
                for word in tokens:
                    lemma = word.lemma_.lower().strip()
                    if lemma not in STOP_WORDS and re.search('^[a-zA-Z]+$', lemma):
                        filtered_tokens.append(lemma)

                return filtered_tokens

            data_s['Tweet_Content_Token'] = data_s['Tweet_Content'].apply(spacy_tokenizer)
            all_words = ' '.join([text for i in data_s['Tweet_Content_Token'] for text in i])
            wordcloud = WordCloud().generate(all_words)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.subheader('Tweets Wordcloud')
            st.pyplot()

    # Keywords Replies
    st.sidebar.subheader("Tweets Replies Wordcloud")
    if st.sidebar.checkbox("Show", False, key='4'):
        if start_date < end_date:

            def spacy_tokenizer(sentence):
                parser = Spanish()
                tokens = parser(sentence)
                filtered_tokens = []
                for word in tokens:
                    lemma = word.lemma_.lower().strip()
                    if lemma not in STOP_WORDS and re.search('^[a-zA-Z]+$', lemma):
                        filtered_tokens.append(lemma)

                return filtered_tokens

            data_s = data_s[data_s['Comment_Content']!='undefined']
            data_s['Tweet_Replies_Token'] = data_s['Comment_Content'].apply(spacy_tokenizer)
            all_words = ' '.join([text for i in data_s['Tweet_Replies_Token'] for text in i])
            wordcloud = WordCloud().generate(all_words)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.subheader('Tweets Replies Wordcloud')
            st.pyplot()



    #Hashtags
    st.sidebar.subheader("Hashtags")
    if st.sidebar.checkbox("Show", False, key='5'):
        if start_date < end_date:
            data_t = data_t.replace('nan', np.nan)
            data_tc = data_t['hashtag'].dropna()
            d = Counter(data_tc)
            keys = []
            values = []
            for w in sorted(d, key=d.get, reverse=True):
                keys.append(w)
                values.append(d[w])
            df = pd.DataFrame(list(zip(keys, values)),
                              columns=['Hashtags', 'Count'])
            df = df[:20]
            fig = px.bar(df, x='Hashtags', y='Count', color='Count',
                         labels={'pop': 'population of Canada'}, height=400)
            st.subheader('Hashtags')
            st.plotly_chart(fig)




    #Tweets Sentiment
    st.sidebar.subheader("Tweets sentiment")
    if st.sidebar.checkbox("Show", False, key='6'):
        if start_date < end_date:
            stars_df=data[['name', 'labels_r', 'year_month', 'stars_r']].groupby(
            ['labels_r', 'name', 'year_month']).count().reset_index()
            fig = go.Figure(
                data=[go.Pie(labels=stars_df['labels_r'], values=stars_df['stars_r'], hole=.6)])
            st.subheader('Tweet replies stars proportion')
            st.plotly_chart(fig)


    # Show random tweet
    st.sidebar.subheader("Sample Reply Tweet Sentiment")
    random_tweet = st.sidebar.radio("Sentiment", ("1 star", "2 stars", "3 stars","4 stars", "5 stars"))
    if st.sidebar.checkbox("Show", False, key='7'):
        st.subheader(f"Sample {random_tweet.capitalize()} Tweet&Reply")
        data_f=data[data['Author_Name'].notna()]
        s=data_f.query("labels_r == @random_tweet and Comment_Content != 'undefined'").sample(n=1)
        st.header(f"Name:   {s['name'].item()}")
        st.subheader(f"Tweet:")
        st.write(f"{s['Tweet_Content'].item()}")
        st.write(f"Number of likes: {s['Tweet_Number_of_Likes'].item()} "
                 f"Number of retweets: {s['Tweet_Number_of_Retweets'].item()}")
        url = s['Tweet_Website']
        st.markdown(url.item())
        st.subheader(f"Reply:")
        st.write(f"{s['Comment_Content'].item()}")
        if st.button('Change'):
            webbrowser.open_new_tab(url)



        #st.header(data.query("labels_r == @random_tweet")[["name"]].sample(n=1).iat[0, 0])

    # Number of tweets by sentiment

if __name__ == "__main__":
    main()
