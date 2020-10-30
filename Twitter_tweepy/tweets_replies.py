import tweepy
import pandas as pd

CONSUMER_KEY = 'xxxxxxxxxxxxxxxxxxxxxxx'
CONSUMER_SECRET = 'xxxxxxxxxxxxxxxxxxxxxx'
ACCESS_TOKEN = 'xxxxxxxxxxxxxxxxxxxxx'
ACCESS_SECRET = 'xxxxxxxxxxxxxxxxxxxx'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)

###Tweets###

username = '@caixabank'
count = 5000

try:
    # Creation of query method using parameters
    tweets = tweepy.Cursor(api.user_timeline, id=username).items(count)

    tweets_list = [
        [tweet.created_at, tweet.id, tweet.text, tweet.user.name, tweet.entities, tweet.in_reply_to_user_id_str,
         tweet.in_reply_to_status_id_str, tweet.favorite_count, tweet.retweet_count] for tweet in tweets]

    # Creation of dataframe from tweets list
    # Add or remove columns as you remove tweet information
    tweets_df = pd.DataFrame(tweets_list)
except BaseException as e:
    print('failed on_status,', str(e))
    time.sleep(3)

tweets_df.columns=['datetime','tweet_id','Tweet_Content','name','entities','reply_user_id','reply_status_id','Tweet_Number_of_Likes','Tweet_Number_of_Retweets']

tweets_df.to_csv(r'Data/CaixaV1.csv', index = False)

###Tweets_replies###

lista=tweets_df.tolist()

lista

lista=[str(i) for i in lista]

name ='@caixabank'
tweet_id = lista

replies=[]
for tweet in tweepy.Cursor(api.search,q='to:'+name).items(150):
    for i in lista:
        if hasattr(tweet, 'in_reply_to_status_id_str'):
            if (tweet.in_reply_to_status_id_str==i):
                replies.append(tweet._json)

replies=pd.DataFrame(replies)

replies.to_csv(r'Data/caixa twitter replies.csv', index = False)