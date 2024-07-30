import tweepy
import pandas as pd
import sys

# Twitter APIの認証情報を入力
API_KEY = 'API-KEY'
API_SECRET_KEY = 'API-SECRET-KEY'
ACCESS_TOKEN = 'ACCESS-TOKEN'
ACCESS_TOKEN_SECRET = 'ACCESS-TOKEN-SECRET'
BEARER_TOKEN = 'BEARER'

# 認証
client = tweepy.Client(bearer_token=BEARER_TOKEN, consumer_key=API_KEY, consumer_secret=API_SECRET_KEY,
                       access_token=ACCESS_TOKEN, access_token_secret=ACCESS_TOKEN_SECRET)


# アカウントデータ取得関数
def get_account_data(username, label):
    user = client.get_user(username=username, user_fields=['public_metrics', 'description'])
    user_id = user.data.id

    tweets = client.get_users_tweets(id=user_id, max_results=100)
    posts = [tweet.text for tweet in tweets.data]

    account_data = {
        "username": username,
        "posts": " ".join(posts),
        "followers": user.data.public_metrics['followers_count'],
        "following": user.data.public_metrics['following_count'],
        "profile_info": user.data.description,
        "label": label
    }

    return account_data


if __name__ == "__main__":
    username = sys.argv[1]
    label = int(sys.argv[2])

    account_data = get_account_data(username, label)

    df = pd.DataFrame([account_data])

    # CSVファイルに保存
    df.to_csv('training_data.csv', mode='a', header=not pd.read_csv('training_data.csv').empty, index=False)