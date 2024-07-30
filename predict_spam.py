import tweepy
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
from sklearn.preprocessing import StandardScaler
import sys

# Twitter APIの認証情報を入力
API_KEY = 'your_api_key'
API_SECRET_KEY = 'your_api_secret_key'
ACCESS_TOKEN = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'
BEARER_TOKEN = 'your_bearer_token'

# 認証
client = tweepy.Client(bearer_token=BEARER_TOKEN, consumer_key=API_KEY, consumer_secret=API_SECRET_KEY,
                       access_token=ACCESS_TOKEN, access_token_secret=ACCESS_TOKEN_SECRET)


# アカウントデータ取得関数
def get_account_data(username):
    user = client.get_user(username=username, user_fields=['public_metrics', 'description'])
    user_id = user.data.id

    tweets = client.get_users_tweets(id=user_id, max_results=100)
    posts = [tweet.text for tweet in tweets.data]

    account_data = {
        "username": username,
        "posts": " ".join(posts),
        "followers": user.data.public_metrics['followers_count'],
        "following": user.data.public_metrics['following_count'],
        "profile_info": user.data.description
    }

    return account_data


# モデルとトークナイザーのロード
model = BertForSequenceClassification.from_pretrained('./spam_model')
tokenizer = BertTokenizer.from_pretrained('./spam_model')

# パイプラインの設定
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

# 正規化用のスケーラーの準備
scaler = StandardScaler()

if __name__ == "__main__":
    username = sys.argv[1]

    account_data = get_account_data(username)

    # 正規化のためのサンプルデータの読み込み
    sample_data = pd.read_csv('training_data.csv')
    scaler.fit(sample_data[['followers', 'following']])

    # 正規化
    df = pd.DataFrame([account_data])
    df[['followers', 'following']] = scaler.transform(df[['followers', 'following']])

    # テキストの結合
    account_text = " ".join([account_data['posts'], account_data['profile_info']])

    # スパム判定
    prediction = pipeline([account_text])

    # 判定結果の表示
    if prediction[0][0]['label'] == 'LABEL_1':
        print("このアカウントはスパムです。")
    else:
        print("このアカウントはスパムではありません。")