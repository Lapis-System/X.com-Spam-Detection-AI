import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# トークナイザーの初期化
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class AccountDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.encodings = tokenizer(texts['posts'].tolist(), texts['profile_info'].tolist(), truncation=True,
                                   padding=True, max_length=512)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # データの読み込み
    data = pd.read_csv('training_data.csv')

    # 数値データの正規化
    scaler = StandardScaler()
    data[['followers', 'following']] = scaler.fit_transform(data[['followers', 'following']])

    # トレーニングデータとテストデータに分割
    train_texts, val_texts, train_labels, val_labels = train_test_split(data[['posts', 'profile_info']], data['label'],
                                                                        test_size=0.2)

    # データセットの準備
    train_dataset = AccountDataset(train_texts, train_labels)
    val_dataset = AccountDataset(val_texts, val_labels)

    # モデルの初期化
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # トレーニングパラメータの設定
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # モデルのトレーニング
    trainer.train()

    # モデルの保存
    model.save_pretrained('./spam_model')
    tokenizer.save_pretrained('./spam_model')
