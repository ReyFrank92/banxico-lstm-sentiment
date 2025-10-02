# -*- coding: utf-8 -*-
# Script exportado desde el cuaderno banxico_lstm_sentiment_clean.ipynb

!pip install nltk

import os, random
import numpy as np
import tensorflow as tf

def set_seed(seed: int = 13):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(2)   # ← cambia 13 por 7, 21, etc. para probar otras semillas

import nltk

from pathlib import Path
import pandas as pd

carpeta = Path(r"C:\Users\Scarl\Documents\CFA\Curso\Python Data Science and AI\Unit 4\Practice ensayo")

txt_files = sorted(carpeta.glob("*.txt"))
print("TXT encontrados:")
for i, p in enumerate(txt_files):
    print(f"[{i}] {p.name}")

idx_a_usar = 0  # <-- cambia este número si quieres otro archivo
path = txt_files[idx_a_usar]
print("\nLeyendo:", path)

def leer_txt_robusto(p: Path):
    errores = []
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(p, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError as e:
            errores.append((enc, "UnicodeDecodeError"))
        except pd.errors.ParserError as e:
            for sep in ["\t", ",", ";", "|"]:
                try:
                    return pd.read_csv(p, sep=sep, engine="python", encoding=enc)
                except Exception:
                    continue
            errores.append((enc, "ParserError"))
        except Exception as e:
            errores.append((enc, repr(e)))
    raise RuntimeError(f"No se pudo leer el archivo. Errores: {errores}")

df_txt = leer_txt_robusto(path)

pd.set_option("display.max_colwidth", 160)
print("Shape:", df_txt.shape)
print("Columnas:", list(df_txt.columns))
df_txt.head(10)

df = df_txt[["sentences", "stance_label"]].rename(
    columns={"sentences": "Text", "stance_label": "Label"}
)

df.head(10)

df.shape

df['Label'].value_counts()

df.info()

import re

import nltk
from nltk.corpus import stopwords

_ = nltk.download('stopwords', quiet=True)  # silencia mensajes de descarga
EN_STOPWORDS = set(stopwords.words('english'))  # no imprime al asignar
print(f"Stopwords cargadas: {len(EN_STOPWORDS)}")  # confirmación breve



def clean_text(text):


    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = text.lower()

    stop_words = set(stopwords.words('english'))
    stop_words.add('heshe')  # <- añadir tu stopword personalizada (en minúsculas)
    words = text.split()
    words = [word for word in words if word not in stop_words]

    cleaned_text = ' '.join(words)

    return cleaned_text

df['Cleaned Text'] = df['Text'].apply(clean_text)

df['Label'] = df['Label'].str.replace(r'^\s*irrelevant\s*$', 'neutral', case=False, regex=True)

df

df.to_excel("df.xlsx", index=False)


!pip install wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def generate_word_cloud(text):
    
    custom_stopwords = {"http", "china", "us", "united states", "political", "politics","stock","stocks", "trump"}

    stopwords = set(STOPWORDS)
    stopwords.update(custom_stopwords)

    wordcloud = WordCloud(width = 1600, height = 800, stopwords = stopwords, min_font_size = 10).generate(text)

    plt.figure(figsize = (12, 12))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

generate_word_cloud(" ".join(df[df['Label'] == 'hawkish']['Cleaned Text']))

generate_word_cloud(" ".join(df[df['Label'] == 'neutral']['Cleaned Text']))

generate_word_cloud(" ".join(df[df['Label'] == 'dovish']['Cleaned Text']))


!pip install transformers

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

import torch
from torch.nn.utils.rnn import pad_sequence

def tokenization_padding(df):

    df['Encoded Text'] = [tokenizer.encode(news) for news in df['Cleaned Text'].tolist()]

    encoded_news_tensor = [torch.tensor(encoded_news) for encoded_news in df['Encoded Text'].tolist()]
    padded_sequence = pad_sequence(encoded_news_tensor, batch_first = True, padding_value = 0).numpy()

    return padded_sequence

X = tokenization_padding(df)
X

X.shape

df

df['Encoded Label'] = df['Label'].replace('hawkish', 0).replace('dovish', 1).replace('neutral', 2)

y = df['Encoded Label'] 
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, shuffle = True)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, shuffle = True)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

print(y_train.shape)
print(y_test.shape)
print(y_val.shape)



import tensorflow as tf
from tensorflow import keras

import tensorflow as tf

inputs = tf.keras.Input(shape=(X_train.shape[1],), dtype="int32")

x = tf.keras.layers.Embedding(
    input_dim=tokenizer.vocab_size,
    output_dim=128,
    mask_zero=True
)(inputs)

x = tf.keras.layers.SpatialDropout1D(0.25)(x)

x = tf.keras.layers.LSTM(
    32, return_sequences=True, activation='tanh', dropout=0.20
)(x)


gmax = tf.keras.layers.GlobalMaxPooling1D()(x)
gavg = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Concatenate()([gmax, gavg])   # ← nuevo pooling combinado


outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

from tensorflow.keras.optimizers import Adam

model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['accuracy'])

model.summary()

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


history = model.fit(X_train,
                    y_train,
                    validation_data = (X_val, y_val),
                    batch_size = 64, #32 original
                    verbose = 1,
                    epochs = 15)

results = model.evaluate(X_test, y_test)

print("Test Accuracy: {:.2f}%".format(results[1] * 100))

predictions = model.predict(X_test)

import numpy as np

y_predict = []

for i in predictions:
  y_predict.append(np.argmax(i))


y_test

import seaborn as sns

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, fmt = 'd', annot = True)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

