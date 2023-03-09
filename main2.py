# Databricks notebook source
!pip install -q -U watermark

# COMMAND ----------

!pip install -qq transformers

# COMMAND ----------

# MAGIC %reload_ext watermark
# MAGIC %watermark -v -p numpy,pandas,torch,transformers

# COMMAND ----------

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# COMMAND ----------

## load the data

df= pd.read_csv("reviews.csv")
df.shape

# COMMAND ----------

df.info()

# COMMAND ----------

df.groupby(df.score).size()

# COMMAND ----------

# no missing data which is good from the information part, what about class imbalance?

# get the value counts for each category
counts = df['score'].value_counts().sort_index()

# create a bar plot
counts.plot(kind='bar')

# set the title and axis labels
plt.title('Count by Score')
plt.xlabel('Scpre')
plt.ylabel('Count')

# show the plot
plt.show()


# COMMAND ----------

## apparently it is very much imbalanced, but i am going to convert the data to negative, neutral and positive sentiment
def to_sentiment(rating):
  rating = int(rating)
  if rating <= 2:
    return 0
  elif rating == 3:
    return 1
  else:
    return 2

df['sentiment'] = df.score.apply(to_sentiment)



# COMMAND ----------

new_names = {0: 'negative', 1: 'neutral', 2: 'positive'}

# get the value counts for each category
counts = df['sentiment'].value_counts().rename(new_names)

# create a bar plot
counts.plot(kind='bar')

# set the title and axis labels
plt.title('Count by Score')
plt.xlabel('Scpre')
plt.ylabel('Count')

# show the plot
plt.show()

# COMMAND ----------

### apprently it is getting more balanced


# COMMAND ----------

# specify the model 
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# COMMAND ----------

### testing testing 
sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'

#### conver the text to tokens
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')

# COMMAND ----------

# special tokens
tokenizer.sep_token, tokenizer.sep_token_id

# COMMAND ----------

# we must add this token to the start of each sentence, so that BERT knows i am doing classification
tokenizer.cls_token, tokenizer.cls_token_id

# COMMAND ----------

# special token for padding
tokenizer.pad_token, tokenizer.pad_token_id

# COMMAND ----------

#BERT understands tokens that were in the training set. 
# Everything else can be encoded using the [UNK] (unknown) token:
tokenizer.unk_token, tokenizer.unk_token_id

# COMMAND ----------

### all these work can be done in encode_plus
encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
)

encoding.keys()

# COMMAND ----------

# the token now stored in a T   ensor and padded to a lenth of 32
print(len(encoding['input_ids'][0]))
encoding['input_ids'][0]

# COMMAND ----------

# the attention mask has same length 
print(len(encoding['attention_mask'][0]))
encoding['attention_mask']

# COMMAND ----------

# we can inverse the tokenization to have a look at the special tokens
tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

# COMMAND ----------

### chosing sequence length
#BERT works with fixed-length sequences. 
# We'll use a simple strategy to choose the max length. 
# Let's store the token length of each review:

token_lens = []

for txt in df.content:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))

# COMMAND ----------

sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count');

# COMMAND ----------

#Most of the reviews seem to contain less than 128 tokens, 
# but we'll be on the safe side and choose a maximum length of 160.
MAX_LEN=160

# COMMAND ----------

class GPReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

# COMMAND ----------

df_train, df_test = train_test_split(
  df,
  test_size=0.1,
  random_state=RANDOM_SEED
)

df_val, df_test = train_test_split(
  df_test,
  test_size=0.5,
  random_state=RANDOM_SEED
)

# COMMAND ----------

df_train.shape, df_val.shape, df_test.shape

# COMMAND ----------

# we need to create data loaders  -- the below is helper function

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=df.content.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

# COMMAND ----------

BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# COMMAND ----------

## look at the example batch from the training data loader

data = next(iter(train_data_loader))
data.keys()

# COMMAND ----------

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

# COMMAND ----------

 

# COMMAND ----------

### finally modelling
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

# COMMAND ----------

last_hidden_state, pooled_output = bert_model(
  input_ids=encoding['input_ids'],
  attention_mask=encoding['attention_mask']
)
