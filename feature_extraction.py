import pickle
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from pyspark.sql import SparkSession
import time

# Paths to JSON data files
TRAIN_DATA = "data/train_data_162k.json"
TEST_DATA = "data/test_data_162k.json"
VAL_DATA = "data/val_data_162k.json"

# Function to load BERT model and tokenizer
def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
    return v_phobert, v_tokenizer

# Load BERT model and tokenizer
phobert, tokenizer = load_bert()
print("Load model done!")

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Feature Extraction") \
    .master("local[*]") \
    .config("spark.executor.memory", "50g") \
    .config("spark.executor.instances", "4") \
    .config("spark.executor.cores", "12") \
    .config("spark.memory.offHeap.enabled", True) \
    .config("spark.driver.memory", "50g") \
    .config("spark.memory.offHeap.size", "16g") \
    .config("spark.ui.showConsoleProgress", False) \
    .config("spark.driver.maxResultSize", "16g") \
    .config("spark.log.level", "ERROR") \
    .getOrCreate()

# Load JSON data into Spark DataFrames
train_data = spark.read.json(TRAIN_DATA)
test_data = spark.read.json(TEST_DATA)
val_data = spark.read.json(VAL_DATA)
print("Load data done!")

# Function to extract BERT features from text
def make_bert_features(v_text):
    v_tokenized = []
    max_len = 256  # Maximum sequence length

    # Use tqdm to display progress
    for i_text in v_text:
        # Tokenize using BERT tokenizer
        line = tokenizer.encode(i_text, truncation=True)
        v_tokenized.append(line)

    # Pad sequences to ensure consistent length
    padded = []
    for i in v_tokenized:
        if len(i) < max_len:
            padded.append(i + [1] * (max_len - len(i)))  # Padding with 1s
        else:
            padded.append(i[:max_len])  # Truncate if sequence is too long

    padded = np.array(padded)

    # Create attention mask
    attention_mask = np.where(padded == 1, 0, 1)

    # Convert to PyTorch tensors
    padded = torch.tensor(padded).to(torch.long)
    attention_mask = torch.tensor(attention_mask)

    # Obtain features from BERT
    with torch.no_grad():
        last_hidden_states = phobert(input_ids=padded, attention_mask=attention_mask)

    v_features = last_hidden_states[0][:, 0, :].numpy()
    print(v_features.shape)
    return v_features

# Extract BERT features for train, test, and validation datasets
train_features = train_data.select("processed_content").rdd.map(make_bert_features)
test_features = test_data.select("processed_content").rdd.map(make_bert_features)
val_features = val_data.select("processed_content").rdd.map(make_bert_features)

# Convert category column to lists
category_list_train = train_data.select("category").rdd.flatMap(lambda x: x).collect()
category_list_test = test_data.select("category").rdd.flatMap(lambda x: x).collect()
category_list_val = val_data.select("category").rdd.flatMap(lambda x: x).collect()

# Convert to one-hot encoding using pd.get_dummies()
y_train = pd.get_dummies(category_list_train)
y_test = pd.get_dummies(category_list_test)
y_val = pd.get_dummies(category_list_val)

# Save data to file using pickle
start_time = time.time()
print("Saving to file")
data_dict = {
    'X_train': train_features.collect(),
    'X_test': test_features.collect(),
    'X_val': val_features.collect(),
    'y_train': y_train,
    'y_test': y_test,
    'y_val': y_val
}

# Save dictionary to pickle file
with open('data/features_162k_phobertbase_v2.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

end_time = time.time()
duration = end_time - start_time
print(f'Total feature extraction time: {duration:.2f} seconds')
print("Done!")
