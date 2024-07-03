import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging

import warnings
import time
import pickle
warnings.filterwarnings("ignore")

logging.set_verbosity_error()

# Function to set seed for reproducibility
def seed_everything(seed_value):
    np.random.seed(seed_value)  # Set seed for numpy random numbers
    torch.manual_seed(seed_value)  # Set seed for PyTorch random numbers

    if torch.cuda.is_available():  # If CUDA is available, set CUDA seed
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = True  # Improve performance by allowing cudnn benchmarking

seed_everything(86)  # Set seed value for reproducibility

model_name = "bluenguyen/longformer-phobert-base-4096"  # Pretrained model name
max_len = 512  # Maximum sequence length for tokenizer (512, but can use 256 if phobertbase)
n_classes = 13  # Number of output classes
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # Load tokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, otherwise CPU
EPOCHS = 5  # Number of training epochs
N_SPLITS = 5  # Number of folds for cross-validation

TRAIN_PATH = "data/train_data_162k.json"  
TEST_PATH = "data/test_data_162k.json"  
VAL_PATH = "data/val_data_162k.json"  

# Function to read data from JSON file
def get_data(path):
    df = pd.read_json(path, lines=True)
    return df

# Read the data from JSON files
train_df = get_data(TRAIN_PATH)
test_df = get_data(TEST_PATH)
valid_df = get_data(VAL_PATH)

# Combine train and validation data
train_df = pd.concat([train_df, valid_df], ignore_index=True)

# Apply StratifiedKFold
skf = StratifiedKFold(n_splits=N_SPLITS)
for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.category)):
    train_df.loc[val_, "kfold"] = fold

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        To customize dataset, inherit from Dataset class and implement
        __len__ & __getitem__
        __getitem__ should return
            data:
                input_ids
                attention_masks
                text
                targets
        """
        row = self.df.iloc[index]
        text, label = self.get_input_data(row)

        # Encode_plus will:
        # (1) split text into token
        # (2) Add the '[CLS]' and '[SEP]' token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map token to their IDS
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_masks': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long),
        }


    def labelencoder(self, text):
        label_map = {
            'Cong nghe': 0, 'Doi song': 1, 'Giai tri': 2, 'Giao duc': 3, 'Khoa hoc': 4,
            'Kinh te': 5, 'Nha dat': 6, 'Phap luat': 7, 'The gioi': 8, 'The thao': 9,
            'Van hoa': 10, 'Xa hoi': 11, 'Xe co': 12
        }
        return label_map.get(text, -1)

    def get_input_data(self, row):
        text = row['processed_content']
        label = self.labelencoder(row['category'])
        return text, label

class NewsClassifier(nn.Module):
    def __init__(self, n_classes, model_name):
        super(NewsClassifier, self).__init__()
        # Load a pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        # Dropout layer to prevent overfitting
        self.drop = nn.Dropout(p=0.3)
        # Fully-connected layer to convert BERT's hidden state to the number of classes to predict
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        # Initialize weights and biases of the fully-connected layer using the normal distribution method
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        # Get the output from the BERT model
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        # Apply dropout
        x = self.drop(output)
        # Pass through the fully-connected layer to get predictions
        x = self.fc(x)
        return x

def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = NewsDataset(df_train, tokenizer, max_len)
    valid_dataset = NewsDataset(df_valid, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=2)

    return train_loader, valid_loader

# Function to train the model for one epoch
def train(model, criterion, optimizer, train_loader, lr_scheduler):
    model.train()  # Set the model to training mode
    losses = []  # List to store losses during training
    correct = 0  # Variable to store number of correct predictions

    # Iterate over batches in the training data loader
    for batch_idx, data in enumerate(train_loader):
        input_ids = data['input_ids'].to(device)  # Move input_ids to GPU/CPU
        attention_mask = data['attention_masks'].to(device)  # Move attention_mask to GPU/CPU
        targets = data['targets'].to(device)  # Move targets to GPU/CPU

        optimizer.zero_grad()  # Clear gradients from previous iteration
        outputs = model(  # Forward pass through the model
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, targets)  # Calculate the loss
        _, pred = torch.max(outputs, dim=1)  # Get the predicted labels

        correct += torch.sum(pred == targets)  # Count correct predictions
        losses.append(loss.item())  # Append the current loss value to losses list
        loss.backward()  # Backpropagation: compute gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to prevent exploding gradients
        optimizer.step()  # Update model parameters
        lr_scheduler.step()  # Update learning rate scheduler

        # Print training progress every 1000 batches
        if batch_idx % 1000 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}, Accuracy: {correct.double() / ((batch_idx + 1) * train_loader.batch_size):.4f}')

    train_accuracy = correct.double() / len(train_loader.dataset)  # Calculate training accuracy
    avg_loss = np.mean(losses)  # Calculate average loss
    print(f'Train Accuracy: {train_accuracy:.4f} Loss: {avg_loss:.4f}')

# Function to evaluate the model
def eval(model, criterion, valid_loader, test_loader=None):
    model.eval()  # Set the model to evaluation mode
    losses = []  # List to store losses during evaluation
    correct = 0  # Variable to store number of correct predictions

    with torch.no_grad():  # Disable gradient calculation for evaluation
        data_loader = test_loader if test_loader else valid_loader  # Choose between test and validation data loader
        for batch_idx, data in enumerate(data_loader):
            input_ids = data['input_ids'].to(device)  # Move input_ids to GPU/CPU
            attention_mask = data['attention_masks'].to(device)  # Move attention_mask to GPU/CPU
            targets = data['targets'].to(device)  # Move targets to GPU/CPU

            outputs = model(  # Forward pass through the model
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(outputs, targets)  # Calculate the loss
            _, pred = torch.max(outputs, dim=1)  # Get the predicted labels

            correct += torch.sum(pred == targets)  # Count correct predictions
            losses.append(loss.item())  # Append the current loss value to losses list

    dataset_size = len(test_loader.dataset) if test_loader else len(valid_loader.dataset)  # Determine dataset size
    accuracy = correct.double() / dataset_size  # Calculate accuracy
    avg_loss = np.mean(losses)  # Calculate average loss

    # Print evaluation results (either test or validation)
    if test_loader:
        print(f'Test Accuracy: {accuracy:.4f} Loss: {avg_loss:.4f}')
    else:
        print(f'Valid Accuracy: {accuracy:.4f} Loss: {avg_loss:.4f}')

    return accuracy  # Return accuracy for further analysis or logging

total_start_time = time.time()

# Main training loop
for fold in range(skf.n_splits):
    print(f'----------- Fold: {fold + 1} ------------------')
    train_loader, valid_loader = prepare_loaders(train_df, fold=fold)
    model = NewsClassifier(n_classes=13).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * EPOCHS
    )
    best_acc = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 30)

        train(model, criterion, optimizer, train_loader, lr_scheduler)
        val_acc = eval(model, criterion, valid_loader)

        if val_acc > best_acc:
            torch.save(model.state_dict(), f'phobert_fold{fold + 1}.pth')
            best_acc = val_acc
        print(f'Best Accuracy for Fold {fold + 1}: {best_acc:.4f}')
        print()
    print(f'Finished Fold {fold + 1} with Best Accuracy: {best_acc:.4f}')
    print('--------------------------------------')


total_end_time = time.time()

total_duration = total_end_time - total_start_time
print(f'Total training time: {total_duration:.2f} seconds')