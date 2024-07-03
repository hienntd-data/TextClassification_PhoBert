# Streamlit
import streamlit as st 
import os
import pandas as pd
import pickle
import json
# Preprocessing
import re
import phonlp
import underthesea
import re

# Visualize
import numpy as np


# Model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

# Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up the Streamlit page
st.set_page_config(layout='wide')

# Define variables
PREPROCESSED_DATA = "data/val_data_162k.json"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS)

# Define class names
class_names = ['Cong nghe', 'Doi song', 'Giai tri', 'Giao duc', 'Khoa hoc', 'Kinh te',
               'Nha dat', 'Phap luat', 'The gioi', 'The thao', 'Van hoa', 'Xa hoi', 'Xe co']

# Define the NewsClassifier class for BERT-based models
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

@st.cache_data
def load_models(model_type):
    models = None
    model = None

    if model_type == 'phobertbase':
        models = []
        for fold in range(skf.n_splits):
            model = NewsClassifier(n_classes=13, model_name='vinai/phobert-base-v2')
            model.to(device)
            model.load_state_dict(torch.load(f'models/phobert_256_fold{fold+1}.pth', map_location=device))
            model.eval()
            models.append(model)
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        max_len = 256
    elif model_type == 'longformer':
        models = []
        for fold in range(skf.n_splits):
            model = NewsClassifier(n_classes=13, model_name='bluenguyen/longformer-phobert-base-4096')
            model.to(device)
            model.load_state_dict(torch.load(f'models/phobert_fold{fold+1}.pth', map_location=device))
            model.eval()
            models.append(model)
        tokenizer = AutoTokenizer.from_pretrained("bluenguyen/longformer-phobert-base-4096")
        max_len = 512
    elif model_type == 'bilstm_phobertbase':
        model = load_model("models/bilstm_phobertbase.h5", compile=False)
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
	phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        max_len = 256
    else:
        raise ValueError("Invalid model type specified.")
    
    if models is not None:
        return models, tokenizer, max_len
    else:
        return model, tokenizer, max_len, phobert



# Function to preprocess text using PhonLP and Underthesea
def preprocess_text(text):
    nlp_model = phonlp.load(save_dir="./phonlp")
    text = re.sub(r'[^\w\s.]', '', text)
    sentences = underthesea.sent_tokenize(text)
    preprocessed_words = []
    for sentence in sentences:
        try:
            word_tokens = underthesea.word_tokenize(sentence, format="text")
            tags = nlp_model.annotate(word_tokens, batch_size=64)
            filtered_words = [word.lower() for word, tag in zip(tags[0][0], tags[1][0]) if tag[0] not in ['M', 'X', 'CH'] 
                              and word not in ["'", ","]]
            preprocessed_words.extend(filtered_words)
        except Exception as e:
            pass
    return ' '.join(preprocessed_words)

# Function to tokenize text using BERT tokenizer
def tokenize_text(text, tokenizer, max_len=256):
    tokenized = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )
    return tokenized['input_ids'], tokenized['attention_mask']
def get_vector_embedding(padded, attention_mask, phobert):
    # Obtain features from BERT
    with torch.no_grad():
        last_hidden_states = phobert(input_ids=padded, attention_mask=attention_mask)

    v_features = last_hidden_states[0][:, 0, :].numpy()
    return v_features
# Function to get BERT features
def get_bert_features(input_ids, attention_mask, phobert):
    with torch.no_grad():
        last_hidden_states = phobert(input_ids=input_ids, attention_mask=attention_mask)
    features = last_hidden_states[0][:, 0, :].numpy()
    return features

# Function to predict label using BiLSTM model
def predict_label(text, tokenizer, phobert, model, class_names, max_len):
    processed_text = preprocess_text(text)
    input_ids, attention_mask = tokenize_text(processed_text, tokenizer, max_len)
    input_ids = torch.tensor(input_ids).to(torch.long).to(device)
    attention_mask = torch.tensor(attention_mask).to(torch.long).to(device)

    with torch.no_grad():
        features = get_bert_features(input_ids, attention_mask, phobert)
        features = np.expand_dims(features, axis=1)
        prediction = model.predict(features)
    
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_names[predicted_label_index]

    confidence_scores = {class_names[i]: float(prediction[0][i]) for i in range(len(prediction[0]))}
    confidence_df = pd.DataFrame([confidence_scores])
    confidence_df = confidence_df.melt(var_name='Label', value_name='Confidence')

    return predicted_label, confidence_df

# Function to infer predictions using ensemble of BERT-based models
def infer(text, tokenizer, models, class_names, max_len):
    tokenized = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    with torch.no_grad():
        all_outputs = []
        for model in models:
            model.eval()
            output = model(input_ids, attention_mask)
            all_outputs.append(output)

        all_outputs = torch.stack(all_outputs)
        mean_output = all_outputs.mean(0)
        _, predicted = torch.max(mean_output, dim=1)

    confidence_scores = mean_output.softmax(dim=1).cpu().numpy()
    confidence_df = pd.DataFrame([confidence_scores[0]], columns=class_names)
    confidence_df = confidence_df.melt(var_name='Label', value_name='Confidence')
    predicted_label = class_names[predicted.item()]

    return confidence_df, predicted_label

# Function to load BERT model and tokenizer
def load_bert():
    phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
    return phobert, tokenizer

# Function to plot HTML data
def plot_data(train_html_path, test_html_path, val_html_path):
    if not (os.path.exists(train_html_path) and os.path.exists(test_html_path) and os.path.exists(val_html_path)):
        st.error("HTML files not found.")
        return

    with open(train_html_path, "r", encoding="utf-8") as f_train:
        train_content = f_train.read()
        st.components.v1.html(train_content, height=600, scrolling=True)

    with open(test_html_path, "r", encoding="utf-8") as f_test:
        test_content = f_test.read()
        st.components.v1.html(test_content, height=600, scrolling=True)

    with open(val_html_path, "r", encoding="utf-8") as f_val:
        val_content = f_val.read()
        st.components.v1.html(val_content, height=600, scrolling=True)





def main():
    
    #st.title("News Classifier App")
    activities = ["Introduction", "Text Preprocessing", "Feature Extraction", "Train and Evaluate Models", "Prediction"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    # Preprocessing data
    if choice == "Text Preprocessing":
        st.info("Text Preprocessing")
        preprocessing_task = ["No Options", "Data Overview", "Process Text Demo", "Load Preprocessed Data"]
        task_choice = st.selectbox("Choose Task", preprocessing_task)
        if task_choice == "Data Overview":
            st.markdown("This dataset consists of Vietnamese news articles collected from various Vietnamese online news portals such as Thanh Nien, VNExpress, BaoMoi, etc. The dataset was originally sourced from a MongoDB dump containing over 20 million articles.")
            st.markdown("From this large dataset, our team extracted approximately 162,000 articles categorized into 13 distinct categorie and split into training, test and validation sets after preprocessing the data with 70%, 15% and 15% respectively.")
            st.markdown("Link to dataset: https://github.com/binhvq/news-corpus")
            st.image("images/sample_data.png", caption="Sample original data", use_column_width=True)
            summary_df = pd.read_csv("assets/summary_data.csv")
            st.dataframe(summary_df)
            train_images = "images/article_by_categories_train_data.html"
            test_images = "images/article_by_categories_test_data.html"
            val_images = "images/article_by_categories_val_data.html"
            plot_data(train_images, test_images, val_images)
            st.image("images/token_length_distribution.png",caption="Distribution of Token Count per Sentence", use_column_width=True)
        elif task_choice == "Process Text Demo":
            st.markdown("**Preprocessing Steps:**")
            st.markdown("- Standardize Vietnamese words, convert to lower case")
            st.markdown("- Utilize techniques such as regular expressions to remove unwanted elements: html, links, emails, numbers,...")
            st.markdown("- Employ a POS tagging tool to determine the grammatical category of each word in the sentence and filter out important components")
            
            news_text = st.text_area("Enter Text","Type Here")
            if st.button("Execute"):
                st.subheader("Original Text")
                st.info(news_text)
                preprocessed_news = preprocess_text(news_text)
                st.subheader("Preprocessed Text")
                st.success(preprocessed_news)
        elif task_choice == "Load Preprocessed Data":
            df = pd.read_json(PREPROCESSED_DATA, encoding='utf-8', lines=True)
            st.dataframe(df.head(20), use_container_width=True)

    # Feature Extration
    if choice == "Feature Extraction":
        st.info("Feature Extraction")
        
        feature_extraction_task = ["No Options", "PhoBert"]
        task_choice = st.selectbox("Choose Model",feature_extraction_task)
        if task_choice == "PhoBert":
            st.markdown("**Feature Extraction Steps:**")
            st.markdown("- Tokenize using PhoBert's Tokenizer. Note that when tokenizing we will add two special tokens, [CLS] and [SEP] at the beginning and end of the sentence.")
            st.markdown("- Insert the tokenized text sentence into the model with the attention mask. Attention mask helps the model only focus on words in the sentence and ignore words with additional padding. Added words are marked = 0")
            st.markdown("- Take the output and take the first output vector (which is in the special token position [CLS]) as a feature for the sentence to train or predict (depending on the phase).")
            phobert, tokenizer = load_bert()
            text = st.text_area("Enter Text","Type Here")
            if st.button("Execute"):
                st.subheader("Sentence to ids")
                padded, attention_mask = tokenize_text([text], tokenizer, max_len=256)
                st.write("Padded Sequence:", padded)
                st.write("Attention Mask:", attention_mask)

                st.subheader("Vector Embedding of Sentence")
                v_features = get_vector_embedding(padded, attention_mask, phobert)
                st.write("Vector Embedding:", v_features)
            

    if choice == "Prediction":
         st.info("Predict with new text")
         
         all_dl_models = ["No Options", "BiLSTM + phobertbase", "longformer-phobertbase", "phobertbase"]
         model_choice = st.selectbox("Choose Model", all_dl_models)
         
         if model_choice == "BiLSTM + phobertbase":
                model, tokenizer, max_len, phobert = load_models(model_type="bilstm_phobertbase")
                news_text = st.text_area("Enter Text", "Type Here")
                if st.button("Classify"):
                    st.header("Original Text")
                    st.info(news_text)
                    st.header("Predict")
                    processed_news = preprocess_text(news_text)
                    predicted_label, confidence_df = predict_label(processed_news, tokenizer, phobert, model, class_names, max_len)
                    st.subheader("Confidence Per Label")
                    st.dataframe(confidence_df, use_container_width=True)
                    st.subheader("Predicted Label")
                    st.success(predicted_label)
                
         if model_choice == "longformer-phobertbase":
                models, tokenizer, max_len = load_models(model_type="longformer")
                news_text = st.text_area("Enter Text", "Type Here")
                if st.button("Classify"):
                    st.header("Original Text")
                    st.info(news_text)
                    st.header("Predict")
                    df_confidence, predicted_label = infer(news_text, tokenizer, models, class_names, max_len)
                    st.subheader("Confidence Per Label")
                    st.dataframe(df_confidence, use_container_width=True)
                    st.subheader("Predicted Label")
                    st.success(predicted_label)
         if model_choice == "phobertbase":
                models, tokenizer, max_len = load_models(model_type="phobertbase")
                news_text = st.text_area("Enter Text", "Type Here")
                if st.button("Classify"):
                    st.header("Original Text")
                    st.info(news_text)
                    st.header("Predict")
                    df_confidence, predicted_label = infer(news_text, tokenizer, models, class_names, max_len)
                    st.subheader("Confidence Per Label")
                    st.dataframe(df_confidence, use_container_width=True)
                    st.subheader("Predicted Label")
                    st.success(predicted_label)
    if choice == "Train and Evaluate Models":
         st.info("Train and Evaluate Models")
         training_task = ["No Options", "Model Definitions", "Hyperparameters", "Result of Evaluation"]
         training_choice = st.selectbox("Choose Options", training_task)
         if training_choice == "Model Definitions":
             st.subheader("Longformer-phobertbase Model and Phobertbase Model")
             # Display model architecture
             st.code("""
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
                    """, language='python')
             # Explanation for each layer
             st.markdown("""
                - **Dropout Layer**: The dropout layer with a dropout probability of 0.3 helps prevent overfitting during training.
                - **Fully-connected Layer**: The fully-connected layer (`self.fc`) converts the output of the BERT model to a set of class predictions corresponding to the number of classes. This is achieved by a linear transformation using the BERT hidden size as the input dimension and the number of classes (`n_classes`) as the output dimension.
                - **Weight Initialization**: The weights and biases of the fully-connected layer are initialized using a normal distribution to facilitate better training.
                - **Forward Method**: In the forward method, the BERT model is called with the input IDs and attention mask. The output is passed through the dropout layer and then through the fully-connected layer to produce the final predictions.
                """)

             st.subheader("BiLSTM Model with Phobert feature extraction")
             # Display model architecture
             st.image("images/bilstm_phobertbase_summary.png")

             # Explanation for each layer
             st.markdown("""
              **Input Layer (input_1):** This layer accepts the input data and prepares it for further processing by the model. 
              It receives input in the shape (None, 1, 768), where `None` represents the batch size, `1` represents the sequence length (or time steps), and `768` represents the feature dimension. 

              **Bidirectional LSTM Layer (bidirectional):** This layer processes the input sequence bidirectionally, combining information from both past and future states to enhance learning. 
              It takes input in the shape (None, 1, 768) and outputs (None, 1, 448), reducing the feature dimension to `448`.

              **Dropout Layer (dropout):** Dropout is applied to regularize the model by randomly setting a fraction of input units to zero during training, preventing overfitting. 
              It takes input in the shape (None, 1, 448) and outputs (None, 1, 448), maintaining the same shape as the input.

              **Second Bidirectional LSTM Layer (bidirectional_1):** Another BiLSTM layer further refines the sequence representation by processing it bidirectionally again. 
              It takes input in the shape (None, 1, 448) and outputs (None, 1, 288), reducing the feature dimension to `288`.

              **Second Dropout Layer (dropout_1):** Another dropout layer is applied to further regularize the model after the second BiLSTM layer. 
              It takes input in the shape (None, 288) and outputs (None, 288), maintaining the same shape as the input.

              **Dense Layer (dense):** This fully connected layer applies a non-linear transformation to the extracted features, aiding in capturing complex patterns in the data. 
              It takes input in the shape (None, 288) and outputs (None, 160), reducing the dimensionality of the data to `160`.

              **Output Dense Layer (dense_1):** The final dense layer with softmax activation produces probabilities across multiple classes, making predictions based on the learned features. 
              It takes input in the shape (None, 160) and outputs (None, 13), corresponding to the number of classes in the classification task.
              """)
         if training_choice == "Hyperparameters":
             dl_model = ["No Options", "BiLSTM + phobertbase", "longformer-phobertbase and phobertbase"]
             model_choice = st.selectbox("Choose Model", dl_model)
             if st.button("Show Result"):
                if model_choice == "BiLSTM + phobertbase":
                  st.header("Optuna Hyperparameter Optimization")
                  st.markdown("""
                  We used `Optuna` for hyperparameter optimization due to its efficiency and advanced search algorithms. It automates the optimization process, reducing manual effort and improving model performance.
                  
                  The study is set to `maximize` the target metric. `TPESampler` is used for efficient and adaptive search, while `HyperbandPruner` stops unpromising trials early to save resources and speed up the optimization process.
                  """)
                  
                  # Explanation of Optuna terms
                  st.subheader("Understanding Optuna Terms")
                  st.markdown("""
                  **Pruner Trials:** These are trials that Optuna has pruned during the optimization process to reduce resource consumption. Pruning helps discard trials that are unlikely to yield better results or are taking too long to converge.
                  
                  **Complete Trials:** These trials are successfully completed by Optuna and have provided valid results. Optuna uses these trials to evaluate and select the best hyperparameters based on the defined optimization objective.
                  
                  **Failed Trials:** Trials that have encountered errors or failed to complete due to technical issues or improper configurations. These trials do not contribute valid results to the optimization process.
                  """)
                  
                  # Load and display trial information
                  trials = pd.read_csv("assets/study_bilstm_256_trials.csv")
                  st.subheader("Number of Completed Trials out of 100 trials")
                  st.dataframe(trials.style.format(precision=6), height=600, hide_index=True, use_container_width=True)
                  
                  # Load best hyperparameters and display
                  with open("hyperparameters/BiLSTM_phobertbase.json", 'r', encoding='utf-8') as file:
                      bilstm_phobertbase_best_param = json.load(file)
                  bilstm_phobertbase_best_param_df = pd.DataFrame([bilstm_phobertbase_best_param])
                  st.subheader("Best Hyperparameters")
                  st.dataframe(bilstm_phobertbase_best_param_df.style.format(precision=6), hide_index=True, use_container_width=True)
                  
                  # Display optimization history plot with title
                  st.subheader("Optimization History Plot")
                  with open("images/study_bilstm_phobertbase_optimize_history.html", "r", encoding="utf-8") as f:
                      content = f.read()
                      st.components.v1.html(content, height=600, scrolling=True)
                        
                if model_choice == "longformer-phobertbase and phobertbase":
                        
                        with open("./hyperparameters/phobertbase.json", 'r', encoding='utf-8') as file:
                            param = json.load(file)
                        param_df = pd.DataFrame([param])
                        st.subheader("Best Hyperparamters")
                        st.dataframe(param_df.style.format(precision=6), hide_index=True, use_container_width=True)
         if training_choice == "Result of Evaluation":
            st.markdown("To evaluate the performance of our models, we used several key metrics:")
            st.markdown("1. **Accuracy**: The proportion of correctly classified instances among the total instances.")
            st.markdown("2. **Precision**: The proportion of true positives among all positive predictions, indicating the accuracy of the positive predictions.")
            st.markdown("3. **Recall**: The proportion of true positives among all actual positives, reflecting the model's ability to capture all relevant instances.")
            st.markdown("4. **F1-score**: The harmonic mean of precision and recall, providing a balance between the two metrics.")
            st.markdown("5. **Confusion Matrix**: A table that displays the true positives, true negatives, false positives, and false negatives, used to evaluate the overall performance and error types of the model.")
            task = ["No Options", "Overall", "Evaluate per Label"]
            task_choice = st.selectbox("Choose Options", task)
            if task_choice == "Overall":
              result = pd.read_csv("assets/model_results.csv")
              st.dataframe(result, height=150, hide_index=True, use_container_width=True)
            if task_choice == "Evaluate per Label":
              st.subheader("Confusion Matrix Comparison")
              col1, col2, col3 = st.columns(3)  
              
              with col1:
                  st.image("images/confusion_matrix_bilstm_phobertbase.png", caption="BiLSTM with PhoBert feature extraction", use_column_width=True)
              
              with col2:
                  st.image("images/confusion_matrix_phobertbase.png", caption="phobertbase", use_column_width=True)
              
              with col3:
                  st.image("images/confusion_matrix_longformer.png", caption="longformer-phobertbase", use_column_width=True)

              st.subheader("Classification Report Comparison")
              col4, col5, col6 = st.columns(3)  
              
              with col4:
                  st.markdown("**BiLSTM with PhoBert feature extraction**")
                  bilstm_report = pd.read_csv("assets/classification_report_bilstm_phobertbase.csv")
                  st.dataframe(bilstm_report, height=600, hide_index=True, use_container_width=True)
              
              with col5:
                  st.markdown("**phobertbase**")
                  phobertbase_report = pd.read_csv("assets/classification_report_phobertbase.csv")
                  st.dataframe(phobertbase_report, height=600, hide_index=True, use_container_width=True)
              
              with col6:
                  st.markdown("**longformer-phobertbase**")
                  longformer_report = pd.read_csv("assets/classification_report_longformer.csv")
                  st.dataframe(longformer_report, height=600, hide_index=True, use_container_width=True)
    if choice == "Introduction":
      st.markdown(
        """
        <style>
            .title {
                font-size: 35px;
                font-weight: bold;
                text-align: center;
                color: #2c3e50;
                margin-top: 0px;
            }
            .university {
                font-size: 30px;
                font-weight: bold;
                text-align: center;
                color: #34495e;
                margin-top: 0px;
            }
            .faculty {
                font-size: 30px;
                font-weight: bold;
                text-align: center;
                color: #34495e;
                margin-bottom: 20px;
            }
            .subtitle {
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                color: #34495e;
                margin-bottom: 10px;
            }
            .student-info, .instructor-info {
                font-size: 18px;
                text-align: center;
                color: #7f8c8d;
                margin: 10px 20px;
            }
            .note {
                font-size: 16px;
                color: #95a5a6;
                margin-top: 20px;
                font-style: italic;
                text-align: left;
                margin: 20px;
            }
            
        </style>
        """,
        unsafe_allow_html=True
    )

      st.markdown('<div class="university">HCMC University of Technology and Education</div>', unsafe_allow_html=True)
      st.markdown('<div class="faculty">Faculty of Information Technology</div>', unsafe_allow_html=True)

      # Use Streamlit's st.image to display the logos
      left_co, cent_co,last_co, t, f, s, s = st.columns(7)
      with t:
          st.image("images/logo.png")

      st.markdown('<div class="subtitle">Graduation Thesis</div>', unsafe_allow_html=True)
      st.markdown('<div class="title">Vietnamese News and Articles Classification using PhoBERT</div>', unsafe_allow_html=True)

      st.markdown(
        """
        <div class="student-info">
            <p>Nguyen Thi Dieu Hien - 20133040</p>
            <p>Bui Tan Dat - 20133033</p>
        </div>

        <div class="instructor-info">
            <p>Instructor: PhD. Nguyen Thanh Son</p>
        </div>

        <div class="note">
            Note: This is an interactive web application to demonstrate various tasks related to news classification using deep learning models. Choose an activity from the sidebar to get started.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
	main()
