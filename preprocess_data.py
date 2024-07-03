import phonlp
import underthesea
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, StringType
import re

# Paths to original and processed data files
ORIGINAL_DATA = "./data/news_v2/news_v2.json"
PROCESSED_DATA = "./data/processed_data/final_data.json"

# Load NLP model
nlp_model = phonlp.load(save_dir="./phonlp")

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Preprocessing") \
    .master("local[*]") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.instances", "64") \
    .config("spark.executor.cores", "1") \
    .config("spark.memory.offHeap.enabled", True) \
    .config("spark.driver.memory", "50g") \
    .config("spark.memory.offHeap.size", "16g") \
    .config("spark.ui.showConsoleProgress", False) \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.log.level", "ERROR") \
    .getOrCreate()

print("Loading data....")
df = spark.read.json(ORIGINAL_DATA)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s.]', '', text)  # Remove special characters
    # Tokenize text into sentences
    sentences = underthesea.sent_tokenize(text)
    
    # List to store preprocessed words
    preprocessed_words = []
    
    # Iterate through each sentence
    for sentence in sentences:
        try:
            word_tokens = underthesea.word_tokenize(sentence, format="text")
            # Tokenize words and perform POS tagging
            tags = nlp_model.annotate(word_tokens, batch_size=64)
            
            # Filter words based on POS tags
            filtered_words = [word.lower() for word, tag in zip(tags[0][0], tags[1][0]) if tag[0] not in ['M', 'X', 'CH'] 
                              and word not in ["'", ","]]
            
            # Append filtered words to the result list
            preprocessed_words.extend(filtered_words)
        except Exception as e:
            pass
    
    # Convert list of words to string and return
    return ' '.join(preprocessed_words)

# Register preprocess_text function as a Spark UDF
preprocess_udf = udf(lambda text: preprocess_text(text), StringType())

# Add "processed_content" column to DataFrame by applying preprocess_text function to "content" column
df_processed = df.withColumn("processed_content", preprocess_udf(df["content"]))

# Select "processed_content" and "category" columns from DataFrame
selected_columns = ["processed_content", "category"]
df_selected = df_processed.select(selected_columns)

# Number of partitions
num_partitions = 1024 

# Write DataFrame with specified number of partitions
df_selected.repartition(num_partitions).coalesce(1).write.json(PROCESSED_DATA)
