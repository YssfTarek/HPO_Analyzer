import os
import re
import string
from collections import Counter
from docx import Document
from nltk.corpus import stopwords

#Path where docx data is stored
FOLDER_PATH = "/Users/youssef/Documents/HPO_Analyzer/data"

# Step 1: Extract text from a Word file
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Step 2: Clean and tokenize the text
def clean_and_tockenize(text):
    # Lowercase the text
    text = text.lower()
    
    #remove punctuation
    words = text.split()
    
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in words if word not in stop_words and word.isalpha()]

    return cleaned_words

# Step 3: Analyze word frequency
def analyze_word_frequency(folder_path):
    all_words = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            full_path = os.path.join(folder_path, filename)
            print(f"Prcossing {filename}")
            text = extract_text_from_docx(full_path)
            words = clean_and_tockenize(text)
            all_words.extend(words)
            
    word_counts = Counter(all_words)
    return word_counts.most_common(50) #top 50 words

if __name__ == "__main__":
    top_words = analyze_word_frequency(FOLDER_PATH)
    
    print("\nTop 50 most common words:\n")
    
    for word, count in top_words:
        print(f"{word}:{count}")