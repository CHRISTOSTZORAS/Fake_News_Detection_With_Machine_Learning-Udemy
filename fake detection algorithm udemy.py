import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from collections import Counter
#data importing and data cleaning
new_articles=pd.read_csv(r'C:\\Users\\tzwrakos\\OneDrive\\Υπολογιστής\\Udemy\\dataset\\news_articles.csv')
new_articles.dropna(axis=0,inplace=True)
new_articles.drop_duplicates(inplace=True)
cleaned_dataset=pd.read_csv(r'C:\\Users\\tzwrakos\\OneDrive\\Υπολογιστής\\Udemy\\dataset\\cleaned_dataset.csv')

#new source credibility analysis
source_counts=cleaned_dataset.groupby(["site_url","label"]).size().unstack(fill_value=0)
source_counts["Percentage Real (%)"]=(source_counts['Real']/(source_counts['Real']+source_counts['Fake']))*100
source_counts["Percentage Fake (%)"]=(source_counts['Fake']/(source_counts['Real']+source_counts['Fake']))*100

sorted_sources=source_counts.sort_values(by='Percentage Real (%)',ascending=False)
print("Top 10 Most Credible News Sources:")
for source, row in sorted_sources.head(10).iterrows():
    print(f"News {source},fake news= {row['Percentage Fake (%)']:.1f}%")
    
print("Top 10 Least Credible News Sources:")
for source, row in sorted_sources.tail(10).iterrows():
    print(f"News {source},fake news= {row['Percentage Fake (%)']:.1f}%")
    
#detecting keywords associated with fake news
stop_words = set(stopwords.words("english"))

title_counter = Counter()
text_counter = Counter()

for index, row in cleaned_dataset.iterrows():
  title_words = word_tokenize(row["title"])
  text_words = word_tokenize(row["text"])

  title_words = [word.lower() for word in title_words if word.isalpha() and word.lower() not in stop_words]
  text_words = [word.lower() for word in text_words if word.isalpha() and word.lower() not in stop_words]

  if row["label"] == "Fake":
    title_counter.update(title_words)
    text_counter.update(text_words)

top_keywords_title = title_counter.most_common(5)
top_keywords_text = text_counter.most_common(5)

print("Top 5 Keywords Associated with Fake News Titles:")
for keyword, count in top_keywords_title:
  print(f"{keyword}:{count} times")
print("Top 5 Keywords Associated with Fake News Texts:")
for keyword, count in top_keywords_text:
  print(f"{keyword}:{count} times")
