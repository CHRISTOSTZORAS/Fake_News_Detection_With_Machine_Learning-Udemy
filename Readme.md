# 📰 Fake Article Detection 🧠


This repository contains a machine learning pipeline for detecting fake news. It uses various models such as **Random Forest**, **Logistic Regression**, and **Fairlearn** for fairness evaluation. Below are the key steps implemented in the project.

## Table of Contents

- [Introduction](#introduction)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Keyword Analysis](#keyword-analysis)
- [Title and Text Length Correlation](#title-and-text-length-correlation)
- [Sensationalism Detection](#Sensationalism-detection)
- [Emotion Analysis Using NLP](#sentiment-analysis)
- [Fake News Prediction Models Using Feature Engineering](#feature-engineering)
- [Fake News Prediction Models Using Logistic Regression](#logistic-regression)
- [Fake News Prediction Models Using Random Forest](#random-forest)
- [Evaluating Fake News Detection Model with confusion Matrix](#confussion-matrix)
- [Fairness Audit](#fairness-audit)
- [Installation](#installation)
- [Contact](#contact)

---

# 📖 Introduction

This project performs fake news detection using multiple approaches:

1. **Keyword Analysis**: Identifying keywords associated with fake news.
2. **Title and Text Length Correlation**: Analyzing the length of titles and text in fake vs real news articles.
3. **Sensationalism Detection**: Identifying sensationalist keywords in news text.
4. **Emotion Analysis**: Detecting sentiment polarity in news text to check if fake news leans more towards extreme emotions.
5. **Fake News Prediction Models**: Using models such as **Random Forest** and **Logistic Regression** for classification.
6. **Fairness Audit**: Ensuring the model performs equitably across different demographic groups using Fairlearn.

---

# 🛠️ Data Cleaning and Preprocessing

### Data Import and Cleaning
The dataset consists of articles with titles, text, and labels (Real or Fake).

- Missing values and duplicates are removed.
- The dataset is then split into features (`title`, `text`) and labels (`label`).

```python
new_articles = pd.read_csv('path_to_data.csv')
new_articles.dropna(axis=0, inplace=True)
new_articles.drop_duplicates(inplace=True)
```
---

# 🛠️ Keyword Analysis
We perform a **keyword analysis**  to identify the most common words associated with **Fake News**. By analyzing the frequency of words in both the **titles** and **texts** of fake news articles, we can gain insights into the language patterns commonly used in fake news.
### Key Steps:
1. **Data Preprocessing**:
   - Tokenized the **titles** and **texts** of the articles.
   - Removed **stopwords** and non-alphabetic characters.
2. **Top 5 Keywords**:
   - **Titles**: The most common words found in fake news titles.
   - **Texts**: The most common words found in fake news texts.
### Results:
#### Top 5 Keywords Associated with Fake News Titles:
- **trump**: 135 times
- **hillary**: 129 times
- **clinton**: 121 times
- **title**: 91 times
- **us**: 59 times
#### Top 5 Keywords Associated with Fake News Texts:
- **clinton**: 1990 times
- **trump**: 1975 times
- **one**: 1419 times
- **us**: 1385 times
- **said**: 1353 times

---

# 🛠️ Title and Text Length Correlation
We analyze the average lengths of **titles** and **texts** for both **Real** and **Fake News**. This analysis helps identify any significant differences in the structure of news articles that are labeled as fake compared to those that are real.

### Key Insights:

- **Average Title Length**: Fake news articles tend to have **longer titles** than real news articles.
- **Average Text Length**: Fake news articles generally have **shorter text** compared to real news articles.

### Key Results:

#### Average Title Length:
- **Real News**: 56.67 characters
- **Fake News**: 65.07 characters

#### Average Text Length:
- **Real News**: 3209.14 characters
- **Fake News**: 2843.22 characters

### Observations:
- Fake news articles have **longer titles** and **shorter text** on average compared to real news. This could indicate that fake news often relies on more sensational or eye-catching titles while providing less substantive content.

---

# 🛠️ Sensationalism Analysis
This project performs an analysis to detect **sensationalism** in news articles. Sensationalist language is often used in fake news to attract attention. We detect this by checking for specific sensationalist keywords in the **text** of the news articles.

### Key Steps:

1. **Detect Sensationalism**:
   - We define a list of **sensationalist keywords** such as "shocking", "outrageous", "unbelievable", "mind-blowing", and "explosive".
   - These keywords are searched for in the **text** of each news article.
   
2. **Create Contingency Table**:
   - A contingency table is created to see how the **sensationalism** correlates with the **credibility** (fake vs. real).

3. **Chi-Squared Test**:
   - We use a **Chi-squared test** to determine if there is a significant association between **sensationalism** and **credibility**.
### Results:

#### Contingency Table:
The contingency table shows how often each label (Fake/Real) has sensationalist content:

| Sensationalism | Fake | Real |
|----------------|------|------|
| **False**      | 1252 | 746  |
| **True**       | 29   | 8    |

### Chi-Squared Test:

- **Chi-squared statistic**: 3.202575897618069
- **P-value**: 0.0735223856196211

### Conclusion:
Based on the **Chi-squared test**, with a **p-value** of 0.0735, which is greater than the significance level of 0.05, there **is no significant association between sensationalism and the credibility of news**.

---

# 🛠️ Emotion Analysis Using NLP
This project performs **emotion analysis** on news articles to detect the emotional tone of the content. By using **Sentiment Analysis** with the **VADER SentimentIntensityAnalyzer** from the `nltk` library, we classify the sentiment of the articles as **Positive**, **Negative**, or **Neutral**. 

### Key Steps:

1. **Sentiment Analysis**:
   - We use the **VADER SentimentIntensityAnalyzer** to analyze the sentiment of the **text** of the news articles.
   - Based on the **compound sentiment score**, the articles are categorized as:
     - **Positive** if the score is greater than or equal to 0.05
     - **Negative** if the score is less than 0.05
     - **Neutral** if the score is exactly 0

2. **Sentiment Classification**:
   - After applying sentiment analysis to the **text** of each article, a **Sentiment** label (Positive, Negative, or Neutral) is assigned to each article.

### Results:

Here are some examples of the articles and their corresponding **sentiment labels**:

| Text                                                                 | Sentiment |
|----------------------------------------------------------------------|-----------|
| "Print they should pay all the back all the money."                  | Positive  |
| "Why did attorney general Loretta Lynch plead the fifth?"            | Negative  |
| "Red state \nFox News Sunday reported this morning."                 | Positive  |
| "Email Kayla Mueller was a prisoner and tortured by ISIS."           | Positive  |
| "Email healthcare reform to make America great again."               | Positive  |

### Observations:
- Many articles related to politics tend to have either **positive** or **negative** sentiments, based on the subject matter.
- Neutral sentiments are relatively rare in the dataset.

---
  
# 🛠️ Fake News Prediction Models Using Feature Engineering
In this project, we perform **Feature Engineering** to identify key features associated with **Fake News** articles. We extract these features based on **word frequencies** and **site URL patterns**, helping improve future machine learning models.

### Key Features:

1. **Feature 1: Top Keywords Associated with Fake News**
   - We extract the most frequent words used in **fake news** articles, excluding stop words.
   
2. **Feature 2: Fake News Percentage by Site URL**
   - We calculate the percentage of fake news articles published by specific site URLs.
   - Sites with a high percentage of fake news articles are flagged as potentially more prone to spreading misinformation.

3. **Feature 3: Fake News Prediction Function**

    - We use the extracted **Top Keywords** and the **Fake News Percentage by Site URL** to build a simple rule-based model that predicts whether a news article is **Fake** or **Real**. The model checks if the article's title contains any of the top keywords associated with fake news and whether the news source is highly associated with fake news.

### Prediction Logic:
- If the title contains one or more keywords associated with fake news and the site URL has a high percentage of fake news, the article is classified as **Fake News**.
- Otherwise, the article is classified as **Real News**.


### Results:

#### Top 10 Keywords Associated with Fake News:

After analyzing the fake news data, the following words are most frequently associated with fake news articles:

```python
['clinton', 'trump', 'said', 'hillary', 'people', 'like', 'just', 'new', 'election', 'time']
```
Here are some of the site URLs with the highest number of fake new articles:
| URL                                                                 | NUMBER OF FAKE ARTICLES |
|-----------------------------|------|
| "frontpagemag.com"          | 100  |
| "clickhole.com "            | 100  |
| "adwnews.com"               | 100  |
| "activistpost.com"          | 100  |
| "prisonplanet.com"          | 100  |

---

# 🛠️ Fake News Prediction Models Using Logistic Regression 

This project involves detecting fake news articles using a **Logistic Regression** model. The model is trained to classify news articles as either **Real** or **Fake** based on the content of the articles.

### What We Did:

1. **Data Preprocessing**:
   - We first checked the dataset for missing values and handled any that were present.
   - The **labels** (Real / Fake) were encoded into numerical values using **LabelEncoder**.

2. **Text Feature Extraction**:
   - We applied **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the text data into numerical features. This method helps to identify important words in the articles while ignoring commonly used stop words.

3. **Model Training**:
   - We trained a **Logistic Regression** model on the processed data. The model was fit using a training dataset, and the performance was evaluated on a test set.

4. **Prediction and Evaluation**:
   - The model was used to predict the labels for news articles in the test set.
   - The performance of the model was evaluated using **accuracy**, which indicates how well the model can distinguish between real and fake news.

### How It Was Done:

- **TF-IDF** was used to extract meaningful features from the text.
- **Logistic Regression**, a machine learning algorithm, was used to train a binary classifier that predicts whether a news article is real or fake based on the extracted features.
- The dataset was split into a **training** set and a **test** set, allowing the model to learn from the training set and be evaluated on the unseen test set.

### Results:

- The **Logistic Regression** model achieved an **accuracy of 73%** on the test set, meaning it correctly classified 73% of the articles as either real or fake.
- The model was also used to make predictions on new articles. For example:
  - For the input: "Olympiacos has had a rough time with its transfers this year..." the prediction was **Real News**.
  - For the input: "Hillary Clinton and Donald Trump said that they are getting married..." the prediction was **Fake News**.

### Conclusion:

The **Logistic Regression** model was able to classify news articles with an accuracy of **73%**. This approach provides a solid foundation for detecting fake news, although there is room for improvement, such as incorporating additional features or using more advanced models.

---

# 🛠️ Fake News Prediction Models Using Random Forest
In this part of the project, we utilize a **Random Forest Classifier** to predict whether a news article is **Fake** or **Real** based on its **title**. The model uses **TF-IDF** (Term Frequency-Inverse Document Frequency) for text feature extraction and **Label Encoding** for the labels.

### What We Did:

1. **Data Preprocessing**:
   - The **title** column is selected as the primary feature to classify the articles.
   - **TF-IDF** was applied to convert the text data (titles) into numerical features, allowing the model to understand and classify the data.
   - **Label Encoding** was used to convert the categorical labels (Real/Fake) into numerical values for model training.

2. **Model Training**:
   - A **Random Forest Classifier** was trained using the **TF-IDF** feature representation of the article titles. Random Forest is an ensemble learning method that builds multiple decision trees to make a prediction.

3. **Model Prediction**:
   - The trained model was used to predict whether an article's title corresponds to **Fake News** or **Real News**.

### How It Was Done:

- **TF-IDF Vectorizer** was used to extract features from the **title** of each article.
- A **Random Forest Classifier** with 100 trees was trained on these features to learn the pattern of **Fake News** vs **Real News**.
- The model was then tested on new data to make predictions.

### Results:

The model was used to predict the label of a new article based on its title:

#### Example Input:

**Title**:
`"University announces educational programming related to recent geopolitical conflict in the Middle East"`

#### Example Output:

**Prediction**:
`Real News`

This prediction indicates that the article with this title is likely to be **Real News** according to the model.

---

# 🛠️ Evaluating Fake News Detection Model with confusion Matrix
In this part of the project, we evaluate the performance of the **Fake News Detection** model using a **Random Forest Classifier**. The model is evaluated by comparing the predicted labels to the actual labels using a **confusion matrix**. This helps to understand how well the model classifies the articles as either **Fake** or **Real**.

### What We Did:

1. **Model Training**:
   - We trained a **Random Forest Classifier** using the **TF-IDF** feature vectors extracted from the **title** of the news articles.
   - The model was trained on 80% of the data and evaluated on the remaining 20% using the **test set**.

2. **Model Evaluation**:
   - We used the **confusion matrix** to evaluate the performance of the model. The confusion matrix shows the counts of **True Positives**, **True Negatives**, **False Positives**, and **False Negatives**.

3. **Visualization**:
   - The **confusion matrix** was visualized using a heatmap to make it easier to interpret and identify areas where the model might be underperforming.

### Confusion Matrix:

The **confusion matrix** helps assess how well the model performs in classifying both **Real** and **Fake** news articles. It shows the following results:
- **True Positives (TP)**: 214 (Real news correctly classified as Real)
- **True Negatives (TN)**: 63 (Fake news correctly classified as Fake)
- **False Positives (FP)**: 40 (Fake news incorrectly classified as Real)
- **False Negatives (FN)**: 90 (Real news incorrectly classified as Fake)

### Performance Evaluation:

- **Accuracy**: The model is able to correctly classify 214 out of 214 Real news articles and 63 out of 153 Fake news articles.


### Results:

The confusion matrix and heatmap show the following key observations:
- The model performs well with a high number of **True Positives (214)**, meaning that most **Real News** articles are correctly identified.
- However, there are some **False Negatives (90)**, where **Real News** articles are incorrectly classified as **Fake News**.
- There are also **False Positives (40)**, where **Fake News** articles are incorrectly classified as **Real News**.

---
# 🛠️ Fairness Audit
In this part of the project, we conduct a **Fairness Audit** to evaluate whether the **Fake News Detection** model performs equitably across different demographic groups. We use **Demographic Parity** as a metric to measure if the model's performance is consistent for all users, regardless of the group they belong to.

### What We Did:
1. **Diversity of Training Data**:
   - We ensured that the dataset used for training is **diverse** and **representative**. This is important to ensure that the model is exposed to a wide range of sources, perspectives, and writing styles. A diverse dataset reduces the likelihood of bias and increases the fairness of the model.
2. **Feature Engineering**:
   - We carefully engineered features to minimize potential bias. Features that could introduce or amplify bias, such as **sensitive demographic information**, were excluded from the model.
3. **Fairness Audit**:
   - We performed a **Fairness Audit** by calculating the **Demographic Parity Difference** between different groups in the dataset. This metric helps assess if the model's predictions are equitable for all groups.

### Demographic Parity Difference:

#### Metric Explanation:

- **Demographic Parity Difference** measures the disparity in the positive prediction rates between two groups (e.g., **Fake News** vs **Real News**).
  - A value of **0** indicates no disparity, meaning the model is equally likely to predict fake or real news for both groups.
  - A value greater than **0** indicates some level of disparity, where one group may be more likely to be predicted as **Fake News** than the other.

#### Result:
The **Demographic Parity Difference** between the two groups was calculated and the result was:

**Demographic Parity Difference: 0.254284390921723**

This result suggests that there is a **moderate disparity** in the prediction rates between the groups, with one group being more likely to be predicted as **Fake News** than the other.
#### Conclusion:
- The **Demographic Parity Difference** of **0.254** indicates that there is some level of bias in the model, as it favors one group over the other.
- Further improvements can be made by adjusting the model, engineering more balanced features, and ensuring that the model is exposed to an even more diverse and representative dataset.
- A **Fairness Audit** is essential to ensure that machine learning models, especially those used in **Fake News Detection**, are fair and equitable for all users.

# 🖥️ Installation
### 1️⃣ Clone the repository:
```bash
git clone https://github.com/your-repo/tzwrakos-chatbox.git
cd tzwrakos-chatbox
```

### 2️⃣ (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

### 3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

cd fake-article-detection
# 🤝 Contributing
Contributions are welcome! To contribute:

    --Fork the repository.
    --Create a new branch: git checkout -b feature-branch
    --Commit your changes: git commit -m 'Add new feature'
    --Push to the branch: git push origin feature-branch
    --Submit a Pull Request

# 📬 Contact
Follow me in Linkedin:'https://www.linkedin.com/in/christos-tzoras/'
