# Data Source

## Dataset
**Title:** Financial Tweets  
**Source:** [Kaggle – Financial Tweets Dataset](https://www.kaggle.com/datasets/davidwallach/financial-tweets)  
**Uploader:** David Wallach

## Description
This dataset contains tweets related to finance, the stock market, and investing. It is commonly used for sentiment analysis and natural language processing (NLP) tasks, including financial market prediction models.

## Contents
- **Text:** The body of each tweet.
- **Timestamp:** When the tweet was posted.
- **Ticker Symbols (if available):** Stock tickers mentioned in the tweet.
- **Additional Metadata:** May include user information, retweets, and likes depending on the dataset version.

## Usage Notes
- You must have a Kaggle account to download the dataset.
- Follow the Kaggle dataset’s license and terms of use when using or redistributing the data.
- For reproducibility, this project does not store the raw dataset in the repository.  
  Please download it directly from Kaggle.

## How to Download
1. Sign in to your Kaggle account.
2. Navigate to the dataset page: [Financial Tweets Dataset](https://www.kaggle.com/datasets/davidwallach/financial-tweets).
3. Click **"Download"** to save the `.csv` file locally.
4. Place the downloaded file in the `/data` directory of this project.

## How This Dataset Is Used in This Project
In this project, the **Financial Tweets** dataset is used as the core text source for training and evaluating machine learning models to classify sentiment related to financial markets. The workflow includes:
1. **Preprocessing** the tweets by removing stop words, URLs, punctuation, and performing tokenization.
2. **Feature Extraction** using vectorization to convert tweets into numerical representations.
3. **Model Training** on labeled sentiment classes using algorithms such as Transformer-based models.
4. **Evaluation** of model performance using accuracy, F1-score, precision, and recall.
5. **Application** of the trained model to unseen financial text data to demonstrate real-world sentiment prediction capabilities.


---
**Last Updated:** August 2025
