# LSTM Spam Email Detection Using Common NLP Techniques

This project builds a deep learning model using LSTM (Long Short-Term Memory networks) to automatically classify email messages as spam or not spam ("ham"). The project leverages natural language processing methods and a public dataset to demonstrate robust text classification for spam filtering applications.

## Features

- **Preprocessing:** Cleans emails, removes stopwords, and tokenizes text data.
- **Dataset:** Uses the [TREC 2007 Spam/Ham Dataset](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2007), ensuring ethical and high-quality spam detection.
- **Model Workflow:**
  - Splits data into training and test sets (~53,668 emails).
  - Text sequences converted for LSTM input.
  - Sequential neural network with Embedding layer and LSTM for contextual learning.
  - Final Dense layer for binary output.
- **Evaluation Metrics:** 
  - Accuracy, precision, recall, F1-score, ROC-AUC
  - Confusion matrix and classification report
- **Predictions:**  
  - Sample code cells for classifying new emails (e.g., "You've won a free iPhone!" → SPAM).

## Results

- Achieves high accuracy (≥99%) on test set.
- Robust performance measures for real-world spam filtering.

## Usage

1. **Clone the repository:**
git clone https://github.com/LAKAVATHRAGHURAM/LSTM-Spam-Email-Detection-NLP.git


2. **Install requirements:**
- Install dependencies with `pip install -r requirements.txt`
- (NLTK, pandas, numpy, scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn)
3. **Run the notebook:**
- Open `Build_an_LSTM_Model_to_Detect_an_Automated_Spam_E-mail_Using_Common_Natural_Language_Processing.ipynb` in Jupyter or Google Colab.

## Project Structure

- `Build_an_LSTM_Model_to_Detect_an_Automated_Spam_E-mail_Using_Common_Natural_Language_Processing.ipynb` – main notebook
- Data loading and cleaning sections
- Model building, training, and evaluation
- Sample predictions and report

## Credits

- Developed by LAKAVATH RAGHURAM (2022BCS0129)  
- Dataset from Kaggle: [emails-for-spam-or-ham-classification-trec-2007](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2007)

## License

This project is licensed under the MIT License.

---



