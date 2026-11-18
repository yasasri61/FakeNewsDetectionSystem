# AI Fake News Detection (Python + Streamlit)

A small educational NLP / machine-learning project. It trains a TF-IDF + Logistic
Regression classifier and provides a browser-based interface with Streamlit.

## Dataset

Download the **Fake and Real News Dataset**:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Extract the two files into the `data/` folder so the layout is:

```text
fake_news_detection_project/
├── app.py
├── train.py
├── requirements.txt
├── data/
│   ├── Fake.csv
│   └── True.csv
└── model/                 # created after training
```

## Setup

Use Python 3.9 or newer. Open a terminal in this project folder:

```bash
python -m venv .venv
```

Activate it:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This creates `model/fake_news_pipeline.joblib` and `model/metrics.json`.
The script prints accuracy, precision, recall, F1-score, and a confusion matrix.

## Run the GUI

```bash
streamlit run app.py
```

Streamlit opens the app in your browser. Paste a news article and click
**Analyze news**.

## How it works

1. Load the real and fake CSV files and assign labels.
2. Combine each article's title and body.
3. Split into training and held-out test data.
4. TF-IDF converts text into numeric features (unigrams and bigrams).
5. Logistic Regression learns to classify the text.
6. A Streamlit interface displays the prediction and model scores.

## Important limitations

This is a classroom demonstration, not a fact-checking service. The model learns
language and source patterns from one particular dataset. It may fail on new
events, different publishers, satire, short headlines, or manipulated articles.
Its displayed scores are classifier scores, not proof that a claim is true.
For a fairer evaluation, investigate duplicate stories, publisher/source leakage,
time-based splits, and test on a separate recent dataset.

# These are specially written testing examples, not actual published news.
## 1. Fake-style news examples
## Expected: FAKE
News 1:

Scientists have discovered that drinking two glasses of salt water every morning can increase human intelligence by 200%. The discovery was made after testing thousands of students. Experts recommend this method to improve memory and concentration.

News 2:

A new government announcement claims that all college examinations across India will be permanently cancelled from next month. Students will receive degrees automatically without attending classes or writing exams.

News 3:

A technology company has developed a mobile phone that can charge its battery fully within just one second using sunlight. The company claims the phone never needs electricity or a charging cable.

News 4:

Researchers claim that eating chocolate every day can make people live up to 150 years. A secret ingredient in chocolate reportedly stops the ageing process completely.

News 5:

A viral social media message claims that the government will deposit ₹50,000 into every citizen's bank account tomorrow. People only need to share their Aadhaar number through a WhatsApp link to receive the money.
## 2. Real-style news examples
## Expected: REAL
News 6:

India's Chandrayaan-3 mission successfully achieved a soft landing on the Moon on August 23, 2023. The mission demonstrated India's capability in lunar exploration and made India the first country to land near the Moon's south polar region.

News 7:

The Reserve Bank of India is responsible for regulating monetary policy and overseeing the country's banking system. It uses tools such as the repo rate to manage inflation and maintain financial stability.

News 8:

Solar energy is a renewable source of electricity generated using sunlight. Solar panels convert sunlight into electrical energy and are increasingly used in homes, industries and large power plants.

News 9:

The World Health Organization works with countries to improve public health, prevent diseases and respond to health emergencies. It also provides guidance on vaccination and disease prevention.

News 10:

Artificial intelligence is increasingly being used in healthcare for medical image analysis, patient monitoring and supporting clinical decisions. These technologies can assist healthcare professionals but do not replace the need for medical expertise.