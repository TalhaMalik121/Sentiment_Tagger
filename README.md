# 🎯 Sentiment & Emotion Detector  

Analyze text at **paragraph** and **phrase** level for both **sentiment** and **emotion** using pre-trained transformer models.  
Built with ⚡ Hugging Face Transformers + 🧠 PyTorch + 📚 NLTK.  

---

## ✨ Features  

✅ **Paragraph-level Sentiment** — Positive / Neutral / Negative  
✅ **Paragraph-level Emotion** — 28+ fine-grained emotions (GoEmotions)  
✅ **Phrase-level Insights** — Break down sentences into smaller chunks for deeper analysis  
✅ **GPU Acceleration** — Automatically uses CUDA if available  

---

## 📦 Dependencies  

| Library        | Purpose |
|----------------|---------|
| **pandas**     | CSV reading & data handling |
| **torch**      | Model inference |
| **transformers** | Loading pre-trained Hugging Face models |
| **nltk**       | Sentence tokenization |
| **re** *(built-in)* | Regex-based phrase splitting |

---

## ⚙️ Installation  

1️⃣ **Clone** or download this repository.  
2️⃣ **Install all dependencies** with:  

```bash
pip install -r requirements.txt
```

**requirements.txt**  
```
pandas
torch
transformers
nltk
```

---

## 🚀 Usage  

1. Prepare your CSV file:  
   - Name: `1000paragraphs.csv`  
   - Must have a column named **`paragraph`**  

2. Run the script:  
```bash
python sentiment_emotion_detector.py
```

3. Example Output:  

```
📄 Paragraph 1:
This is a wonderful day, but I feel a bit nervous about tomorrow.

📊 Overall Paragraph Sentiment ➤ 😄 Positive (Confidence: 0.87)
🎭 Overall Paragraph Emotion ➤ joy (Confidence: 0.78)

✂️ Phrase 1: This is a wonderful day
   ➤ Sentiment: 😄 Positive (Confidence: 0.92)
   ➤ Emotion:   joy (Confidence: 0.85)

✂️ Phrase 2: I feel a bit nervous about tomorrow
   ➤ Sentiment: 😡 Negative (Confidence: 0.68)
   ➤ Emotion:   nervousness (Confidence: 0.80)
```

---

## 🧠 Models Used  

- **Sentiment Analysis** → [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)  
- **Emotion Detection** → [`bhadresh-savani/bert-base-go-emotion`](https://huggingface.co/bhadresh-savani/bert-base-go-emotion)  

---

## 📌 Notes  

- 🛠 Automatically downloads **NLTK punkt tokenizer** if not found.  
- ⚡ Runs on **GPU** if available, otherwise CPU.  
