# Biomedical Report NLP Summariser: Translating Radiology Reports to Layperson Summaries
This project demonstrates the application of light-weight NLP and sequence-to-sequence modeling to generate patient-friendly summaries from expert radiology reports. 

---

## Background
Imagine you are a patient reading your radiology report for the first time. You open the document and see:

> Mild cardiomegaly with no evidence of pulmonary edema or pleural effusion.

Without medical training, it is difficult to understand what this means or whether it should be a cause for concern. Now compare that to the following explanation:

> The heart appears slightly enlarged, but there are no signs of fluid in or around the lungs.

Both statements describe the same clinical findings, yet the second communicates them in language that is far more accessible to a non-clinical reader. This gap between expert medical language and patient understanding is the motivation behind this project.

Radiology reports are dense with technical terminology. For patients' better engagement and informed decision-making, this project bridges the gap between expert clinical language and layperson comprehension using a fine-tuned **T5-small** model.

T5-small is the Chinese Room, learning a correct text to text mapping as it consumes a dataset. One of its advantages is its lightweight nature - unlike large chat models such as ChatGPT. The un-fine-tuned T5-small does not have conversational abilities and will often produce incoherent or repetitive outputs if used out-of-the-box. However, its compact size — just 236 MB for the final fine-tuned model — makes it extremely efficient for domain-specific tasks like radiology summarisation. Its training is faster, and its memory usage is lighter.

The BioLaySumm dataset (ACL 2025 workshop, Subtask 2.1) contains paired radiology reports and corresponding layperson summaries. The model’s goal is to produce summaries that retain key clinical findings while simplifying complex language for readability.

---

## Objectives
- Fine-tune a pretrained **T5-small** model for text-to-text summarisation in the biomedical domain.  
- Develop a preprocessing, tokenisation, and training pipeline for the dataset.  
- Evaluate model performance using ROUGE metrics and compare results to human-written summaries.  
- Generate patient-friendly summaries suitable for non-clinical audiences.

---

## Literature Context
Sequence-to-sequence transformer models, such as T5 (Raffel et al., 2020), are effective for summarization and translation tasks. Prior medical NLP work often focuses on classification or decision support rather than layperson summaries. Text simplification models improve patient comprehension and engagement (Devaraj et al., 2021). This project applies state-of-the-art transformers in a domain-specific, clinically relevant context.

---

## Data Preparation and Exploration
- Initial dataset: **170,991 rows** (150,454 train, 10,000 validation, 10,537 test).  
- **Preprocessing steps:**  
  - Remove missing values and duplicate report-summary pairs  
  - Filter overly long reports (>1000 chars) and summaries (>512 chars)  
  - Remove image references and non-textual artifacts  
- **Filtered dataset:** 151,583 high-quality rows  
- **Final split:** 106,103 train / 22,742 validation / 22,738 test

Exploratory analysis revealed wide variation in report lengths, terminology, and complexity. Short reports were reproduced literally, while longer reports required summarisation and simplification.

---

## Model Training
- **Architecture:** T5-small (Text-to-Text Transfer Transformer)  
- **Training strategy:** Parameter-efficient fine-tuning (PEFT)  
- **Hardware:** NVIDIA RTX 3060 (8 GB), Intel i7-9700KF CPU  
- **Hyperparameters:**  
  - Batch size: 8 (train), 2 (eval)  
  - Gradient accumulation: 2  
  - Learning rate: 5e-5  
  - Epochs: 3  
  - Mixed precision (FP16) enabled  
- **Training duration:** ~1h 5min  
- **Final training loss:** 1.0988

The pipeline included tokenisation, attention masking, padding, and beam search to generate human-readable summaries.

---

## Evaluation
**Test set:** 22,738 samples  
**ROUGE scores:**  
- **ROUGE-1:** 0.654 (unigram overlap)  
- **ROUGE-2:** 0.459 (bigram overlap)  
- **ROUGE-L:** 0.595 (longest common subsequence)  
- **ROUGE-Lsum:** 0.595 (summary-level coverage)

These scores measure how much the model’s generated summaries overlap with the reference summaries at the word and phrase level; for example, a ROUGE-1 score of 0.654 means that approximately 65.4% of the important words in the human-written summary also appear in the model-generated summary.

---

## Examples

**Example 1**  
- **Report:** Growth in left pulmonary hilum (~2 cm), prompting thoracic CT scan.  
- **Model summary:** Growth in left lung hilum (~2 cm), prompting chest CT scan.  
- **Ground-truth:** New growth in left lung area (~2 cm); patient required a special chest CT scan.  

**Example 2**  
- **Report:** Follow-up for COVID-19; bilateral reticular pattern and subtle bibasal opacities.  
- **Model summary:** Follow-up shows net-like pattern in both lungs and faint hazy areas at the bottom.  
- **Ground-truth:** Same as previous X-ray; lungs show net pattern and faint cloudiness at bottom.  

**Example 5**  
- **Report:** Stable right lower lobe nodule with no interval change compared to prior imaging.  
- **Model summary:** A small spot in the lower right lung remains unchanged since the last scan.  
- **Ground-truth:** The small nodule in the lower right lung looks the same as before.  

---

## Discussion and Conclusion
What surprised me most was how coherent and genuinely useful the generated summaries were. After experimenting with the untuned model, I expected little more than radiology-esque terms garbled together. Instead, the fine-tuned T5-small consistently produced clear, readable summaries comparable to what a human might write. This project reinforced for me that impactful NLP systems do not always need to be large or conversational—carefully fine-tuned compact models can already deliver practical value, while avoiding much of the overhead that comes with chat-style models.

---

## Usage
1. Prepare preprocessed datasets (train/val/test in parquet format).  
2. Load model and tokeniser:

```python
from modules import load_model
```
3. Tokenise datasets with `get_tokensied_datasets(tokeniser).
4. Fine-tune using Hugging Face `Trainer`.
5. Evaluate using `predict.py` or `my_evaluate.py`

---

## Bibliography

### Core NLP / T5
- Raffel et al., 2020. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*. JMLR. https://arxiv.org/abs/1910.10683
- *T5 (language model)*. Wikipedia. https://en.wikipedia.org/wiki/T5_%28language_model%29

### Medical Text Simplification
- Devaraj et al., 2021. *Paragraph-level Simplification of Medical Texts*. NAACL-HLT 2021. https://aclanthology.org/2021.naacl-main.395/

### BioLaySumm 2025 & Radiology Summarisation
- Xiao et al., 2025. *Overview of the BioLaySumm 2025 Shared Task on Lay Summarization*. BioNLP 2025. https://aclanthology.org/2025.bionlp-1.31.pdf
- Zhang et al., 2025. *AEHRC at BioLaySumm 2025: Leveraging T5 for Lay Summarisation of Radiology Reports*. BioNLP 2025. https://aclanthology.org/2025.bionlp-share.21/
- Bechler et al., 2025. *TLPIQ at BioLaySumm: Hide and Seq, a FLAN-T5 Model for Biomedical Summarization*. BioNLP 2025. https://aclanthology.org/2025.bionlp-share.25/
- BioLaySumm 2025 Shared Task (official task description). https://biolaysumm.org/

### Related Work
- Sheang & Saggion, 2021. *Controllable Sentence Simplification with a Unified Text-to-Text Transfer Transformer*. INLG 2021. https://aclanthology.org/2021.inlg-1.38/
{index=7}


model, tokenizer = load_model(model_name="t5-small-local", use_lora=False)
