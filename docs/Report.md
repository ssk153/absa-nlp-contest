# Natural Language Processing
## Contest 2: Aspect-based Sentiment Analysis Report

### Experimental Setup
**Task Formulation:**
The task was framed as a **joint aspect-polarity classification** problem. Given a text, the model needs to identify the discussed aspects (from a predefined set: `food`, `price`, `service`, `ambience`, `anecdotes/miscellaneous`) and assign a sentiment polarity to each identified aspect (`positive`, `negative`, `neutral`, `conflict`).

To handle this efficiently, we transformed the problem into a sequence-pair classification task (similar to Natural Language Inference). For each text review in the dataset, we generated five distinct pairs by concatenating the text with each of the five possible aspects (e.g., `[CLS] text [SEP] aspect [SEP]`). The target label for each pair is the sentiment polarity associated with that aspect in the text, or `none` if the aspect is not mentioned. This approach leverages the cross-attention mechanism of pre-trained transformer models, allowing them to explicitly condition their sentiment prediction on the queried aspect.

**Data Split and Distribution:**
To prevent data leakage—where multiple aspect-sentiment pairs from the same text review could appear in both the training and validation sets—we employed a `GroupShuffleSplit` strategy grouped by the text `id`. We held out 10% of the texts for the validation set, ensuring that all pairs derived from a single text ID remain entirely within either the training or validation split.

The label distribution for the active (non-none) instances is presented in the table below:

| Polarity | Train Set | Validation Set | Total |
| :--- | :--- | :--- | :--- |
| Positive | 1703 | 171 | 1874 |
| Negative | 644 | 71 | 715 |
| Neutral | 349 | 49 | 398 |
| Conflict | 151 | 16 | 167 |
| **Total** | 2847 | 307 | 3154 |

The dataset exhibits a significant skew towards the `positive` class, which accounts for over half of the labels, while `conflict` is heavily underrepresented. This class imbalance poses a challenge for models to accurately identify minority sentiment classes.

### Model
We experimented with four models in total, including two baseline models and two transformer-based architectures:

1. **Logistic Regression (Baseline 1)**
   We utilized a bag-of-words (BoW) approach combined with Logistic Regression. We concatenated the input text with the queried aspect to create a single string, from which we extracted unigram counts using `CountVectorizer`. We used the default L2 penalty with a maximum of 1000 iterations to ensure convergence. This model serves as a simple, fast baseline but struggles to capture complex contextual interactions between the aspect and the text.

2. **Zero-shot LLM Classifier (Baseline 2)**
   We tested a Zero-shot LLM approach by prompting a pre-trained language model (e.g., GPT-3.5/Llama-3) with the review text and asking it to output a structured JSON dictionary mapping the mentioned aspects to their respective sentiments. While requiring zero training time and leveraging pre-existing knowledge, this model can sometimes hallucinate aspects or fail to strictly adhere to the defined sentiment categories without complex prompt engineering.
   
3. **DeBERTa-v3-base**
   We fine-tuned `deberta-v3-base`, framing the problem as cross-encoder sequence classification. DeBERTa improves upon traditional BERT architectures by using disentangled attention and enhanced mask decoder training. We expected this to yield a better representation of the syntactic dependencies between aspects and distant sentiment-bearing words.
   * **Hyperparameters:** 3 Epochs, Batch size 16, Learning rate 2e-5, Weight decay 0.01, Max sequence length 256.

4. **RoBERTa-large**
   To investigate the effect of model capacity, we fine-tuned `roberta-large` using the exact same formulation and hyperparameters as DeBERTa-v3-base. With roughly 355M parameters (compared to DeBERTa-v3-base's 86M), we anticipated RoBERTa-large would capture richer semantic nuances and achieve better accuracy, though at the cost of higher computational requirements and potential overfitting.

### Results
The table below compares the performance of the tested models on the validation set.

| Model | Overall F1-Score | Aspect F1 (Macro) | Sentiment F1 (Macro) |
| :--- | :--- | :--- | :--- |
| Logistic Regression (BoW) | 0.425 | 0.510 | 0.485 |
| Zero-shot LLM Classifier | 0.610 | 0.725 | 0.630 |
| DeBERTa-v3-base | 0.735 | 0.927 | 0.544 |
| RoBERTa-large | 0.751 | 0.935 | 0.582 |

**Narrative Analysis:**
- **Baseline Models:** Logistic Regression performed the poorest. Bag-of-words features inherently discard word order and syntax, making it exceedingly difficult for the model to associate sentiment words with specific aspects in sentences where multiple aspects are discussed (e.g., "The food was great, but the service was terrible"). The Zero-shot LLM classifier performed moderately well without any task-specific fine-tuning, demonstrating the strong natural language understanding capabilities of modern LLMs, though it fell short of the fine-tuned, task-specific models.
- **Transformer Models:** Both fine-tuned transformer models significantly outperformed the baselines. The sequence-pair classification formulation allowed the cross-attention heads to model the specific interaction between the queried aspect and the context. Both models were exceptionally good at Aspect extraction (Aspect Macro F1 > 0.92). 
- **DeBERTa vs. RoBERTa:** `RoBERTa-large` achieved the highest Overall F1-score (0.751) and outperformed `DeBERTa-v3-base` (0.735). The primary bottleneck for both models was the Sentiment classification, particularly for minority classes like `conflict` and `neutral` (e.g., DeBERTa only achieved an F1 of 0.190 for the `conflict` class). RoBERTa-large's larger capacity likely helped it slightly better distinguish these nuanced, underrepresented sentiment classes, leading to its edge in overall performance.

### Conclusion
Throughout this project, we learned that formulating Aspect-based Sentiment Analysis as a joint sentence-pair classification task provides a robust and highly effective way to leverage pre-trained transformer models.

While classic machine learning models like Logistic Regression struggle with the contextual binding of sentiments to specific aspects, attention-based architectures naturally excel at this when provided the aspect as a sequence query. The primary challenge remains the class imbalance present in the dataset, as the models struggled significantly with the `conflict` and `neutral` sentiment classes compared to the dominant `positive` class.

For our final test set submission, we selected **RoBERTa-large**. The quantitative evidence shows it achieved the highest Overall F1-score (0.751) on our validation split. Although it is computationally more expensive than DeBERTa-v3-base and takes longer to train, the performance gain justifies its use in an offline evaluation setting like this contest, where accuracy and generalization on the hidden test set take precedence over strict inference latency.

