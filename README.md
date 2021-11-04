# Language Models - Tunable Layers

Do we really need to tune all layers in a language model? Here are a few examples where tuning just some top layers is enough to achieve the same performance as tuning the complete model.

## How many layers should we tune?

### BERT base
| Layers   | CoLA     | SST-2    | MRPC     |  RTE     | STS-B    |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 0 (head) | 40.1     | 85.1     | 73.2     | 64.6     | 80.0     |
| 1        | 53.0     | 90.5     | 79.4     | 65.4     | 85.2     |
| 2        | 54.3     | 91.3     | 83.6     | 66.0     | 87.2     |
| 3        | 55.6     | 92.0     | __86.5__ | 66.4     | 88.3     |
| 4        | __59.0__ | 92.1     | 86.3     | 66.8     | 88.9     |
| 5        | __59.0__ | 92.1     | 85.3     | 67.4     | 89.2     |
| 11 (full)| __59.0__ | __92.2__ | 85.6     | __68.5__ | __89.4__ |

The key takeaway is that sometimes it's enough to tune just 4 or 5 top layers of a language model ✌️️

- I used 'bert-base-cased' model.
- Scores are recorded on the development set.
- Metrics: Matthews (MCC) for CoLA, Spearman's rank for STS-B, accuracy for others.
- Tuning only the top needs more epochs and doesn't overfit so fast with small datasets.
- Zero means that we're only training the prediction head, eleven layers refer to the full model.

```python
from glue_benchmark import GLUE_COLA
from encoders import LanguageModel
from task_heads import TextClassifier, TextRegressor
from trainer import ModelTrainer

# 1. load any GLUE corpus (e.g. Corpus of Linguistic Acceptability)
corpus = GLUE_COLA()

print(corpus)

# 2. pick a language model (BERT, SpanBERT, Electra or Ernie) and select tunable layers
# "all" for full finetuning, 0 for training just the prediction head
language_model = LanguageModel("bert-base-cased",
                               tunable_layers=4)

# 3. use classification or regression head 
# classification for CoLA and other tasks, regression for STSB
classifier = TextClassifier(encoder=language_model,
                            num_classes=corpus.num_classes)

# 4. create model trainer
trainer = ModelTrainer(model=classifier,
                      corpus=corpus)

# 5. start training
trainer.train(learning_rate=5e-5, # use slightly larger lr than usual
              batch_size=16,
              epochs=15,
              shuffle_data=True)
```


