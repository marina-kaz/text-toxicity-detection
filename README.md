# Toxicity Detection

Toxicity detection in text is a critical task in natural language processing aimed at identifying and mitigating harmful or offensive content within digital communication platforms. With the exponential growth of online interactions, ensuring a safe and respectful environment for users has become increasingly important. This detection process involves leveraging machine learning and deep learning techniques to automatically analyze textual data and classify it based on its toxicity level. By flagging and moderating toxic content in real-time, such systems play a crucial role in fostering healthier online communities, protecting users from harm, and promoting positive digital interactions.

## Prerequisities

Python version: 3.8
Dependencies list: [requirements.txt](requirements.txt)

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

## Training

### Data Description

Literature describes the following typed of toxicity found in the text:
* Toxic: This category encompasses general toxicity in language, including rude, disrespectful, or offensive remarks that can harm or upset others.
* Severe Toxic: Severe toxicity indicates content that goes beyond general toxicity, often containing highly abusive, extremely offensive, or explicitly harmful language, imagery, or threats.
* Obscene: Obscene content involves language or material that is sexually explicit, vulgar, or inappropriate for public consumption, often crossing societal norms or standards of decency.
* Threat: This category includes explicit threats of harm or violence towards individuals or groups, whether physical, emotional, or psychological in nature.
* Insult: Insulting content involves disparaging or derogatory remarks directed at individuals or groups, aimed at demeaning, belittling, or humiliating them.
* Identity Hate: Identity hate encompasses content that targets specific identities or characteristics, such as race, ethnicity, religion, gender, sexual orientation, or disability, with the intent to discriminate, marginalize, or incite hatred against them.

The training data must include the field with the text and columns for each respective toxicity type with the score ranging from 0 to 1. The model therefore learns to identify which types of toxicity are present in the text provided.

### Methology

Current project aims at detecting text toxicity via BERT model, trained with BCEWithLogitsLoss.
For more information refer to the [training script](scripts/train.py). Notice that parameters for training are specified [there](utils/parameters.py).

```bash
python3 scripts/train.py --path-to-data data/train.csv \
                         --model bert-base-cased \ 
                         --device cuda:0 \
                         --ckpt-path checkpoints/best_model.pth
```

## Inference

```bash
python3 scripts/infer.py --text "Heck you bro" \ 
                         --model checkpoints/best_model.pth \
                         --device cuda:0
```

## Results

### Parameter search

Apart from training, hyperparameter search was conducted to select optimal values for learning rate and weight decay. The results are as follows:

| wd / lr | 1e-5     | 1e-2     | 1e-4     | 5e-4     | 1e-3     | 5e-3     |     5e-5 |
|---------|----------|----------|----------|----------|----------|----------|----------|
| 1e-5    | 0.970    | 0.975    | 0.972    | 0.974    | 0.973    | 0.970    | 0.965    |
| 5e-5    | 0.978    | 0.982    | 0.980    | 0.981    | 0.979    | 0.976    | 0.971    |
| 1e-4    | 0.982    | 0.984    | 0.982    | 0.983    | 0.980    | 0.976    | 0.970    |
| 5e-4    | 0.984    | 0.986    | 0.984    | 0.983    | 0.980    | 0.975    | 0.968    |
| 1e-3    | 0.985    | **0.987**    | 0.985    | 0.984    | 0.980    | 0.973    | 0.966    |
| 5e-3    | 0.980    | 0.982    | 0.980    | 0.979    | 0.975    | 0.966    | 0.958    |
| 1e-2    | 0.972    | 0.973    | 0.971    | 0.969    | 0.964    | 0.954    | 0.945    |

The resulting accuracy is higher than 80% of solutions according to e.g. [this](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/leaderboard) competition.
