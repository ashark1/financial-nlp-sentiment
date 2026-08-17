# Model card

## Baseline

TF-IDF vectorisation followed by class-balanced logistic regression.

## Transformer comparison

FinBERT (`ProsusAI/finbert`) is available as an optional inference path for financial sentiment classification.

## Intended use

Educational and portfolio use for classifying short pieces of financial text into positive, neutral, and negative sentiment.

## Not intended for

- investment advice
- automated trading
- estimating future share-price movement
- high-stakes financial decisions

## Data

The repository includes a small synthetic dataset purely to demonstrate the training pipeline. Real evaluation is designed to use Financial PhraseBank through the dataset loader.

## Evaluation

The baseline training command writes evaluation metrics to `artifacts/metadata.json`. Model comparison writes a separate JSON file so benchmark results come from an actual run rather than being hard-coded.

## Known limitations

The baseline does not explicitly model financial entities, sarcasm, long context, or temporal information. FinBERT improves domain specificity but still does not make sentiment equivalent to future price movement.
