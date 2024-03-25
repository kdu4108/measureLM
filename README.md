# Persuasiveness and Susceptibility in In-Context Learning

## Getting started

First, clone the repo:
```
git clone git@github.com:kdu4108/measureLM.git
```

Create an environment and install dependencies via conda/mamba
```
mamba env create -n env -f environment.yml
mamba activate env
```
OR virtualenv/pip:
```
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Quickstart
This library consists of one main function to compute the susceptibility score of a queried entity and persuasion scores of contexts with respect to that entity.
See the code snippet below (also found in `example.py`) for an example of how to run this function.

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from measuring.estimate_probs import estimate_cmi

query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
entity = "The Dark Knight"
contexts = [
    "Here's a movie review: 'The movie was the greatest.'",
    "Here's a movie review: 'The movie was great.'",
    "Here's a movie review: 'The movie was not great.'",
    "Here's a movie review: 'The movie was not the greatest.'",
    "Here's a movie review: 'The movie was terrific and I loved it.'",
    "Here's a movie review: 'The movie was awful and I hated it.'",
    "Here's a movie review: 'The movie was the best best best movie I've ever seen.'",
    "Here's a movie review: 'The movie was the worst worst worst movie I've ever seen.'",
    "Here's a movie review: 'The movie was meh.'",
    "Here's a movie review: 'The movie was kinda meh.'",
    "Here's a movie review: 'blah blah blah.'",
]
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "EleutherAI/pythia-70m-deduped"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
)

query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
susceptibility_score, _, persuasion_scores = estimate_cmi(query, entity, contexts, model, tokenizer)
print(f"Susceptibility score for entity '{entity}': {susceptibility_score}.")
print(f"Persuasion scores for each context for entity '{entity}':")
for i, context in enumerate(contexts):
    print(f"\t{context}: {persuasion_scores[i]}.")
```

## Running experiments from the paper
Here are the steps to regenerate the experiments in the paper. The key steps are (1) accessing or regenerating the dataset and (2) running the main entry point, `susceptibility_scores.py`.

### Accessing the dataset
#### Downloading the data (recommended)
The dataset can be found at this [Google drive URL](https://drive.google.com/file/d/1y0g9N4aPpEP2_EcZo3u3hMCiTXzdqsvc/view?usp=sharing).
Download the file and copy it to `data/YagoECQ/yago_qec.json`.
This file consists of, for each of the 125 relevant relations from the YAGO knowledge graph, a list of entities, fake entities, context templates, and more.

#### Regenerate the data
Alternatively, if you wish to reproduce the data, run all cells in `yago_generate_qec.ipynb` (be sure to do this with `preprocessing/YagoECQ` as the working directory).

### Running a single experiment
The main entry point to run a single experiment is `susceptibility_scores.py`. The most important arguments to this script are:
* `DATASET_NAME` (positional argument, determines which dataset to run the experiment on. Must exactly match the name of a dataset defined in `preprocessing/datasets.py`).
* `--RAW_DATA_PATH` (the raw data corresponding to `DATASET_NAME`.)
* `--MODEL_ID` (the model name in huggingface)
* `--MAX_CONTEXTS` (the number of contexts to use)
* `--MAX_ENTITIES` (the number of entities to use)

The remaining arguments (visible via `python susceptibility_scores.py --help`) are either dataset-specific (e.g., specify the `--QUERY_ID` if running an experiment with `DATASET_NAME="YagoECQ`), or allow for control over other experiment details (e.g., which query types to use, the model's batch size for inference, how to sample entities, etc.).

An example command is:
```
python susceptibility_scores.py YagoECQ -Q http://schema.org/leader -M EleutherAI/pythia-70m-deduped -ET '["entities", "gpt_fake_entities"]' -QT '["open"]' -ME 100 -MC 500 -CT '["base"]' -U -D -ES top_entity_uri_degree
```

### Running the full suite of experiments
If you have access to a slurm cluster, you can kick off the full suite of Yago experiments via
```
python slurm/susceptibility_scores/batch_submit_susceptibility_scores.py
```

Friend-enemy experiments can be run via
```
python slurm/susceptibility_scores/batch_submit_susceptibility_scores-friendenemy.py
```

## Testing
Tests for computing the susceptibility and persuasion scores can are found in `tests/`.
To run tests, run
```
python -m pytest tests/
```
