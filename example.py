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


"""
Output:

```
Susceptibility score for entity 'The Dark Knight': 0.011675727006159483.
Persuasion scores for each context for entity 'The Dark Knight':
        Here's a movie review: 'The movie was the greatest.': 0.005240366315139609.
        Here's a movie review: 'The movie was great.': 0.002376393193422723.
        Here's a movie review: 'The movie was not great.': 0.004375192089839939.
        Here's a movie review: 'The movie was not the greatest.': 0.003711586225890182.
        Here's a movie review: 'The movie was terrific and I loved it.': 0.00896091933535683.
        Here's a movie review: 'The movie was awful and I hated it.': 0.009592836733846238.
        Here's a movie review: 'The movie was the best best best movie I've ever seen.': 0.03600051247311878.
        Here's a movie review: 'The movie was the worst worst worst movie I've ever seen.': 0.019317693383659938.
        Here's a movie review: 'The movie was meh.': 0.0047939527011736285.
        Here's a movie review: 'The movie was kinda meh.': 0.019275893108693785.
        Here's a movie review: 'blah blah blah.': 0.014787651507612682.
```
"""
