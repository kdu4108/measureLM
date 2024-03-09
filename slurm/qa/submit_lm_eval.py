import subprocess
import json
from typing import List, Dict

from transformers import AutoTokenizer

model_id_and_quantize_tuples = [
    ("EleutherAI/pythia-70m-deduped", False),
    ("EleutherAI/pythia-410m-deduped", False),
    ("EleutherAI/pythia-1.4b-deduped", False),
    ("EleutherAI/pythia-6.9b-deduped", True),
    ("EleutherAI/pythia-12b-deduped", True),
]

for model_id, do_quantize in model_id_and_quantize_tuples:
    cmd = [
        "sbatch",
        "slurm/qa/submit_lm_eval.cluster",
        model_id,
    ]
    print(cmd)
    subprocess.check_call(cmd)
