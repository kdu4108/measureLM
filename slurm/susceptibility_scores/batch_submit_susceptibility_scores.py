import subprocess
import json

RUN_LOCALLY = True
YAGO_QEC_PATH = "data/YagoECQ/yago_qec.json"  # assuming you are running from the root project directory

with open(YAGO_QEC_PATH) as f:
    yago_qec = json.load(f)

dataset_names_and_rdps = [("YagoECQ", YAGO_QEC_PATH)]
seeds = [0]  # val every 10

if RUN_LOCALLY:
    model_id_and_quantize_tuples = [("EleutherAI/pythia-70m-deduped", False)]
    max_contexts = [10]
    max_entities = [5]
    query_ids = list(yago_qec.keys())[:5]
else:
    model_id_and_quantize_tuples = [("EleutherAI/pythia-6.9b-deduped", True)]
    max_contexts = [450]
    max_entities = [90]
    query_ids = list(yago_qec.keys())

cap_per_type = False
ablate = False

for ds, rdp in dataset_names_and_rdps:
    for seed in seeds:
        for model_id, do_quantize in model_id_and_quantize_tuples:
            for qid in query_ids:
                for mc in max_contexts:
                    for me in max_contexts:
                        if RUN_LOCALLY:
                            subprocess.run(
                                [
                                    "python",
                                    "susceptibility_scores.py",
                                    f"{ds}",
                                    "-P",
                                    rdp,
                                    "-S",
                                    f"{seed}",
                                    "-M",
                                    f"{model_id}",
                                    "-Q",
                                    f"{qid}",
                                    "-MC",
                                    f"{mc}",
                                    "-ME",
                                    f"{me}",
                                ]
                                + (["-Q"] if do_quantize else [])
                                + (["-A"] if ablate else [])
                                + (["-T"] if cap_per_type else [])
                            )
                        else:
                            subprocess.check_call(
                                [
                                    "sbatch",
                                    "slurm/susceptibility_scores/submit_susceptibility_score.cluster",
                                    f"{ds}",
                                    f"{rdp}",
                                    f"{seed}",
                                    f"{model_id}",
                                    f"{qid}",
                                    f"{mc}",
                                    f"{me}",
                                ]
                                + (["-Q"] if do_quantize else [])
                                + (["-A"] if ablate else [])
                                + (["-T"] if cap_per_type else [])
                            )
