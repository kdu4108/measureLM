from typing import Optional
from requests.exceptions import HTTPError

import wandb


def load_artifact_from_wandb(artifact_name: str, save_dir: str, verbose=True) -> Optional[str]:
    # Load artifact if it exists.
    # This will try to download the entire directory of the data_id,
    #  including ALL logged subdirectories (e.g. the qa_triples, tokenized_qa_triples, and encoded_qa_triples)
    #  in addition to the dataframe pickles.
    artifact, artifact_files = None, None
    try:
        artifact = wandb.run.use_artifact(f"{artifact_name}:latest")
        artifact.download(root=save_dir)
        artifact_files = [file.path for file in artifact.manifest.entries.values()]
        if verbose:
            print("Downloading artifact files:", artifact_files)
    except (wandb.CommError, HTTPError) as e:
        # HTTPError occurs when the artifact has been deleted, this appears to be a bug with wandb. Catching this exception for now.
        print(type(e), e)

    return artifact, artifact_files
