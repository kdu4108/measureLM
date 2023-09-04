import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from measureLM import helpers

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

def visualize_token_ranks(scored_tokens, tokens, prompts):
    for prompt_id, prompt_token_ranks in scored_tokens.items():
        all_tokens_ranks = np.array(list(prompt_token_ranks.values()))
        layers = np.array(list(prompt_token_ranks.keys()))

        fig, ax = plt.subplots(1, 1, figsize=(6, 2), gridspec_kw={'hspace': 0.4})
        labelsize, titlefont, markersize = 10, 12, 8

        all_tokens_ranks = np.stack(all_tokens_ranks)
        all_tokens_ranks = np.swapaxes(all_tokens_ranks, 0, 1)

        lines = []
        for tokens_ranks in all_tokens_ranks:
            line, = ax.plot(layers, tokens_ranks, marker=".", markersize=markersize)
            lines.append(line)

        if tokens is not None:
            ax.legend(lines, tokens, loc='upper left', frameon=False)
        if prompts is not None:
            prompt = prompts[0]
            if len(prompts) > 1:
                prompt = prompts[prompt_id]
            ax.set_title(prompt, fontsize=titlefont, color="black", loc='center')

        ax.set_xlabel('layers', fontsize=labelsize)
        ax.set_ylabel('reciprocal rank', fontsize=labelsize)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        #fig.show()
        #fig.savefig(helpers.ROOT_DIR / "results" / "plots" / "test.pdf", dpi=200, bbox_inches='tight')

if __name__ == "__main__":
    pass