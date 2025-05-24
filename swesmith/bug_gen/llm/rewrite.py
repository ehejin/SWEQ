"""
Purpose: Given a repository, blank out various functions/classes, then ask the model to rewrite them.

Usage: python -m swesmith.bug_gen.llm.rewrite \
    --model <model> \
    --type <entity_type> \
    repo  # e.g., tkrajina__gpxpy.09fc46b3

Where model follows the litellm format.

Example:

python -m swesmith.bug_gen.llm.rewrite tkrajina__gpxpy.09fc46b3 --model claude-3-7-sonnet-20250219 --type class
"""

import ast
import argparse
import json
import litellm
import logging
import os
import random
import shutil
import subprocess
import torch
import yaml
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import completion
from litellm.cost_calculator import completion_cost
from swesmith.bug_gen.criteria import filter_min_simple_complexity
from swesmith.bug_gen.llm.utils import (
    PROMPT_KEYS,
    extract_code_block,
    strip_function_body,
)
from swesmith.bug_gen.utils import (
    ENTITY_TYPES,
    BugRewrite,
    CodeEntity,
    apply_code_change,
    extract_entities_from_directory,
    get_patch,
)
from swesmith.constants import LOG_DIR_BUG_GEN, PREFIX_BUG, PREFIX_METADATA
from swesmith.utils import clone_repo
from swesmith.vllm_utils import get_vllm_model, generate_response
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Any


LM_REWRITE = "lm_rewrite"

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
litellm.suppress_debug_info = True
random.seed(24)


def get_function_signature(node):
    """Generate the function signature as a string."""
    args = [ast.unparse(arg) for arg in node.args.args]  # For Python 3.9+
    args_str = ", ".join(args)
    return f"def {node.name}({args_str})"


def main(
    repo: str,
    config_file: str,
    model: str,
    entity_type: str,
    n_workers: int,
    redo_existing: bool = False,
    max_bugs: int = None,
    **kwargs,
):
    configs = yaml.safe_load(open(config_file))
    print(f"Cloning {repo}...")
    clone_repo(repo)
    print(f"Extracting entities from {repo}...")
    candidates = [
        x
        for x in extract_entities_from_directory(repo, entity_type)
        if filter_min_simple_complexity(x, 3)
    ]
    if kwargs["debug"]:
        candidates = candidates[:5]
    if max_bugs:
        random.shuffle(candidates)
        candidates = candidates[:max_bugs]

    # Set up logging
    # log_dir = LOG_DIR_BUG_GEN / repo
    time_date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_DIR_BUG_GEN / "lm_rewrite" / f"{repo}__{model.replace('/', '_')}__{time_date_str}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging bugs to {log_dir}")
    if not redo_existing:
        print("Skipping existing bugs.")

    # Initialize vLLM model if using Qwen
    if "Qwen" in model:
        print(f"Loading {model}...")
        vllm_model = get_vllm_model(
            model=model,
            max_model_len=kwargs["max_model_len"],
            enforce_eager=False,
            num_gpus=torch.cuda.device_count(),
            gpu_memory_utilization=kwargs["gpu_memory_utilization"],
        )
    else:
        vllm_model = None

    def _process_candidate(candidate: CodeEntity) -> dict[str, Any]:
        bug_dir = (
            log_dir / candidate.file_path.replace("/", "__") / candidate.src_node.name
        )
        if not redo_existing:
            if bug_dir.exists() and any(
                [
                    str(x).startswith(f"{PREFIX_BUG}__{configs['name']}")
                    for x in os.listdir(bug_dir)
                ]
            ):
                return {"n_bugs_generated": 0, "cost": 0.0}

        try:
            # Blank out the function body
            blank_function = BugRewrite(
                rewrite=strip_function_body(candidate.src_code),
                explanation="Blanked out the function body.",
                strategy=LM_REWRITE,
            )
            apply_code_change(candidate, blank_function)
        except Exception:
            return {"n_generation_failed": 1, "cost": 0.0}

        # Get prompt content
        prompt_content = {
            "func_signature": get_function_signature(candidate.src_node),
            "func_to_write": blank_function.rewrite,
            "file_src_code": open(candidate.file_path).read(),
        }

        # Generate a rewrite
        messages = [
            {
                "content": configs[k].format(**prompt_content),
                "role": "user" if k != "system" else "system",
            }
            for k in PROMPT_KEYS
            if k in configs
        ]
        messages = [x for x in messages if x["content"]]

        if vllm_model is not None:
            response = generate_response(
                chat=messages,
                vllm_model=vllm_model,
                model=model,
                n=1,
                max_tokens=kwargs["max_new_tokens"],
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                top_k=kwargs["top_k"],
                repetition_penalty=kwargs["repetition_penalty"],
            )
            content = response[0]
            cost = 0.0  # Can't compute cost for vLLM
        else:
            response: Any = completion(model=model, messages=messages, n=1, temperature=0)
            choice = response.choices[0]
            message = choice.message
            content = message.content
            cost = completion_cost(completion_response=response)

        # Revert the blank-out change to the current file and apply the rewrite
        code_block = extract_code_block(content)
        explanation = content.split("```", 1)[0].strip()

        subprocess.run(
            f"cd {repo}; git reset --hard",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        rewrite = BugRewrite(
            rewrite=code_block,
            explanation=explanation,
            strategy=LM_REWRITE,
            cost=cost,
            output=content,
        )
        apply_code_change(candidate, rewrite)
        patch = get_patch(repo, reset_changes=True)

        # Log the bug
        bug_dir.mkdir(parents=True, exist_ok=True)
        uuid_str = f"{configs['name']}__{rewrite.get_hash()}"
        metadata_path = f"{PREFIX_METADATA}__{uuid_str}.json"
        bug_path = f"{PREFIX_BUG}__{uuid_str}.diff"

        with open(bug_dir / metadata_path, "w") as f:
            json.dump(rewrite.to_dict(), f, indent=2)
        with open(bug_dir / bug_path, "w") as f:
            f.write(patch)
        print(f"Wrote bug to {bug_dir / bug_path}")

        return {"n_bugs_generated": 1, "cost": cost}

    stats = {"cost": 0.0, "n_bugs_generated": 0, "n_generation_failed": 0}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_process_candidate, candidate) for candidate in candidates
        ]

        with logging_redirect_tqdm():
            with tqdm(total=len(candidates), desc="Candidates") as pbar:
                for future in as_completed(futures):
                    cost = future.result()
                    for k, v in cost.items():
                        stats[k] += v
                    pbar.set_postfix(stats, refresh=True)
                    pbar.update(1)

    shutil.rmtree(repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bug patches for functions/classes/objects in a repository."
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo", type=str, help="Repository to generate bug patches for."
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the configuration file.", required=True
    )
    parser.add_argument("--model", type=str, help="Model to use for rewriting.")
    parser.add_argument(
        "--type",
        dest="entity_type",
        type=str,
        choices=list(ENTITY_TYPES.keys()),
        help="Type of entity to generate bug patches for.",
    )
    parser.add_argument(
        "--n_workers", type=int, help="Number of workers to use", default=1
    )
    parser.add_argument(
        "--redo_existing", action="store_true", help="Redo existing bugs."
    )
    parser.add_argument(
        "--max_bugs", type=int, help="Maximum number of bugs to generate."
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    # vLLM Sampling Parameters
    parser.add_argument("--max_model_len", type=int, default=8192, help="Max model length")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Max new tokens")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="GPU memory utilization")
    parser.add_argument("--temperature", type=float, help="Sampling temperature", default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")

    args = parser.parse_args()
    main(**vars(args))
