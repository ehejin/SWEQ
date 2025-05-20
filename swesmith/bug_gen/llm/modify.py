"""
Purpose: Given a repository, generate bug patches for functions/classes/objects in the repository.

Usage: python -m swesmith.bug_gen.llm.modify \
    --n_bugs <n_bugs> \
    --config_file <config_file> \
    --model <model> \
    --type <entity_type>
    repo  # e.g., tkrajina__gpxpy.09fc46b3

Where model follows the litellm format.

Example:

python -m swesmith.bug_gen.llm.modify tkrajina__gpxpy.09fc46b3 --config_file configs/bug_gen/class_basic.yml --model claude-3-7-sonnet-20250219 --n_bugs 1
"""

import argparse
import shutil
import jinja2
import json
import litellm
import logging
import os
import random
import yaml

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dotenv import load_dotenv
from litellm import completion
from litellm.cost_calculator import completion_cost
from swesmith.bug_gen.criteria import MAP_KEY_TO_CRITERIA
from swesmith.bug_gen.llm.utils import PROMPT_KEYS, extract_code_block
from swesmith.bug_gen.utils import (
    ENTITY_TYPES,
    BugRewrite,
    CodeEntity,
    apply_code_change,
    extract_entities_from_directory,
    get_patch,
)
from swesmith.constants import (
    LOG_DIR_BUG_GEN,
    ORG_NAME,
    PREFIX_BUG,
    PREFIX_METADATA,
)
from swesmith.utils import clone_repo, does_repo_exist
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Any, Optional

from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"


load_dotenv(dotenv_path=os.getenv("SWEFT_DOTENV_PATH"))

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
litellm.suppress_debug_info = True

def get_vllm_model(model, max_model_len, enforce_eager, num_gpus=1, gpu_memory_utilization=None):
    model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    memory_per_model = 0.9 if gpu_memory_utilization is None else gpu_memory_utilization
    return LLM(
        model=model,
        dtype="float16",
        quantization="bitsandbytes",
        load_format="bitsandbytes", 
        enable_lora=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=memory_per_model,
        tensor_parallel_size=num_gpus,
        enforce_eager=True #enforce_eager NOTE: SKIP for NOW
    )

def generate_response(
    chat,
    vllm_model: Optional[LLM] = None,
    peft_dir: Optional[str] = None,
    **generation_kwargs,
):
    """Generate a response from the assistant model.
    
    Args:
        chat: Either a single message or list of messages for batch processing
        local_model: Optional HuggingFace model
        local_tokenizer: Optional HuggingFace tokenizer
        is_api_model: Whether to use API model
        vllm_model: Optional vLLM model
        **generation_kwargs: Additional generation parameters
        
    Returns:
        str or List[str]: Generated response(s)
    """
    assistant_model_name = generation_kwargs['model']
    if isinstance(chat, str):
        chat = [{"role": "user", "content": chat}]
    elif isinstance(chat[0], str):
        chat = [[{"role": "user", "content": message}] for message in chat]

    generation_kwargs.pop("model", None)
    sampling_params = convert_to_sampling_params(generation_kwargs)
    lora_request = LoRARequest("interactive_adapter", 1, peft_dir) if peft_dir else None
    responses = vllm_model.chat(
        messages=chat,
        sampling_params=sampling_params,
        lora_request=lora_request
    )
    
    results = []
    for response_set in responses:
        if response_set.outputs:
            results.append(response_set.outputs[0].text)
            # results.extend([c.text for c in response_set.outputs])
        else:
            results.append("")  # Fallback for empty responses
    return results


def convert_to_sampling_params(generation_kwargs: dict) -> SamplingParams:
    """Convert generation kwargs to vllm SamplingParams."""

    # Valid sampling parameter keys from SamplingParams class
    valid_params = {
        "n",
        "best_of",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "stop_token_ids",
        "bad_words",
        "ignore_eos",
        "max_tokens",
        "min_tokens",
        "logprobs",
        "prompt_logprobs",
        "detokenize",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "truncate_prompt_tokens",
    }

    # Filter valid params and log unmapped ones
    sampling_kwargs = {}
    for key, value in generation_kwargs.items():
        if key in valid_params:
            sampling_kwargs[key] = value
        else:
            print(
                f"Warning: Parameter '{key}' not found in VLLM-supported sampling parameters"
            )

    # Create SamplingParams object
    return SamplingParams.from_optional(**sampling_kwargs)

def gen_bug_from_code_lm(
    candidate: CodeEntity, configs: dict, n_bugs: int, model: str, vllm_model=None
) -> list[BugRewrite]:
    """
    Given the source code of a function, return `n` bugs with an LM
    """

    def format_prompt(prompt: str | None, config: dict, candidate: CodeEntity) -> str:
        if not prompt:
            return ""
        env = jinja2.Environment()

        def jinja_shuffle(seq):
            result = list(seq)
            random.shuffle(result)
            return result

        env.filters["shuffle"] = jinja_shuffle
        template = env.from_string(prompt)
        return template.render(**asdict(candidate), **config.get("parameters", {}))

    def get_role(key: str) -> str:
        if key == "system":
            return "system"
        return "user"

    bugs = []
    messages = [
        {"content": format_prompt(configs[k], configs, candidate), "role": get_role(k)}
        for k in PROMPT_KEYS
    ]
    # Remove empty messages
    messages = [x for x in messages if x["content"]]

    if vllm_model is not None:
        response = generate_response(
                    chat=messages,
                    vllm_model=vllm_model,
                    model=model,
                    n=n_bugs,
                    temperature=1,
                )
    else:
        response: Any = completion(model=model, messages=messages, n=n_bugs, temperature=1)
    
    if vllm_model is None:
        response_iter = response.choices
    else:
        response_iter = response 

    for choice in response_iter:
        if vllm_model is None:
            message = choice.message
            content = message.content
            cost = (
                completion_cost(completion_response=response, model=model) / n_bugs
                    if vllm_model is None
                    else 0.0  # Can't compute cost for vLLM
        )
        else:
            content = choice
            cost = 0
        explanation = (
            content.split("Explanation:")[-1].strip()
            if "Explanation" in content
            else content.split("```")[-1].strip()
        )
        bugs.append(
            BugRewrite(
                rewrite=extract_code_block(content),
                explanation=explanation,
                cost=cost,
                output=content,
                strategy="llm",
            )
        )
    return bugs


def main(
    config_file: str,
    entity_type: str,
    model: str,
    n_bugs: int,
    repo: str,
    *,
    n_workers: int = 1,
    **kwargs,
):
    # Check arguments
    assert does_repo_exist(repo), f"Repository {repo} does not exist in {ORG_NAME}."
    assert os.path.exists(config_file), f"{config_file} not found"
    assert n_bugs > 0, "n_bugs must be greater than 0"
    configs = yaml.safe_load(open(config_file))
    assert all(key in configs for key in PROMPT_KEYS + ["criteria", "name"]), (
        f"Missing keys in {config_file}"
    )

    # Clone repository, identify valid candidates
    print("Cloning repository...")
    clone_repo(repo)
    print("Extracting candidates...")
    candidates = extract_entities_from_directory(repo, entity_type)
    print(f"{len(candidates)} candidates found for {entity_type} in {repo}")
    candidates = [x for x in candidates if MAP_KEY_TO_CRITERIA[configs["criteria"]](x)]
    print(f"{len(candidates)} candidates passed criteria")
    if not candidates:
        print(f"No candidates found for {entity_type} in {repo}.")
        return

    print(f"Generating bugs for {entity_type} in {repo} using {model}...")
    '''if not kwargs.get("yes", False):
        if input("Proceed with bug generation? (y/n): ").lower() != "y":
            return'''

    # Set up logging
    log_dir = LOG_DIR_BUG_GEN / repo
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging bugs to {log_dir}")

    if "qwen" in model:
        print("Building QWEN model!!")
        vllm_model = get_vllm_model(
            model=model,
            max_model_len=8192,
            enforce_eager=False,
            num_gpus=4,  
        )
    else:
        vllm_model = None

    def _process_candidate(candidate: CodeEntity, vllm_model):
        # Run bug generation
        bugs = gen_bug_from_code_lm(candidate, configs, n_bugs, model, vllm_model)
        cost, n_bugs_generated, n_generation_failed = sum([x.cost for x in bugs]), 0, 0

        for bug in bugs:
            # Create artifacts
            bug_dir = (
                log_dir
                / candidate.file_path.replace("/", "__")
                / candidate.src_node.name
            )
            bug_dir.mkdir(parents=True, exist_ok=True)
            uuid_str = f"{configs['name']}__{bug.get_hash()}"
            metadata_path = f"{PREFIX_METADATA}__{uuid_str}.json"
            bug_path = f"{PREFIX_BUG}__{uuid_str}.diff"

            try:
                with open(bug_dir / metadata_path, "w") as f:
                    json.dump(bug.to_dict(), f, indent=2)
                apply_code_change(candidate, bug)
                patch = get_patch(repo, reset_changes=True)
                if not patch:
                    raise ValueError("Patch is empty.")
                with open(bug_dir / bug_path, "w") as f:
                    f.write(patch)
            except Exception as e:
                print(
                    f"Error applying bug to {candidate.src_node.name} in {candidate.file_path}: {e}",
                )
                # import traceback
                # print(f"Traceback:\n{''.join(traceback.format_exc())}")
                (bug_dir / metadata_path).unlink(missing_ok=True)
                n_generation_failed += 1
                continue
            else:
                n_bugs_generated += 1
        return {
            "cost": cost,
            "n_bugs_generated": n_bugs_generated,
            "n_generation_failed": n_generation_failed,
        }

    stats = {"cost": 0.0, "n_bugs_generated": 0, "n_generation_failed": 0}
    with tqdm(total=len(candidates), desc="Candidates") as pbar:
        for cand in candidates:
            cost = _process_candidate(cand, vllm_model)   # ← run directly
            for k, v in cost.items():
                stats[k] += v
            pbar.set_postfix(stats, refresh=True)
            pbar.update(1)

    '''with ThreadPoolExecutor(max_workers=n_workers) as executor:
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
                    pbar.update(1)'''

    shutil.rmtree(repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo",
        type=str,
        help="Name of a SWE-smith repository to generate bugs for.",
    )
    parser.add_argument(
        "--type",
        dest="entity_type",
        type=str,
        choices=list(ENTITY_TYPES.keys()),
        default="func",
        help="Type of entity to generate bugs for.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for bug generation",
        default="openai/gpt-4o",
    )
    parser.add_argument(
        "--n_bugs",
        type=int,
        help="Number of bugs to generate per entity",
        default=1,
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Configuration file containing bug gen. strategy prompts",
        required=True,
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument(
        "--n_workers", type=int, help="Number of workers to use", default=1
    )
    args = parser.parse_args()
    main(**vars(args))
