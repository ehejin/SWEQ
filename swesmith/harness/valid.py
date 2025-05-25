"""
Purpose: Transform a bunch of patches that cause bugs into a SWE-bench style dataset.

Usage: python -m swesmith.harness.valid \
    <path to directory containing patches> \
    --run_id <unique identifier for this run> \
    --max_workers <number of workers to use>
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import docker
from docker.errors import NotFound

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    FAIL_TO_PASS,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
)
from swebench.harness.docker_build import close_logger
from swebench.harness.utils import run_threadpool
from swesmith.constants import (
    KEY_IMAGE_NAME,
    KEY_MIN_PREGOLD,
    KEY_PATCH,
    KEY_TIMED_OUT,
    LOG_TEST_OUTPUT_PRE_GOLD,
    MAP_REPO_TO_SPECS,
    REF_SUFFIX,
    LOG_DIR_RUN_VALIDATION,
    TIMEOUT,
)
from swesmith.harness.grading import get_valid_report
from swesmith.harness.utils import run_patch_in_container
from swesmith.utils import get_repo_commit_from_image_name


def print_report(log_dir: Path) -> None:
    time_outs, f2p_none, f2p_some, other = 0, 0, 0, 0
    for folder in os.listdir(log_dir):
        if LOG_REPORT in os.listdir(log_dir / folder):
            with open(log_dir / folder / LOG_REPORT, "r") as f:
                report = json.load(f)
            if KEY_TIMED_OUT in report:
                time_outs += 1
            elif len(report[FAIL_TO_PASS]) > 0:
                f2p_some += 1
            elif len(report[FAIL_TO_PASS]) == 0:
                f2p_none += 1
            else:
                other += 1
    print(f"Total instances: {len(os.listdir(log_dir))}")
    print(f"- Timed out: {time_outs}")
    print(f"- Fail to pass: 0 ({f2p_none}); 1+ ({f2p_some})")
    print(f"- Other: {other}")


def run_validation(
    instance: dict, run_id: str, timeout: int = TIMEOUT, run_min_pregold: bool = False
) -> None:
    """
    Run per-instance validation. Steps are generally:
    1. Run the patch on the instance.
    2. Get the report from the test output.
    """
    instance_id = instance[KEY_INSTANCE_ID]
    valid_folder = LOG_DIR_RUN_VALIDATION / run_id

    # Clean up any previous container with the same name
    client = docker.from_env()
    container_name = f"swesmith.val.{run_id}.{instance_id}"
    try:
        old = client.containers.get(container_name)
        try:
            old.remove(force=True)
        except Exception as e:
            # container existed but removal failed; log & continue
            print(f"Warning: failed to remove old container {container_name}: {e}")
    except docker.errors.NotFound:
        # no such container, great
        pass
    val_postgold_path = (
        valid_folder / f"{instance[KEY_IMAGE_NAME]}{REF_SUFFIX}" / LOG_TEST_OUTPUT
    )
    report_path = valid_folder / instance_id / LOG_REPORT

    if run_min_pregold:
        ref_inst_id = f"{instance[KEY_INSTANCE_ID]}{REF_SUFFIX}"
        logger, timed_out = run_patch_in_container(
            {**instance, KEY_INSTANCE_ID: ref_inst_id},
            run_id,
            LOG_DIR_RUN_VALIDATION,
            timeout=timeout,
        )
        close_logger(logger)
        if timed_out:
            logger.info(f"Timed out (pre-gold) for {instance_id}.")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                f.write(json.dumps({KEY_TIMED_OUT: True, "timeout": timeout}, indent=4))
            shutil.rmtree(valid_folder / ref_inst_id)
            return

        # Copy pre-gold test output to the post-gold folder and remove the pre-gold folder
        val_postgold_path = valid_folder / instance_id / LOG_TEST_OUTPUT_PRE_GOLD
        val_postgold_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            valid_folder / ref_inst_id / LOG_TEST_OUTPUT,
            val_postgold_path,
        )
        shutil.rmtree(valid_folder / ref_inst_id)

    logger, timed_out = run_patch_in_container(
        instance,
        run_id,
        LOG_DIR_RUN_VALIDATION,
        patch=instance[KEY_PATCH],
        timeout=timeout,
    )

    if timed_out:
        logger.info(f"Timed out for {instance_id}.")
        with open(report_path, "w") as f:
            f.write(json.dumps({KEY_TIMED_OUT: True, "timeout": timeout}, indent=4))
        close_logger(logger)
        return

    # Get report from test output
    logger.info(f"Grading answer for {instance_id}...")
    report = get_valid_report(
        val_pregold_path=valid_folder / instance_id / LOG_TEST_OUTPUT,
        val_postgold_path=val_postgold_path,
        instance=instance,
    )
    logger.info(f"Report: {json.dumps(report)}")

    # Write report to report.json
    with open(report_path, "w") as f:
        f.write(json.dumps(report, indent=4))
    close_logger(logger)


def main(
    bug_patches: str,
    run_id: str,
    max_workers: int,
    timeout: int,
    timeout_ref: int,
    redo_existing: bool = False,
) -> None:
    # Bug patch should be a dict that looks like this:
    # {
    #     "instance_id": <instance_id>,
    #     "patch" / "model_patch": <bug inducing patch>,
    #     "image_name": <image_name = repo_commit>,
    # }
    print(f"[{run_id}] Running validation for {bug_patches}...")
    # 1) clean up ANY old containers leftover from previous runs
    client = docker.from_env()
    prefix = f"swesmith.val.{run_id}."
    # remove *every* existing container with that prefix
    for c in client.containers.list(all=True, filters={"name": prefix}):
        try:
            c.remove(force=True)
        except Exception:
            pass
    
    # 2) now load your JSON, group by image, mkdir the run dir, etc…
    bug_patches = json.load(open(bug_patches, "r"))
    bug_patches = [
        {
            **x,
            KEY_PATCH: x.get(KEY_PATCH, x.get(KEY_PREDICTION)),
        }
        for x in bug_patches
    ]
    print(f"Found {len(bug_patches)} candidate patches.")

    completed = []
    log_dir_parent = LOG_DIR_RUN_VALIDATION / run_id
    if not redo_existing and log_dir_parent.exists():
        for folder in os.listdir(log_dir_parent):
            # Identify completed instances (does report.json exist)
            log_report_path = log_dir_parent / folder / LOG_REPORT
            if log_report_path.exists():
                completed.append(folder)
        print(
            f"Skipping {len(completed)} completed instances... (--redo_existing to skip this step)"
        )
    bug_patches = [x for x in bug_patches if x[KEY_INSTANCE_ID] not in completed]
    log_dir_parent.mkdir(parents=True, exist_ok=True)

    # Group patches by image_name:
    image_name_to_bug_patches = dict()
    for bug_patch in bug_patches:
        image_name = bug_patch[KEY_IMAGE_NAME]
        if image_name not in image_name_to_bug_patches:
            image_name_to_bug_patches[image_name] = list()
        image_name_to_bug_patches[image_name].append(bug_patch)

    # Log
    if len(image_name_to_bug_patches) == 0:
        print("No patches to run.")
        print_report(log_dir_parent)
        return
    print("Will run validation for these images:")
    for image_name, patches in image_name_to_bug_patches.items():
        print(f"- {image_name}: {len(patches)} patches")

    # Run validation
    payloads = list()
    if timeout_ref is None:
        timeout_ref = timeout
    for image_name, bug_patches in image_name_to_bug_patches.items():
        ref_dir = LOG_DIR_RUN_VALIDATION / run_id / f"{image_name}{REF_SUFFIX}"
        repo, commit = get_repo_commit_from_image_name(image_name)
        is_min_pregold = KEY_MIN_PREGOLD in MAP_REPO_TO_SPECS[repo][commit]
        if not is_min_pregold and not os.path.exists(ref_dir):
            # Run pytest for each repo/commit to get pre-gold behavior.
            print(f"Running pre-gold for {image_name}...")
            logger, timed_out = run_patch_in_container(
                {
                    KEY_IMAGE_NAME: image_name,
                    KEY_INSTANCE_ID: f"{image_name}{REF_SUFFIX}",
                },
                run_id,
                LOG_DIR_RUN_VALIDATION,
                timeout=timeout_ref,
            )
            close_logger(logger)
            if timed_out:
                # If timed out, skip this repo/commit (remove log directory)
                print(
                    f"Timed out for {image_name}, not running validation. (Increase --timeout?)"
                )
                shutil.rmtree(ref_dir)
                continue

        # Add payloads
        for bug_patch in bug_patches:
            payloads.append((bug_patch, run_id, timeout, is_min_pregold))

    run_threadpool(run_validation, payloads, max_workers)
    print("All instances run.")
    print_report(log_dir_parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform a bunch of patches that cause bugs into a SWE-bench style dataset."
    )
    parser.add_argument(
        "bug_patches",
        type=str,
        help="Json file containing bug patches.",
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Unique identifier for this run."
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Number of workers to use."
    )
    parser.add_argument(
        "--timeout", type=int, default=TIMEOUT, help="Timeout for each run."
    )
    parser.add_argument(
        "--timeout_ref",
        type=int,
        default=None,
        help="Timeout for each run of the reference.",
    )
    parser.add_argument(
        "--redo_existing",
        action="store_true",
        help="Redo completed validation instances.",
    )
    args = parser.parse_args()
    main(**vars(args))
