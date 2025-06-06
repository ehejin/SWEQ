import hashlib
import os
import platform
import random
import string
import subprocess

from ghapi.all import GhApi
from pathlib import Path
from swesmith.constants import MAP_REPO_TO_SPECS, ORG_NAME, LOG_DIR_ENV_RECORDS


def get_arch_and_platform() -> tuple[str, str]:
    """
    Get the architecture and platform for the current machine.
    """
    arch = "x86_64" if platform.machine() not in {"aarch64", "arm64"} else "arm64"
    if arch == "x86_64":
        pltf = "linux/x86_64"
    elif arch == "arm64":
        pltf = "linux/arm64/v8"
    else:
        raise ValueError(f"Invalid architecture: {arch}")
    return arch, pltf


def get_image_name(repo: str, commit: str, arch: str | None = None) -> str:
    """
    Get the docker image ID for a repository at a specific commit.
    """
    arch = arch or get_arch_and_platform()[0]
    return f"swesmith.{arch}.{repo.replace('/', '__').lower()}.{commit[:8]}"


def get_repo_commit_from_image_name(image_name: str) -> tuple[str, str]:
    """
    Get the repository and commit from a docker image ID.
    """
    # Parsing supports repos with '.' in their name
    image_name = image_name.split(".", 2)[-1]
    repo = image_name.rsplit(".", 1)[0].replace("__", "/")
    partial_commit = image_name.rsplit(".", 1)[-1]
    for repo_name in MAP_REPO_TO_SPECS:
        # Hack because docker image_name must be lowercase
        if repo_name.lower() == repo:
            repo = repo_name
            break
    commit = get_full_commit(repo, partial_commit)
    return repo, commit


def get_env_yml_path(repo: str, commit: str) -> str:
    """
    Get the path to the environment.yml file for a repository at a specific commit.
    """
    if len(commit) != 40:
        raise ValueError(
            f"Must provide full commit hash, not partial commit ({commit})"
        )
    return f"{LOG_DIR_ENV_RECORDS}/sweenv_{repo.replace('/', '__')}_{commit}.yml"


def get_full_commit(repo, partial_commit) -> str:
    """
    Get the full commit hash for a repository at a specific commit.
    """
    for commit in MAP_REPO_TO_SPECS[repo]:
        if commit.startswith(partial_commit):
            return commit

    raise ValueError(f"Commit {partial_commit} not found for repository {repo}.")


def get_repo_name(repo, commit) -> str:
    """
    Get the SWE-smith GitHub repository name for a repository at a specific commit.
    """
    return f"{repo.replace('/', '__')}.{commit[:8]}"


def clone_repo(repo: str, dest: str | None = None, org: str = ORG_NAME) -> str | None:
    """Clone a repository from GitHub."""
    if not os.path.exists(dest or repo):
        print("INSIDE IF")
        print("https://github.com/", org, '/', repo, '.git')
        clone_cmd = (
            f"git clone https://github.com/{org}/{repo}.git"
            if dest is None
            else f"git clone https://github.com/{org}/{repo}.git {dest}"
        )
        subprocess.run(
            clone_cmd,
            check=True,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return repo if dest is None else dest
    return None


def generate_hash(s):
    return "".join(
        random.Random(int(hashlib.sha256(s.encode()).hexdigest(), 16)).choices(
            string.ascii_lowercase + string.digits, k=8
        )
    )


def get_test_paths(dir_path: str, ext: str = ".py") -> list[Path]:
    """
    Get all testing file paths relative to the given directory.
    """
    return [
        Path(os.path.relpath(os.path.join(root, file), dir_path))
        for root, _, files in os.walk(Path(dir_path).resolve())
        for file in files
        if (
            (
                any([x in root.split("/") for x in ["tests", "test", "specs"]])
                or file.lower().startswith("test")
                or file.rsplit(".", 1)[0].endswith("test")
            )
            and (ext is None or file.endswith(ext))
        )
    ]


def does_repo_exist(repo: str) -> bool:
    """
    Check if a repository exists in project organization.
    """
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    api = GhApi(token=GITHUB_TOKEN)
    org_repos = [
        x["name"]
        for page in range(1, 3)
        for x in api.repos.list_for_org(ORG_NAME, per_page=100, page=page)  # type: ignore
    ]
    return repo in org_repos
