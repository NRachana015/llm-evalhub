import json
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def get_prompt(version: str) -> dict:
    """
    Load a specific prompt version.
    """

    prompt_file = PROMPTS_DIR / f"{version}.json"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt version '{version}' was not found."
        )

    with open(prompt_file, "r", encoding="utf-8") as file:
        prompt = json.load(file)

    return prompt


def list_prompts() -> list[dict]:
    """
    Return all available prompt versions.
    """

    prompts = []

    for prompt_file in sorted(PROMPTS_DIR.glob("v*.json")):
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompts.append(json.load(file))

    return prompts