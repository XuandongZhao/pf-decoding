from typing import List
import json


def load_prompts(json_path: str, nsamples: int = None) -> List[str]:
    if json_path.endswith('qa.jsonl'):
        with open(json_path, "r") as f:
            prompts = [json.loads(line)['prefix'] for line in f.readlines()][:nsamples]
            prompt_template = """Your role is that of a helpful Assistant tasked with responding to a user referred to as 'Human'. Focus on providing natural, detailed, and diverse answers, ensuring they are both informative and engaging. \n\nHuman: {}\nAssistant: """
            new_prompts = [prompt_template.format(prompt) for prompt in prompts]
    elif json_path.endswith('c4.jsonl'):
        with open(json_path, "r") as f:
            prompts = [json.loads(line)['prefix'] for line in f.readlines()][:nsamples]
            prompt_template = "{}"
            new_prompts = [prompt_template.format(prompt) for prompt in prompts]
    else:
        raise NotImplementedError
    return new_prompts


if __name__ == "__main__":
    prompts = load_prompts('data/c4.jsonl', nsamples=1000)
    # prompts = load_prompts('data/qa.jsonl', nsamples=1000)
    print(prompts[0])
    print('Number of prompts:', len(prompts))
