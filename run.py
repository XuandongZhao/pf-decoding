import argparse
import time
import json
from generate import WmGenerator, OpenaiGenerator, MarylandGenerator, PFGenerator
import numpy as np
from utils import load_prompts
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# hf_cache_dir = '/home/xuandong/mnt/hf_models'
hf_cache_dir = '[YOUR_CACHE_DIR]'


def main(args):
    print("Setting: ", args)
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, use_flash_attention_2=True, cache_dir=hf_cache_dir, device_map="auto").eval()
    args.ngpus = torch.cuda.device_count()
    print(f"Using {args.ngpus}/{torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU")

    print(args.ngram, args.seed, args.seeding, args.hash_key, args.payload)

    prompts = load_prompts(json_path=args.prompt_path, nsamples=args.nsamples)
    print(f"Loaded {len(prompts)} prompts")
    print('Prompt example: ', prompts[0])

    output_name = f"{args.prompt_path.split('/')[-1].split('.')[0]}-{args.model_name.split('/')[-1]}-{str(args.nsamples)}-T{str(args.temperature)}-B{str(args.batch_size)}-P{str(args.ngram)}.jsonl"
    print('Output name: ', output_name)
    os.makedirs(args.output_dir, exist_ok=True)
    start_point = 0  # if resuming, start from the last line of the file
    if os.path.exists(os.path.join(args.output_dir, output_name)):
        with open(os.path.join(args.output_dir, output_name), "r") as f:
            for _ in f:
                start_point += 1
    print(f"Starting from {start_point}")

    oai_generator = OpenaiGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, payload=args.payload)
    pf_generator = PFGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, payload=args.payload, nowm=False)
    umd_generator = MarylandGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, payload=args.payload, gamma=args.gamma, delta=args.delta)

    print("temperature: ", args.temperature, "top_p: ", args.top_p, "max_gen_len: ", args.max_gen_len)
    print("batch_size: ", args.batch_size)

    all_times = []

    with open(os.path.join(args.output_dir, output_name), "a") as f:
        for ii in range(start_point, len(prompts), args.batch_size):
            # generate chunk
            time0 = time.time()
            chunk_size = min(args.batch_size, len(prompts) - ii)
            oai_results = oai_generator.generate(
                prompts[ii:ii + chunk_size],
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p
            )
            pf_results = pf_generator.generate(
                prompts[ii:ii + chunk_size],
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p
            )
            umd_results = umd_generator.generate(
                prompts[ii:ii + chunk_size],
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p
            )
            time1 = time.time()
            # time chunk
            speed = chunk_size / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta))
            all_times.append(time1 - time0)
            print(f"Generated {ii:5d} - {ii + chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            for prompt, oai_result, pf_result, umd_result in zip(prompts[ii:ii + chunk_size], oai_results, pf_results, umd_results):
                f.write(json.dumps({
                    "prompt": prompt,
                    "oai_result": oai_result,
                    "pf_result": pf_result,
                    "umd_result": umd_result,
                    "speed": speed,
                    "eta": eta}) + "\n")
                f.flush()

    print(f"Average time per prompt: {np.mean(all_times):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Args', add_help=False)
    # model parameters
    parser.add_argument('--model_name', type=str, default='NousResearch/Llama-2-7b-chat-hf')
    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default="data/c4.jsonl")

    # generation parameters
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_gen_len', type=int, default=256)

    # watermark parameters
    parser.add_argument('--method', type=str, default='openai',
                        help='Choose between: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.), pf (Zhao et al.)')
    parser.add_argument('--method_detect', type=str, default='openai',
                        help='Statistical test to detect watermark. Choose between: same (same as method), openai, openaiz, openainp, maryland, marylandz, pf')
    parser.add_argument('--seeding', type=str, default='hash',
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4,
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=2.0,
                        help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317,
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none',
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')

    # multibit
    parser.add_argument('--payload', type=int, default=0, help='message')
    parser.add_argument('--payload_max', type=int, default=0,
                        help='maximal message, must be inferior to the vocab size at the moment')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='number of samples to generate, if None, take all prompts')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--split', type=int, default=None,
                        help='split the prompts in nsplits chunks and chooses the split-th chunk. \
                        Allows to run in parallel. \
                        If None, treat prompts as a whole')
    parser.add_argument('--nsplits', type=int, default=None,
                        help='number of splits to do. If None, treat prompts as a whole')

    # distributed parameters
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()
    main(args)
    print("Done!")
