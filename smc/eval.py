"""
evaluate_twisted_truthfulqa.py
------------------------------
Benchmark
    • baseline LM   (plain p)
    • twisted-SMC   (q  = p·ψ   with your TransformerTwistModel)
on the TruthfulQA *generation* split.

Example
-------
python eval.py \
       --model_name gpt2 \
       --twist_ckpt output/final_model.pt \
       --split validation \
       --num_particles 8 \
       --max_new 100 \
       --device cuda
"""

import argparse, re, math, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from tqdm import tqdm

# ---------- import your own modules ---------------------------------
# make sure PYTHONPATH includes the folder where these live
from model          import ModelWrapper, TransformerTwistModel
from smc_sampling   import smc_proposal_sampling

from utils import setup_logging


logger = setup_logging('/home/jovyan/fida/code/smc/twisted-smc/refactored/output/eval', False)
# ---------- text-similarity helpers ---------------------------------

def _norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return t.strip()

def exact_match(pred: str, refs: list[str]) -> bool:
    n_pred = _norm(pred)
    return any(n_pred == _norm(r) for r in refs)

def rouge_l(pred: str, refs: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return max(scorer.score(pred, r)["rougeL"].fmeasure for r in refs)


# ---------- generation wrappers -------------------------------------

@torch.inference_mode()
def generate_baseline(model: AutoModelForCausalLM,
                      tok: AutoTokenizer,
                      prompt: str,
                      max_new: int,
                      temperature: float,
                      device) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids,
                         max_new_tokens=max_new,
                         temperature=temperature,
                         do_sample=True,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True)


@torch.inference_mode()
def generate_twisted(wrapper: ModelWrapper,
                     prompt: str,
                     max_new: int,
                     num_particles: int,
                     temperature: float) -> str:
    tok = wrapper.tokenizer
    device = wrapper.device
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].to(device)

    # smc_proposal_sampling expects python list of ints
    particles, log_w, *_ = smc_proposal_sampling(
        prompt_ids.tolist(),
        wrapper.get_base_model_logits_for_sequence,
        wrapper.get_twist_values_for_particles,
        num_particles=num_particles,
        new_tokens_count=max_new,
        device=device,
        logger_instance=logger,
        record_log_psi_incrementals=False
    )

    # pick best-weight particle
    best = particles[log_w.argmax().item()]
    return tok.decode(best[len(prompt_ids):], skip_special_tokens=True)


# ---------- main evaluation loop ------------------------------------

def evaluate(args):
    # ----- baseline LM ------------------------------------------------
    tok_base = AutoTokenizer.from_pretrained(args.model_name)
    lm_base  = AutoModelForCausalLM.from_pretrained(args.model_name)\
                                   .to(args.device).eval()

    # ----- twisted wrapper  ------------------------------------------
    wrapper = ModelWrapper(args.model_name, device=args.device)
    if args.twist_ckpt:
        wrapper.load_state(args.twist_ckpt)

    # ----- data ------------------------------------------------------
    ds = load_dataset("truthful_qa", "generation")[args.split]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    metrics = {
        "baseline": {"EM": 0, "ROUGE": 0},
        "twisted" : {"EM": 0, "ROUGE": 0},
    }

    for ex in tqdm(ds, desc="TruthfulQA"):
        prompt = ex["question"]
        refs   = [ex["best_answer"]] + ex["correct_answers"]

        # baseline
        out_b = generate_baseline(lm_base, tok_base, prompt,
                                  args.max_new, args.temp, args.device)
        metrics["baseline"]["EM"]    += exact_match(out_b, refs)
        metrics["baseline"]["ROUGE"] += rouge_l(out_b, refs)

        # twisted
        out_t = generate_twisted(wrapper, prompt,
                                 args.max_new, args.num_particles, args.temp)
        metrics["twisted"]["EM"]    += exact_match(out_t, refs)
        metrics["twisted"]["ROUGE"] += rouge_l(out_t, refs)

    n = len(ds)
    for k in metrics:
        metrics[k]["EM"]    /= n
        metrics[k]["ROUGE"] /= n

    print(json.dumps(metrics, indent=2))


# ---------- CLI ------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--twist_ckpt", default=None,
        help="Path to checkpoint with twist_model weights; "
             "if omitted, evaluates base LM twice.")
    p.add_argument("--split", default="validation",
        choices=["train", "validation", "test"])
    p.add_argument("--num_particles", type=int, default=32)
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    evaluate(args)
