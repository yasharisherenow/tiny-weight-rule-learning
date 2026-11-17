"""Train and score one LoRA pair on complementary parcel rules."""

import argparse
import gc
import hashlib
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DEPOTS = ("Aster", "Birch", "Cedar", "Dover")
FIELDS = ("printed", "scanned")
CHOICES = ("A", "B")
TEMPLATES = (
    "Depot: {depot}\nPrinted lane: {printed}\nScanned lane: {scanned}\nWhere should the parcel go?",
    "Parcel at depot {depot}\nLane on printed label: {printed}\nLane from scanner: {scanned}\nChoose the destination.",
    "Dispatch record for {depot}\nThe label lists lane {printed}. The scan lists lane {scanned}.\nSelect the destination code.",
)
CHOICE_SUFFIX = "\nA: {option_a}\nB: {option_b}\nAnswer:"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Parcel:
    id: str
    depot: str
    printed: int
    scanned: int
    option_a: int
    option_b: int
    template: int

    def answer(self, field):
        return "A" if getattr(self, field) == self.option_a else "B"

    def prompt(self, field=None, template=None):
        values = asdict(self)
        index = self.template if template is None else template
        prompt = (TEMPLATES[index] + CHOICE_SUFFIX).format(**values)
        if field is not None:
            return f"Routing rule: At {self.depot}, use the {field} lane.\n\n{prompt}"
        return prompt


def build_data(seed, *, test_lanes: range = range(500, 700)):
    if len(test_lanes) < 9:
        raise ValueError("The test split needs at least nine distinct lane codes.")
    rules_rng = random.Random(seed)
    fields = ["printed", "scanned"] * 2
    rules_rng.shuffle(fields)
    rule_a = dict(zip(DEPOTS, fields, strict=True))
    rule_b = {depot: FIELDS[1 - FIELDS.index(field)] for depot, field in rule_a.items()}
    data = {}
    split_specs = (
        ("train", range(100, 300), 72, 24),
        ("dev", range(300, 500), 16, 8),
        ("test", test_lanes, 32, 16),
    )
    for split, lane_pool, n_disagree, n_agree in split_specs:
        data[split] = build_split(seed, split, lane_pool, DEPOTS, n_disagree, n_agree)
    return rule_a, rule_b, data


def build_split(seed, split, lane_pool, depots, disagreement_count, agreement_count):
    rng = random.Random(f"{seed}:{split}")
    rows = []
    for depot in depots:
        for disagreement, number in ((True, disagreement_count), (False, agreement_count)):
            used = set()
            for i in range(number):
                codes = tuple(rng.sample(lane_pool, 2))
                while tuple(sorted(codes)) in used:
                    codes = tuple(rng.sample(lane_pool, 2))
                used.add(tuple(sorted(codes)))
                printed, alternative = codes
                rows.append(Parcel(
                    f"{split}-{len(rows) + 1:04}",
                    depot,
                    printed,
                    alternative if disagreement else printed,
                    codes[i % 2],
                    codes[1 - i % 2],
                    2 if split == "test" else (i // 2) % 2,
                ))
    rng.shuffle(rows)
    return rows


def check_data(data, rule_a, rule_b):
    if not rule_a or rule_a.keys() != rule_b.keys():
        raise ValueError("The two rulebooks must cover the same nonempty set of depots.")
    lane_pools = [
        {lane for row in rows for lane in (row.option_a, row.option_b)}
        for rows in data.values()
    ]
    for first, second in itertools.combinations(lane_pools, 2):
        if first & second:
            raise ValueError("Training, validation, and test lanes must be disjoint.")
    if {r.template for r in data["test"]} & {r.template for r in data["train"]}:
        raise ValueError("The main test must use a held-out prompt template.")
    for split, rows in data.items():
        if {r.depot for r in rows} != rule_a.keys():
            raise ValueError(f"Unexpected depot coverage in {split}.")
        for depot in rule_a:
            for disagreement in (False, True):
                group = [r for r in rows if r.depot == depot and (r.printed != r.scanned) == disagreement]
                for rule in (rule_a, rule_b):
                    n_first = sum(r.answer(rule[depot]) == "A" for r in group)
                    if not group or 2 * n_first != len(group):
                        raise ValueError(f"Unbalanced answer labels in {split}, {depot}.")
                if any((r.answer(rule_a[depot]) != r.answer(rule_b[depot])) != disagreement for r in group):
                    raise ValueError(f"Complementary rulebooks disagree with the design in {split}.")


def gold(rows, rulebook):
    return [r.answer(rulebook[r.depot]) for r in rows]


def pair_metrics(rows, expected_a, expected_b, answers_a, answers_b):
    result = {}
    for stratum, disagreement in (("disagreement", True), ("agreement", False)):
        indices = [i for i, r in enumerate(rows) if (r.printed != r.scanned) == disagreement]
        result[stratum] = {
            "n": len(indices),
            "a_correct": sum(answers_a[i] == expected_a[i] for i in indices),
            "b_correct": sum(answers_b[i] == expected_b[i] for i in indices),
            "both_correct": sum(answers_a[i] == expected_a[i] and answers_b[i] == expected_b[i]
                                for i in indices),
        }
    return result


def shortcut_metrics(rows, expected_a, expected_b):
    policies = {c: [c] * len(rows) for c in CHOICES}
    policies.update({field: [r.answer(field) for r in rows] for field in FIELDS})
    return {
        f"{a}/{b}": pair_metrics(rows, expected_a, expected_b, answers_a, answers_b)
        for (a, answers_a), (b, answers_b) in itertools.product(policies.items(), repeat=2)
    }


def write_json(path, contents):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_arguments(parser):
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", help="Pinned checkpoint revision. The default Qwen revision is built in.")
    parser.add_argument("--checkpoint", type=Path, help="Local copy of the pinned checkpoint")


def resolve_revision(parser, args):
    if args.revision is None and args.model_id != MODEL_ID:
        parser.error("Supply a pinned --revision when choosing a new base model.")
    if args.revision is None:
        args.revision = REVISION


class Experiment:
    def __init__(self, args):
        import peft
        import torch
        import transformers
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("This run requires CUDA.")
        self.args = args
        self.torch = torch
        self.peft = peft
        self.transformers = transformers
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.model_id = getattr(args, "model_id", MODEL_ID)
        self.revision = getattr(args, "revision", REVISION)
        self.checkpoint = args.checkpoint or self.model_id
        checkpoint_options = {"revision": self.revision} if not args.checkpoint else {"local_files_only": True}
        self.load_options = dict(checkpoint_options)
        self.load_in_4bit = getattr(args, "load_in_4bit", False)
        self.model_loader = transformers.AutoModelForCausalLM
        self.target_modules = list(TARGET_MODULES)
        if self.load_in_4bit:
            import bitsandbytes

            self.bitsandbytes = bitsandbytes
            config = transformers.AutoConfig.from_pretrained(self.checkpoint, **self.load_options)
            if config.model_type == "mistral3":
                self.model_loader = transformers.Mistral3ForConditionalGeneration
                self.target_modules = rf".*\.language_model\..*\.(?:{'|'.join(TARGET_MODULES)})"
            self.load_options.update(
                device_map={"": 0},
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=self.dtype,
                ),
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, **checkpoint_options)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        answer_tokens = [self.tokenizer.encode(f" {c}", add_special_tokens=False) for c in CHOICES]
        if any(len(tokens) != 1 for tokens in answer_tokens) or answer_tokens[0] == answer_tokens[1]:
            raise ValueError("The checkpoint must represent both answer choices as distinct single tokens.")
        self.answer_token_ids = [tokens[0] for tokens in answer_tokens]

    def model(self):
        model = self.model_loader.from_pretrained(
            self.checkpoint, dtype=self.dtype, **self.load_options,
        )
        if self.load_in_4bit:
            layers = [m for m in model.modules() if isinstance(m, self.bitsandbytes.nn.Linear4bit)]
            if not layers or any(m.weight.quant_state is None or not m.weight.is_cuda for m in layers):
                raise RuntimeError("Expected materialized 4-bit linear weights on CUDA.")
            self.quantized_linear_layers = len(layers)
        else:
            model.to(self.device)
        model.config.use_cache = False
        model.config.get_text_config().use_cache = False
        return model

    def encode(self, prompts):
        encoded = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        for suffix, token_id in zip(CHOICES, self.answer_token_ids, strict=True):
            completed = self.tokenizer([p + " " + suffix for p in prompts],
                                       add_special_tokens=False)["input_ids"]
            if any(full != start + [token_id] for start, full in zip(encoded, completed, strict=True)):
                raise ValueError("The answer token must remain separate from its prompt.")
        return encoded

    def logits(self, model, tokens):
        batch = self.tokenizer.pad({"input_ids": tokens}, padding=True, return_tensors="pt").to(self.device)
        positions = (batch["attention_mask"].cumsum(dim=1) - 1).clamp_min(0)
        return model(**batch, position_ids=positions, logits_to_keep=1).logits[:, -1, :].float()

    def score(self, model, encoded):
        model.eval()
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(encoded), self.args.batch_size):
                logits = self.logits(model, encoded[start:start + self.args.batch_size])
                scores.extend(logits[:, self.answer_token_ids].cpu().tolist())
        if any(not math.isfinite(value) for pair in scores for value in pair):
            raise RuntimeError("Scoring produced a non-finite answer logit.")
        return scores

    def train(self, label, train_x, train_y, dev_x, dev_y, *, model=None, record_first_exposure=False):
        torch = self.torch
        if model is None:
            base = self.model()
            if self.load_in_4bit:
                base = self.peft.prepare_model_for_kbit_training(
                    base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            torch.manual_seed(self.args.training_seed)
            model = self.peft.get_peft_model(base, self.peft.LoraConfig(
                task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
                target_modules=self.target_modules,
            ))
        else:
            torch.manual_seed(self.args.training_seed)
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any(".lora_" not in name for name, _ in trainable):
            raise RuntimeError("Expected only LoRA adapter parameters to be trainable.")
        start_hash = self.adapter_hash(trainable)
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=self.args.learning_rate,
                                      weight_decay=0.0)
        dev_history = []
        first_exposure = []
        rng = random.Random(self.args.training_seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for epoch in range(self.args.epochs):
            model.train()
            order = list(range(len(train_x)))
            rng.shuffle(order)
            losses = []
            for offset in range(0, len(order), self.args.batch_size):
                indices = order[offset:offset + self.args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = self.logits(model, [train_x[i] for i in indices])
                targets = torch.tensor([self.answer_token_ids[CHOICES.index(train_y[i])] for i in indices],
                                       device=self.device)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training produced a non-finite loss.")
                if record_first_exposure and epoch == 0:
                    scores = torch.nn.functional.cross_entropy(logits.detach(), targets, reduction="none").cpu()
                    first_exposure.extend(zip(indices, scores.tolist(), strict=True))
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
                if self.load_in_4bit and (len(losses) == 1 or len(losses) % 12 == 0):
                    print(f"Copy {label} epoch {epoch + 1}, step {len(losses)}: "
                          f"loss {loss.item():.4f}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated",
                          flush=True)
            dev_answers = select(self.score(model, dev_x))
            record = {
                "epoch": epoch + 1,
                "optimizer_steps": len(losses),
                "mean_training_loss": torch.stack(losses).mean().item(),
                "dev_correct": sum(pred == target for pred, target in zip(dev_answers, dev_y, strict=True)),
                "dev_total": len(dev_y),
            }
            dev_history.append(record)
            print(f"Copy {label} epoch {record['epoch']}: {record['dev_correct']}/{record['dev_total']} on dev "
                  f"(loss {record['mean_training_loss']:.4f})", flush=True)
        torch.cuda.synchronize()
        info = {
            "initial_adapter_sha256": start_hash,
            "final_adapter_sha256": self.adapter_hash(trainable),
            "trainable_parameters": sum(p.numel() for _, p in trainable),
            "seconds_train_and_dev": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "dev_history": dev_history,
        }
        if record_first_exposure:
            info["first_exposure_nll_nats"] = first_exposure
        if info["final_adapter_sha256"] == start_hash and not record_first_exposure:
            raise RuntimeError("The trainable adapter did not change.")
        if self.load_in_4bit:
            info["quantized_linear_layers"] = self.quantized_linear_layers
        model.save_pretrained(self.args.output / f"adapter-{label.lower()}", safe_serialization=True)
        return model, info

    def adapter_hash(self, parameters):
        h = hashlib.sha256()
        for name, param in parameters:
            h.update(name.encode("utf-8"))
            h.update(param.detach().float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def config(self):
        return {
            "device": self.torch.cuda.get_device_name(),
            "dtype": str(self.dtype),
            "torch": self.torch.__version__,
            "torch_cuda": self.torch.version.cuda,
            "peft": self.peft.__version__,
            "transformers": self.transformers.__version__,
            "choice_token_ids": self.answer_token_ids,
            **({"bitsandbytes": self.bitsandbytes.__version__, "target_modules": self.target_modules}
               if self.load_in_4bit else {}),
        }


def select(scores):
    return ["A" if a >= b else "B" for a, b in scores]


def run(args, rule_a, rule_b, data):
    exp = Experiment(args)
    dev = exp.encode([r.prompt() for r in data["dev"]])
    train = exp.encode([r.prompt() for r in data["train"]])
    test_rows = data["test"]
    test_hidden = exp.encode([r.prompt() for r in test_rows])
    test_seen = exp.encode([r.prompt(template=0) for r in test_rows])
    test_visible = [exp.encode([r.prompt(rule[r.depot]) for r in test_rows]) for rule in (rule_a, rule_b)]
    expected = [gold(test_rows, rule) for rule in (rule_a, rule_b)]
    report = {"runtime": exp.config(), "shortcut_baselines": shortcut_metrics(test_rows, *expected)}
    print(f"Scoring the frozen {exp.model_id} checkpoint on {len(test_rows)} test parcels.", flush=True)
    base = exp.model()
    frozen_hidden = exp.score(base, test_hidden)
    scores = {
        "frozen_hidden": [frozen_hidden, frozen_hidden],
        "frozen_visible": [exp.score(base, x) for x in test_visible],
    }
    for name, pair in scores.items():
        report[name] = pair_metrics(test_rows, *expected, *[select(x) for x in pair])
    write_json(args.output / "baseline.json", report)
    print(json.dumps({k: report[k] for k in scores}), flush=True)
    del base
    gc.collect()
    exp.torch.cuda.empty_cache()

    report["training"] = {}
    scores["trained_hidden_novel_wording"] = []
    scores["trained_hidden_seen_wording"] = []
    method = "4-bit NF4 LoRA" if exp.load_in_4bit else "conventional LoRA"
    for label, rule in zip(CHOICES, (rule_a, rule_b), strict=True):
        print(f"Training copy {label} with {method}.", flush=True)
        model, info = exp.train(label, train, gold(data["train"], rule), dev, gold(data["dev"], rule))
        scores["trained_hidden_novel_wording"].append(exp.score(model, test_hidden))
        scores["trained_hidden_seen_wording"].append(exp.score(model, test_seen))
        report["training"][label] = info
        write_json(args.output / f"metrics-{label.lower()}.json", info)
        del model
        gc.collect()
        exp.torch.cuda.empty_cache()
    if report["training"]["A"]["initial_adapter_sha256"] != report["training"]["B"]["initial_adapter_sha256"]:
        raise RuntimeError("The matched runs did not share an adapter initialization.")

    for name, pair in scores.items():
        report[name] = pair_metrics(test_rows, *expected, *[select(x) for x in pair])
    predictions = []
    for i, row in enumerate(test_rows):
        record = {"id": row.id, "expected_a": expected[0][i], "expected_b": expected[1][i]}
        record["conditions"] = {
            name: [{"scores_a_b": side[i], "prediction": select([side[i]])[0]} for side in sides]
            for name, sides in scores.items()
        }
        predictions.append(record)
    write_json(args.output / "predictions.json", predictions)
    write_json(args.output / "results.json", report)
    print(json.dumps({k: report[k] for k in scores}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    checkpoint_arguments(parser)
    parser.add_argument("--data-protocol", type=Path, help="Reuse the parcel split from a wording study.")
    parser.add_argument("--rule-seed", type=int, default=81715)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--load-in-4bit", action="store_true", help="Freeze NF4 base weights and train with checkpointing.")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    resolve_revision(parser, args)
    if args.epochs <= 0 or args.batch_size <= 0 or not 0 < args.learning_rate < 1:
        parser.error("Training requires positive epochs, batch size, and a learning rate below one.")
    source_sha = None
    if args.data_protocol is None:
        rule_a, rule_b, data = build_data(args.rule_seed)
    else:
        source = args.data_protocol.read_bytes()
        shared = json.loads(source)
        draw = shared["draws"].get(str(args.rule_seed))
        if draw is None:
            parser.error("The shared data protocol does not contain the requested rulebook seed.")
        if shared["choice_suffix"] != CHOICE_SUFFIX or tuple(shared["training_templates"][:2]) != TEMPLATES[:2]:
            raise ValueError("The shared data protocol must use the same training prompts and answer choices.")
        rule_a, rule_b = draw["rulebooks"]
        data = {name: [Parcel(**row) for row in rows] for name, rows in draw["examples"].items()}
        source_sha = hashlib.sha256(source).hexdigest()
    check_data(data, rule_a, rule_b)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {
        "model": args.model_id,
        "revision": args.revision,
        "data_protocol_sha256": source_sha,
        "rule_seed": args.rule_seed,
        "training_seed": args.training_seed,
        "rulebooks": {"a": rule_a, "b": rule_b},
        "templates": TEMPLATES,
        "choice_suffix": CHOICE_SUFFIX,
        "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "lora_rank": 8, "lora_alpha": 16,
                     "target_modules": TARGET_MODULES,
                     **({"base_quantization": "NF4 with double quantization", "gradient_checkpointing": True}
                        if args.load_in_4bit else {})},
        "score": "Choose the more probable of the two legal answer tokens.",
        "primary": "Paired correctness on disagreement items with unseen lane values, unseen wording, and no rulebook.",
        "examples": {k: [asdict(r) for r in rows] for k, rows in data.items()},
    })
    print(f"Validated splits: {', '.join(f'{k} {len(v)}' for k, v in data.items())}.", flush=True)
    if not args.prepare_only:
        run(args, rule_a, rule_b, data)


if __name__ == "__main__":
    main()
