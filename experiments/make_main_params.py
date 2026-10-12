"""Emit the parameter files for the main experiment.

Usage:
    .venv/bin/python experiments/make_main_params.py          # writes both files
    .venv/bin/python experiments/make_main_params.py --count  # just the job counts
"""

import argparse
import collections
import json
import pathlib

from amortised_adaptive_is.core.cost import cost
from experiments.gapis_lambda.table import TUNED_PATH


def solve_k(method: str, budget: int, n: int, t: int, d: int) -> int:
    """Largest integer `K` whose cost fits the budget, or 0 if `K = 1` does not fit."""
    k = 0
    while cost(method, n, k + 1, t, d) <= budget:
        k += 1
    return k
from amortised_adaptive_is.core.specs import TASKS, get_task

FAMILIES = ["blr", "hierarchical", "funnel", "bananahard"]
# (spatial_dim, n_components).
DIMS = [(5, 10), (25, 25), (50, 50)]
SEEDS = [100, 101, 102]

# The GAPIS step size
LAM_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

# The single-target variant of each family
FIX_TAG = dict.fromkeys(("blr", "hierarchical", "bananahard", "funnel"), "fix")

# Shift A: the SAME family with a parameter pushed outside the range it trained on.
def _blr_n(mult):
    """`n_range` scaled by `mult`, from `[3p, 12p]`."""
    return lambda d: f"task.target.n_range=[{3 * (d - 1) * mult},{12 * (d - 1) * mult}]"


SHIFT_A = {
    "funnel": [("x1.5", "task.target.sigma_v_range=[3.0,4.5] task.centre_scale=4.5", "")],
    "hierarchical": [("x2.5", "task.target.tau_true_range=[8.0,20.0] task.centre_scale=20.0", "")],
    "bananahard": [("x2", "task.target.b_range=[0.5,1.0]", ""),
                   ("x4", "task.target.b_range=[1.0,2.0]", "")],
    "blr": [("n4", _blr_n(4), ""),
            ("n16", _blr_n(16), ""),
            ("beta4", "task.target.beta_scale=8.0", ""),
            ("beta10", "task.target.beta_scale=20.0", "")],
}

SHIFT_A_FAR = {
    "funnel": [("x2", "task.target.sigma_v_range=[4.5,6.0] task.centre_scale=6.0", ""),
               ("x3", "task.target.sigma_v_range=[6.0,9.0] task.centre_scale=9.0", "")],
    "hierarchical": [("x4", "task.target.tau_true_range=[20.0,32.0] task.centre_scale=32.0", ""),
                ("x6", "task.target.tau_true_range=[32.0,48.0] task.centre_scale=48.0", "")],
    "bananahard": [("x8", "task.target.b_range=[2.0,4.0]", ""),
                   ("x16", "task.target.b_range=[4.0,8.0]", "")],
    "blr": [("n64", _blr_n(64), ""),
            ("n256", _blr_n(256), ""),
            ("beta25", "task.target.beta_scale=50.0", ""),
            ("beta60", "task.target.beta_scale=120.0", "")],
}

SHIFT_A_ALL = {f: SHIFT_A[f] + SHIFT_A_FAR.get(f, []) for f in SHIFT_A}


COST = {
    "dm_pmc_resampling": "df",
    "m_pmc": "df",
    "gapis": "gapis",
}
# The learnt sampler's K at inference.
LEARNED_K = 15
NOGRAD_K = 2 * (LEARNED_K + 1)

# At higher dimensions, the cost budget is based on GAPIS instead
BUDGET_BY_GAPIS_K = {50: 5}

HIDDEN_DIM = 32
HIDDEN_DIM_BY_D: dict[int, int] = {25: 64, 50: 128}

BASE = (
    "train.cov=diag train.n_epochs=1500 train.lr=0.003 "
    "train.train_batch_size=32 train.val_every=25 train.val_select=loss train.print_every=500 "
    "train.use_log_pi=false train.use_global_w=false train.use_grad_input=true "
    "train.lr_schedule=cosine train.loss=rkl train.anneal_beta0=0.3 train.anneal_frac=0.1"
)
# blr needs a larger mean step
PER_FAMILY_TRAIN = {"blr": "train.max_delta_mu=3.0"}

# The swept GAPIS step size
LAM = json.loads(pathlib.Path(TUNED_PATH).read_text())
LAM_SOURCE = str(TUNED_PATH)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "run_params"



def base_for(d: int) -> str:
    """`BASE` plus the network width for this dimension."""
    return f"{BASE} train.hidden_dim={HIDDEN_DIM_BY_D.get(d, HIDDEN_DIM)}"


def budget_for(n: int, t: int, d: int, learnt_k: int = LEARNED_K) -> int:
    """Target-evaluation budget for one cell, in density-evaluation units."""
    if d in BUDGET_BY_GAPIS_K:
        return cost("gapis", n, BUDGET_BY_GAPIS_K[d], t, d)
    return cost("learnt", n, learnt_k, t, d)


def learnt_k_for(n: int, t: int, d: int, learnt_k: int = LEARNED_K) -> int:
    """The learnt sampler's K at that budget -- `learnt_k` unless the budget is set elsewhere."""
    if d not in BUDGET_BY_GAPIS_K:
        return learnt_k
    return solve_k("learnt", budget_for(n, t, d, learnt_k), n, t, d)


def cells():
    """The (family, d, N) cells of the main experiment.
    """
    for d, n in DIMS:
        for fam in FAMILIES:
            if fam == "hierarchical" and d == 2:
                continue
            if f"{fam}_d{d}" in TASKS:
                yield fam, d, n


# Gaussian-mixture targets
MIX_FAMILIES = ["corners", "gmm5"]
MIX_DIMS = [(2, 5), (5, 10), (25, 25)]


def mix_cells():
    """The `(family, d, N)` cells of the mixtures."""
    for d, n in MIX_DIMS:
        for fam in MIX_FAMILIES:
            if f"{fam}_d{d}" in TASKS:
                yield fam, d, n


def all_cells():
    """Main cells then mixture cells.
    """
    yield from cells()
    yield from mix_cells()


def learnt_lines(learnt_k: int = LEARNED_K) -> list[str]:
    out = []
    for fam, d, n in all_cells():
        task = f"{fam}_d{d}"
        spec = get_task(task, n_components=n)
        k_cell = learnt_k_for(n, spec.n_steps, spec.spatial_dim, learnt_k)
        for s in SEEDS:
            extra = PER_FAMILY_TRAIN.get(fam, "")
            kp = "" if k_cell == LEARNED_K else f"task.k_particles={k_cell} "
            out.append(
                f"task={task} task.n_components={n} {kp}train.seed={s} "
                f"tag=M-{task}-n{n}-k{k_cell}-s{s} "
                f"train.delta_scale=sigma {extra} {base_for(d)} "
                f"wandb.group={task}_n{n} "
                f"wandb.tags=[exp:M,{fam},d{d},n{n},k{k_cell},s{s},rkl,sigma]".replace("  ", " ")
            )
    return out


def baseline_lines(learnt_k: int = LEARNED_K) -> list[str]:
    out = []
    for fam, d, n in all_cells():
        task = f"{fam}_d{d}"
        spec = get_task(task, n_components=n)
        t, dim = spec.n_steps, spec.spatial_dim
        budget = budget_for(n, t, dim, learnt_k)
        ks = {m: solve_k(cost_func, budget, n, t, dim) for m, cost_func in COST.items()}

        ks["true_sampler"] = max(1, solve_k("df", budget, n, t, dim))
        for m, k in ks.items():
            if k < 1:
                print(f"  !! {task}/{m}: infeasible at {budget} units ({m} cannot run), skipped")
                continue

            if m == "gapis" and task not in LAM:
                # No swept step size for this cell.
                print(f"  !! {task}/gapis: no swept lam, row skipped "
                      f"(sweep this cell first)")
                continue
            head = (f"sampler=gapis sampler.lam={LAM[task]['lam']:g}" if m == "gapis"
                    else f"sampler={m}")
            out.append(f"{head} task={task} task.n_components={n} task.k_particles={int(k)} "
                       f"wandb.group={task} wandb.tags=[exp:M,{m},{fam},d{d},n{n},k{learnt_k}]")
    return out


def sweep_lines() -> list[str]:
    """One job per (task, lam): the whole grid on every cell.
    """
    out = []
    for fam, d, n in all_cells():
        task = f"{fam}_d{d}"
        spec = get_task(task, n_components=n)
        t, dim = spec.n_steps, spec.spatial_dim
        k = solve_k(COST["gapis"], budget_for(n, t, dim), n, t, dim)
        if k < 1:
            print(f"  !! {task}/gapis: infeasible, skipped")
            continue
        for lam in LAM_GRID:
            tag = f"{lam:g}"
            out.append(
                f"sampler=gapis sampler.lam={lam:g} task={task} task.n_components={n} "
                f"task.k_particles={int(k)} "
                f"wandb.name=gapis-{task}-n{n}-lam{tag} wandb.group={task}_gapis_sweep "
                f"wandb.tags=[exp:GS,gapis,{fam},d{d},n{n},lam{tag}] "
                f"out=results/runs/gapis-{task}-lam{tag}.json")
    return out


def single_target_train_lines() -> list[str]:
    """Train one sampler per (single target, seed).
    """
    out = []
    for fam, d, n in cells():
        if f"{fam}{FIX_TAG[fam]}_d{d}" not in TASKS:
            continue
        fix = f"{fam}{FIX_TAG[fam]}_d{d}"
        for s in SEEDS:
            extra = PER_FAMILY_TRAIN.get(fam, "")
            out.append(
                f"task={fix} task.n_components={n} train.seed={s} "
                f"tag=S-{fix}-n{n}-k{LEARNED_K}-s{s} "
                f"train.delta_scale=sigma {extra} {base_for(d)} "
                f"wandb.group={fix}_single "
                f"wandb.tags=[exp:ST,{fam},d{d},n{n},s{s},single]".replace("  ", " "))
    return out


def single_target_eval_lines() -> list[str]:
    """Every (trained sampler, evaluation target) pair.
    """
    out = []
    for fam, d, n in cells():

        if f"{fam}{FIX_TAG[fam]}_d{d}" not in TASKS:
            continue
        fix, fam_task = f"{fam}{FIX_TAG[fam]}_d{d}", f"{fam}_d{d}"
        for s in SEEDS:
            ckpts = {"single": f"results/S-{fix}-n{n}-k{LEARNED_K}-s{s}.pt",
                     "multi": (f"results/M-{fam_task}-n{n}-"
                               f"k{learnt_k_for(n, get_task(fam_task, n_components=n).n_steps, d)}"
                               f"-s{s}.pt")}
            for src, ck in ckpts.items():
                for lab, task in (("fix", fix), ("family", fam_task)):
                    name = f"{src}-on-{lab}-{fam}_d{d}-s{s}"
                    out.append(
                        f"sampler=learnt sampler.checkpoint={ck} "
                        f"task={task} task.n_components={n} "
                        f"wandb.name={name} wandb.group={fam}_d{d}_single_vs_multi "
                        f"wandb.tags=[exp:SE,{src},on{lab},{fam},d{d},n{n},s{s}] "
                        f"out=results/runs/SE-{name}.json")
    return out


ABLATION_REF_TAG = "exp:M" # where tables read their comparison rows from
# The architecture ablation runs at two dimensions
ARCH_DIMS = (5, 25)
# The dimensions the shift experiments run at.
SHIFT_DIMS = (5, 25)

GRAD_ABLATION = {
    "nofeat": f"train.use_grad_input=false task.k_particles={NOGRAD_K}",
}

EM_ABLATION = {
    "noem": "train.use_moments=false",
}

OBJ_ABLATION = {
    "nostep": "train.step_weighting=final   train.detach_between_steps=false",
    "nopath": "train.step_weighting=uniform train.detach_between_steps=true",
}

LOSS_ABLATION = {
    "fkl":   "train.loss=fkl",
    "xent":  "train.loss=xent",
    "chisq": "train.loss=chisq",
}

ARCH_ABLATION = {
    "noequiv": "train.scale_free=false",
}

COV_ABLATION = {
    "iso":  "train.cov=iso",
    "full": "train.cov=full",
}

CAPACITY_ABLATION = {
    "h16":  "train.hidden_dim=16",
    "h32":  "train.hidden_dim=32",
    "h128": "train.hidden_dim=128",
}


def _override(base: str, over: str) -> str:
    """`base` with overrides from `over`.
    """
    keys = {t.split("=", 1)[0] for t in over.split() if "=" in t}
    kept = [t for t in base.split() if t.split("=", 1)[0] not in keys]
    return " ".join(kept + over.split())


def _ablation_lines(arms: dict[str, str], exp: str,
                    dims: tuple[int, ...] = (25,)) -> list[str]:
    """One training run per (arm, family, seed, d).
    """
    out = []
    for fam, d, n in cells():
        if d not in dims:
            continue
        task = f"{fam}_d{d}"
        for arm, over in arms.items():
            for s_ in SEEDS:
                extra = PER_FAMILY_TRAIN.get(fam, "")
                out.append(
                    f"task={task} task.n_components={n} train.seed={s_} "
                    f"tag={exp}-{arm}-{task}-n{n}-s{s_} "
                    f"train.delta_scale=sigma {extra} {_override(base_for(d), over)} "
                    f"wandb.group={task}_ablation "
                    f"wandb.tags=[exp:{exp},arm:{arm},{fam},d{d},n{n},s{s_}]".replace("  ", " "))
    return out


def grad_ablation_lines() -> list[str]:
    return _ablation_lines(GRAD_ABLATION, "AG")


EM_DIMS = (5, 25)

def em_ablation_lines() -> list[str]:
    return _ablation_lines(EM_ABLATION, "AE", EM_DIMS)


def obj_ablation_lines() -> list[str]:
    return _ablation_lines(OBJ_ABLATION, "AO")


def loss_ablation_lines() -> list[str]:
    return _ablation_lines(LOSS_ABLATION, "AL")


def arch_ablation_lines() -> list[str]:
    return _ablation_lines(ARCH_ABLATION, "AS", ARCH_DIMS)


def cov_ablation_lines() -> list[str]:
    return _ablation_lines(COV_ABLATION, "AC")


def capacity_ablation_lines() -> list[str]:
    return _ablation_lines(CAPACITY_ABLATION, "AH")


def ablation_controls() -> list[str]:
    """The `exp:M` runs that are used as the comparison row for ablation tables."""
    return [f"M-{fam}_d{d}-n{n}-k{LEARNED_K}-s{s}"
            for fam, d, n in cells() if d == 25 for s in SEEDS]


def shift_a_lines() -> list[str]:
    """Generalisation on modified families.
    """
    out = []
    for fam, d, n in cells():
        if d not in SHIFT_DIMS:
            continue
        task = f"{fam}_d{d}"
        k_matched = {}
        spec = get_task(task, n_components=n)
        t, dim = spec.n_steps, spec.spatial_dim
        b = budget_for(n, t, dim)
        for m, cost_func in COST.items():
            k_matched[m] = solve_k(cost_func, b, n, t, dim)

        k_matched["true_sampler"] = max(1, solve_k("df", b, n, t, dim))
        for lab, override, dt in SHIFT_A_ALL[fam]:
            ov = override(d) if callable(override) else override
            tag = f"{fam}_d{d}-{lab}"
            common = f"task={task} task.n_components={n} {ov} {dt}".strip()

            for s_ in SEEDS:
                ck = (f"results/M-{task}-n{n}-"
                      f"k{learnt_k_for(n, spec.n_steps, spec.spatial_dim)}-s{s_}.pt")
                out.append(
                    f"sampler=learnt sampler.checkpoint={ck} {common} "
                    f"wandb.name=gA-learnt-{tag}-s{s_} wandb.group={tag} "
                    f"wandb.tags=[exp:GA,learnt,{fam},d{d},{lab},s{s_}] "
                    f"out=results/runs/GA-learnt-{tag}-s{s_}.json")
                
            # the baselines on the modified family
            for m in ("dm_pmc_resampling", "m_pmc", "true_sampler"):
                out.append(
                    f"sampler={m} {common} task.k_particles={k_matched[m]} "
                    f"wandb.name=gA-{m}-{tag} wandb.group={tag} "
                    f"wandb.tags=[exp:GA,{m},{fam},d{d},{lab}] "
                    f"out=results/runs/GA-{m}-{tag}.json")
        
            for lam in LAM_GRID:
                out.append(
                    f"sampler=gapis sampler.lam={lam:g} {common} "
                    f"task.k_particles={k_matched['gapis']} "
                    f"wandb.name=gA-gapis-{tag}-lam{lam:g} wandb.group={tag} "
                    f"wandb.tags=[exp:GA,gapis,{fam},d{d},{lab},lam{lam:g}] "
                    f"out=results/runs/GA-gapis-{tag}-lam{lam:g}.json")
    return out


def _ablation_shift_a_lines(exp: str, src_exp: str, arms: tuple[str, ...],
                            dims: tuple[int, ...], name_prefix: str) -> list[str]:
    """Generalisation for checkpoints trained in an ablation run 
    -- for the equivariance generalisation experiment.
    """
    out = []
    for fam, d, n in cells():
        if d not in dims:
            continue
        task = f"{fam}_d{d}"
        for lab, override, dt in SHIFT_A[fam]:
            ov = override(d) if callable(override) else override
            tag = f"{fam}_d{d}-{lab}"
            for arm in arms:
                for s_ in SEEDS:
                    ck = f"results/{src_exp}-{arm}-{task}-n{n}-s{s_}.pt"
                    out.append(
                        f"sampler=learnt sampler.checkpoint={ck} "
                        f"task={task} task.n_components={n} {ov} {dt} "
                        f"wandb.name={name_prefix}-{arm}-{tag}-s{s_} wandb.group={tag} "
                        f"wandb.tags=[exp:{exp},arm:{arm},{fam},d{d},{lab},s{s_}] "
                        f"out=results/runs/{exp}-{arm}-{tag}-s{s_}.json".replace("  ", " "))
    return out


def arch_shift_a_lines() -> list[str]:
    """Generalisation for the non-scale-free sampler.
    """
    return _ablation_shift_a_lines("AGA", "AS", ("noequiv",), ARCH_DIMS, "gAn")


CAPACITY_SHIFT_ARMS = ("h32", "h128")


def capacity_shift_a_lines() -> list[str]:
    """Generalisation for the capacity ablation samplers, `exp:AHA`.
    """
    return _ablation_shift_a_lines("AHA", "AH", CAPACITY_SHIFT_ARMS, (25,), "gAh")


def shift_b_lines() -> list[str]:
    """Cross-family: each trained sampler evaluated on the other families.
    """
    out = []
    by_d = collections.defaultdict(list)
    for fam, d, n in cells():
        if d not in SHIFT_DIMS:
            continue
        by_d[(d, n)].append(fam)
    for (d, n), fams in sorted(by_d.items()):
        for src in fams:
            for tgt in fams:
                src_spec = get_task(f"{src}_d{d}", n_components=n)
                k_src = learnt_k_for(n, src_spec.n_steps, src_spec.spatial_dim)
                for s_ in SEEDS:
                    ck = f"results/M-{src}_d{d}-n{n}-k{k_src}-s{s_}.pt"
                    out.append(
                        f"sampler=learnt sampler.checkpoint={ck} "
                        f"task={tgt}_d{d} task.n_components={n} "
                        f"wandb.name=gB-{src}-on-{tgt}-d{d}-s{s_} "
                        f"wandb.group={tgt}_d{d}_crossfamily "
                        f"wandb.tags=[exp:GB,src:{src},tgt:{tgt},d{d},s{s_}] "
                        f"out=results/runs/GB-{src}-on-{tgt}-d{d}-s{s_}.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true", help="print array sizes only")
    ap.add_argument("--learnt-k", type=int, default=LEARNED_K, help="The learnt sampler's K, which sets the budget.")
    ap.add_argument("--suffix", default="", help="Appended to the output filenames.")
    args = ap.parse_args()
    print(f"  gapis lam source: {LAM_SOURCE}")
    lrn, bas, swp = learnt_lines(args.learnt_k), baseline_lines(args.learnt_k), sweep_lines()
    stt, ste = single_target_train_lines(), single_target_eval_lines()
    ga, gb = shift_a_lines(), shift_b_lines()
    gab, eab = grad_ablation_lines(), em_ablation_lines()
    oab, lab = obj_ablation_lines(), loss_ablation_lines()
    sab, cab = arch_ablation_lines(), cov_ablation_lines()
    hab = capacity_ablation_lines()
    aga = arch_shift_a_lines()
    aha = capacity_shift_a_lines()
    if args.count:
        print(f"  learnt    : {len(lrn)} jobs  -> run_scripts/run_main_learnt.sh")
        print(f"  baselines  : {len(bas)} jobs  -> run_scripts/run_main_baselines.sh")
        print(f"  gapis sweep: {len(swp)} jobs  -> run_scripts/run_gapis_sweep.sh")
        print(f"  single train: {len(stt)} jobs  -> run_scripts/run_single_train.sh")
        print(f"  single eval : {len(ste)} jobs  -> run_scripts/run_single_eval.sh")
        print(f"  gen shift A : {len(ga)} jobs  -> run_scripts/run_gen_shiftA.sh")
        print(f"  gen shift B : {len(gb)} jobs  -> run_scripts/run_gen_shiftB.sh")
        print(f"  grad ablation: {len(gab)} jobs  -> run_scripts/run_grad_ablation.sh")
        print(f"  em ablation  : {len(eab)} jobs  -> run_scripts/run_em_ablation.sh")
        print(f"  obj ablation : {len(oab)} jobs  -> run_scripts/run_obj_ablation.sh")
        print(f"  loss ablation: {len(lab)} jobs  -> run_scripts/run_loss_ablation.sh")
        print(f"  arch ablation: {len(sab)} jobs  -> run_scripts/run_arch_ablation.sh")
        print(f"  cov ablation : {len(cab)} jobs  -> run_scripts/run_cov_ablation.sh")
        print(f"  cap ablation : {len(hab)} jobs  -> run_scripts/run_capacity_ablation.sh")
        print(f"  arch shiftA  : {len(aga)} jobs  -> run_scripts/run_arch_shiftA.sh")
        print(f"  cap shiftA   : {len(aha)} jobs  -> run_scripts/run_capacity_shiftA.sh")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lf = OUT / f"run_params_main_learnt{args.suffix}.txt"
    bf = OUT / f"run_params_main_baselines{args.suffix}.txt"
    lf.write_text("\n".join(lrn) + "\n")
    bf.write_text("\n".join(bas) + "\n")
    print(f"  wrote {lf}    ({len(lrn)} jobs)")
    print(f"  wrote {bf}  ({len(bas)} jobs)")
    sf = OUT / f"run_params_gapis_sweep{args.suffix}.txt"
    sf.write_text("\n".join(swp) + "\n")
    print(f"  wrote {sf}     ({len(swp)} jobs)")
    for nm, lines in (("single_train", stt), ("single_eval", ste),
                      ("gen_shiftA", ga), ("gen_shiftB", gb),
                      ("grad_ablation", gab), ("em_ablation", eab),
                      ("obj_ablation", oab), ("loss_ablation", lab),
                      ("arch_ablation", sab), ("cov_ablation", cab),
                      ("capacity_ablation", hab),
                      ("arch_shiftA", aga), ("capacity_shiftA", aha),
):
        f = OUT / f"run_params_{nm}{args.suffix}.txt"
        f.write_text("\n".join(lines) + "\n")
        print(f"  wrote {f}   ({len(lines)} jobs)")


if __name__ == "__main__":
    main()
