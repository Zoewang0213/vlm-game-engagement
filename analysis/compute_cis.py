"""Bootstrap 95% confidence intervals for Tables 4/5 and the 18-window
subsampling analysis (Appendix E of the paper).

Methodology:
  - Per-game cells: bootstrap the per-window 0/1 correctness vector
    (S1/S2: real per-window predictions in ../results; other columns:
    parametric Binomial(n, p_printed), identical when p = k/n).
    10,000 resamples, percentile 95% CI, seed 42.
  - Average rows: game-level bootstrap (resample the 9 per-game accuracies).
  - Pointwise n = 59 per game; pairwise n per game (near-stable pairs
    excluded at |dE| > 0.05): Borderlands 3: 66, CS:GO Office: 75,
    Blitz Brigade: 66, Corridor 7: 37, Battlefield 42: 61,
    Apex Legends: 68, CSGO19: 46, CSGO18: 69, CS 1.6: 78.
  - Subsampling: draw 10,000 random 18-window subsets of each game's
    59 windows; report the mean and central 95% range of S1/S2 accuracy.

Outputs ci_results.json next to this script.
"""
import csv, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'results')
OUT  = os.path.join(HERE, 'ci_results.json')
B = 10000

GAMES = ['borderlands3','csgooffice','blitzbrigade','corridor7','battlefield','apex','csgo19','csgo18','cs16']
PAIR_N = dict(zip(GAMES,[66,75,66,37,61,68,46,69,78]))

# Printed accuracies from Table 4 (pointwise) and Table 5 (pairwise), paper row order.
T4 = {
 ('IV','S1'):[79.7,61.0,66.1,49.2,64.4,45.8,71.2,44.1,30.5],
 ('IV','S2'):[81.4,42.4,64.4,37.3,52.5,54.2,59.3,25.4,59.3],
 ('IV','S3'):[83.4,58.5,68.7,83.5,46.8,64.1,76.8,81.7,78.2],
 ('IV','S4'):[79.7,55.9,69.5,64.4,39.0,49.2,50.8,49.2,54.2],
 ('IV','S5'):[53.5,71.8,62.7,72.9,64.8,54.6,82.0,76.8,75.4],
 ('IV','S6'):[81.4,33.9,64.4,39.0,39.0,47.5,64.4,23.7,69.5],
 ('QW','S1'):[83.1,57.6,62.7,66.1,62.7,49.2,66.1,30.5,32.2],
 ('QW','S2'):[81.4,44.1,64.4,39.0,57.6,55.9,54.2,25.4,55.9],
 ('QW','S3'):[75.4,71.1,78.9,72.5,80.3,68.3,84.2,66.5,78.2],
 ('QW','S4'):[49.2,86.4,66.1,55.9,61.0,69.5,55.9,86.4,79.7],
 ('QW','S5'):[84.2,70.8,78.2,73.9,68.3,62.7,83.8,71.1,73.6],
 ('QW','S6'):[54.2,78.0,66.1,52.5,72.9,62.7,71.2,89.8,79.7],
 ('GPT','S1'):[83.1,52.5,62.7,57.6,64.4,52.5,55.9,22.8,57.6],
 ('GPT','S2'):[81.4,50.8,64.4,57.6,45.8,52.5,74.6,24.6,57.6],
}
T5 = {
 ('IV','S1'):[54.6,42.7,56.1,51.4,57.4,67.7,43.5,52.2,52.6],
 ('IV','S2'):[51.5,49.3,47.0,40.5,55.7,57.4,52.2,50.7,43.6],
 ('IV','S3'):[53.0,42.7,65.1,51.4,55.7,55.9,50.0,46.4,48.7],
 ('IV','S4'):[50.0,52.0,57.6,67.6,49.2,61.8,41.3,53.6,44.9],
 ('IV','S5'):[50.0,23.1,71.4,70.0,47.4,71.4,70.0,33.3,66.7],
 ('IV','S6'):[25.0,30.8,64.3,70.0,47.4,57.1,70.0,16.7,66.7],
 ('QW','S1'):[57.6,37.3,68.2,73.0,62.3,58.8,45.7,40.6,48.7],
 ('QW','S2'):[65.2,45.3,62.1,51.4,64.0,60.3,45.7,39.1,52.6],
 ('QW','S3'):[53.0,46.7,51.5,35.1,54.1,57.4,43.5,47.8,39.7],
 ('QW','S4'):[62.1,42.7,51.5,64.9,67.2,57.4,41.3,39.1,42.3],
 ('QW','S5'):[87.5,69.2,57.1,50.0,52.6,57.1,50.0,50.0,55.6],
 ('QW','S6'):[87.5,76.9,42.9,50.0,52.6,50.0,40.0,50.0,44.4],
 ('GPT','S1'):[54.6,41.3,65.2,59.5,54.1,57.4,43.5,49.3,42.3],
 ('GPT','S2'):[56.1,45.3,60.6,56.8,57.4,66.2,52.2,52.2,44.9],
}
T4_NO_PERGAME = {('IV','S3'),('IV','S5'),('QW','S3'),('QW','S5')}
T5_NO_PERGAME = {('IV','S5'),('IV','S6'),('QW','S5'),('QW','S6')}

MODEL_DIR = {'IV':'internvl3','QW':'qwen3vl'}
METHOD_DIR = {'S1':'zeroshot','S2':'theory_zeroshot'}

def load_vec(model, method, game):
    fp = os.path.join(BASE, MODEL_DIR[model], METHOD_DIR[method], f'{game}_predictions.csv')
    with open(fp) as fh:
        rows = list(csv.DictReader(fh))
    return [1 if r['pred_01'] == r['gt_01'] else 0 for r in rows]

def pctl(sorted_v, q):
    idx = q * (len(sorted_v) - 1); lo = int(idx); f = idx - lo
    return sorted_v[lo] if lo + 1 >= len(sorted_v) else sorted_v[lo]*(1-f) + sorted_v[lo+1]*f

def ci(v):
    s = sorted(v); return pctl(s, 0.025), pctl(s, 0.975)

def boot_vec(vec, rng):
    n = len(vec)
    return ci([sum(rng.choice(vec) for _ in range(n))/n*100 for _ in range(B)])

def boot_binom(p, n, rng):
    return ci([sum(1 for _ in range(n) if rng.random() < p)/n*100 for _ in range(B)])

def boot_games(accs, rng):
    n = len(accs)
    return ci([sum(rng.choice(accs) for _ in range(n))/n for _ in range(B)])

def subsample(vec, k, rng):
    n = len(vec)
    vals = []
    for _ in range(B):
        idx = rng.sample(range(n), k)
        vals.append(sum(vec[i] for i in idx)/k*100)
    lo, hi = ci(vals)
    return sum(vals)/len(vals), lo, hi

res = {'t4_pergame':{}, 't4_avg':{}, 't5_pergame':{}, 't5_avg':{}, 'subsample':{}}
rng = random.Random(42)

for (m, s), accs in T4.items():
    key = f'{m}_{s}'
    res['t4_avg'][key] = [round(sum(accs)/9,1), *[round(x,1) for x in boot_games(accs, rng)]]
    if (m, s) in T4_NO_PERGAME:
        res['t4_pergame'][key] = None
        continue
    cells = []
    for gi, g in enumerate(GAMES):
        if s in ('S1','S2') and m in ('IV','QW'):
            vec = load_vec(m, s, g)
            assert abs(sum(vec)/len(vec)*100 - accs[gi]) < 0.06, (m,s,g)
            lo, hi = boot_vec(vec, rng)
        else:
            lo, hi = boot_binom(accs[gi]/100, 59, rng)
        cells.append([accs[gi], round(lo,1), round(hi,1)])
    res['t4_pergame'][key] = cells

for (m, s), accs in T5.items():
    key = f'{m}_{s}'
    res['t5_avg'][key] = [round(sum(accs)/9,1), *[round(x,1) for x in boot_games(accs, rng)]]
    if (m, s) in T5_NO_PERGAME:
        res['t5_pergame'][key] = None
        continue
    cells = []
    for gi, g in enumerate(GAMES):
        lo, hi = boot_binom(accs[gi]/100, PAIR_N[g], rng)
        cells.append([accs[gi], round(lo,1), round(hi,1)])
    res['t5_pergame'][key] = cells

for m in ('IV','QW'):
    for s in ('S1','S2'):
        res['subsample'][f'{m}_{s}'] = [
            [round(x,1) for x in subsample(load_vec(m, s, g), 18, rng)] for g in GAMES
        ]

with open(OUT,'w') as fh:
    json.dump(res, fh, indent=1)
print('written', OUT)
