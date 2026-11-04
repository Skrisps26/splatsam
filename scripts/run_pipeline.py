#!/usr/bin/env python3
"""Explicit phase-gated pipeline entry point; defaults to CPU synthetic work."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from splatsam.budget import BudgetTracker
from splatsam.llm.schemas import ScenarioFamily
from splatsam.planner import PlannerParams
from splatsam.utils import ROOT, save_json


def print_cost(stage: str) -> None:
    budget=BudgetTracker(ROOT/'runs/meta_budget.json')
    print(f'[{stage}] Meta USD {budget.total_usd}; GPU USD 0 (Phase A, no resources)')


def calibrate(scene, workers: int):
    from splatsam.curriculum.sampler import sample_scenario
    from splatsam.sim.episode import evaluate
    presets=[ScenarioFamily(), ScenarioFamily(friction_scale=(.5,1.2),mass_scale=(1,2),jitter_m=.04,clutter_count=(1,3),occluder_count=(0,2)), ScenarioFamily(friction_scale=(1,1.5),mass_scale=(.5,1),jitter_m=.01,clutter_count=(0,0),occluder_count=(0,0))]
    pilots=[]
    for index,family in enumerate(presets):
        scenarios=[sample_scenario(family,s,scene) for s in range(5000,5100)]
        start=time.perf_counter(); results=evaluate(scene,scenarios,PlannerParams(),workers=workers)
        rate=float(np.mean([r.success for r in results])); elapsed=time.perf_counter()-start
        pilots.append({'preset':index,'success_rate':rate,'n':100,'wall_seconds':elapsed,'family':family.model_dump(mode='json')})
        print(f'Pilot {index}: {rate:.1%}, 100 episodes in {elapsed:.3f}s',flush=True)
        if .25 <= rate <= .75:
            return family,pilots
    best=min(pilots,key=lambda p:abs(p['success_rate']-.5))
    raise RuntimeError(f'Pilot difficulty gate not met: best={best["success_rate"]:.1%}; inspect physics before freezing')


def handcrafted(args):
    from splatsam.sim.scene import handcrafted_scene
    from splatsam.sim.episode import evaluate, run_episode
    from splatsam.curriculum.sampler import split_scenarios
    from splatsam.eval.seal import seal_test
    from splatsam.eval.statistics import summarize
    scene=handcrafted_scene(args.seed)
    run_dir=ROOT/'runs'/args.run_id; run_dir.mkdir(parents=True,exist_ok=True)
    root=Path(args.root).resolve()
    family,pilots=calibrate(scene,args.workers)
    save_json(run_dir/'pilot.json',pilots)
    baseline=PlannerParams()
    seal_test(scene,family,baseline,root)
    save_json(run_dir/'scene.json',scene)
    save_json(run_dir/'family.json',family)
    baseline.save(run_dir/'baseline_params.json'); baseline.save(run_dir/'final_params.json')
    scenarios=split_scenarios(family,scene,'train')[:args.benchmark]
    start=time.perf_counter(); results=evaluate(scene,scenarios,baseline,workers=args.workers)
    elapsed=time.perf_counter()-start
    save_json(run_dir/'baseline_train.json',[r.model_dump(mode='json') for r in results])
    repeat=run_episode(scene,scenarios[0],baseline)
    assert repeat == results[0], 'serial/parallel determinism failed'
    summary={'mode':'synthetic','stage':'A2','baseline_train':summarize(results),'wall_seconds':elapsed,'episodes':len(results),'pilot':pilots[-1],'deterministic':True,'meta_cost_usd':str(BudgetTracker(ROOT/'runs/meta_budget.json').total_usd),'gpu_cost_usd':0}
    save_json(run_dir/'a2_summary.json',summary)
    print(summary,flush=True)
    if not .25<=sum(r.success for r in results)/len(results)<=.75:
        raise RuntimeError('Frozen baseline success outside 25–75% A2 gate')
    if elapsed>=60:
        raise RuntimeError('100-episode throughput gate exceeded 60s')
    print_cost('A2')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['handcrafted'],default='handcrafted')
    parser.add_argument('--root',default='data_synth/handcrafted')
    parser.add_argument('--run-id',default='handcrafted')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--benchmark',type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.benchmark<=150: parser.error('benchmark must be in 1..150')
    if Path(args.run_id).name!=args.run_id: parser.error('run-id must be a single directory name')
    return handcrafted(args)

if __name__=='__main__': main()
