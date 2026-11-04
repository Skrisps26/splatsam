"""Validation-selected parameter edits; held-out specs are inaccessible here."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
from diskcache import Cache

from splatsam.curriculum.sampler import split_scenarios
from splatsam.eval.statistics import summarize
from splatsam.llm.schemas import CriticLabel, EpisodeResult, PlannerPatch
from splatsam.loop.acceptance import acceptance
from splatsam.planner import PlannerParams
from splatsam.sim.episode import evaluate
from splatsam.utils import canonical, digest, save_json


def improve(
    scene, family, baseline: PlannerParams, client, run_dir: Path, *, workers=4, max_rounds=8, critic=True
) -> dict:
    if not 1 <= max_rounds <= 8:
        raise ValueError("round bound is 1..8")
    train = split_scenarios(family, scene, "train")
    validation = split_scenarios(family, scene, "validation")
    cache = Cache(str(run_dir / "episode_cache"))
    history = []
    current = baseline
    rejected = 0
    validation_evaluations = 0
    fresh_validation = 0

    def batch(scenarios, params, is_validation=False):
        nonlocal validation_evaluations, fresh_validation
        results = [None] * len(scenarios)
        missing = []
        indices = []
        for i, s in enumerate(scenarios):
            key = digest(
                {
                    "scene": scene.model_dump(mode="json"),
                    "scenario": s.model_dump(mode="json"),
                    "params": params.to_dict(),
                    "physics": "parallel-jaw-v1",
                }
            )
            found = cache.get(key)
            if found is None:
                missing.append(s)
                indices.append((i, key))
            else:
                results[i] = EpisodeResult.model_validate(found)
        if missing:
            for (i, key), result in zip(indices, evaluate(scene, missing, params, workers=workers)):
                results[i] = result
                cache[key] = result.model_dump(mode="json")
        if is_validation:
            validation_evaluations += len(scenarios)
            fresh_validation += len(missing)
        return results

    for round_index in range(max_rounds):
        training = batch(train, current)
        diagnoses = []
        if critic:
            ambiguous = [
                (s, r) for s, r in zip(train, training) if r.failure_mode in {"instability", "slip_or_drop"}
            ][:3]
            for scenario, result in ambiguous:
                try:
                    from splatsam.sim.replay import episode_frames

                    frames, _ = episode_frames(scene, scenario, current)
                    content = [
                        {
                            "type": "input_text",
                            "text": f"Simulator failure {result.failure_mode}; diagnose only, never change scoring.",
                        }
                    ]
                    for frame in frames:
                        ok, png = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        if not ok:
                            raise ValueError("keyframe encoding failed")
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64," + base64.b64encode(png).decode(),
                            }
                        )
                    label = client.generate(
                        CriticLabel,
                        model="muse-spark-1.3",
                        instructions="Return one fixed failure enum and one-line reason.",
                        input=[{"type": "message", "role": "user", "content": content}],
                        stage="critic",
                    )
                    diagnoses.append({"seed": scenario.seed, **label.model_dump(mode="json")})
                except (ValueError, RuntimeError, OSError) as error:
                    diagnoses.append(
                        {
                            "seed": scenario.seed,
                            "label": result.failure_mode,
                            "reason": f"critic unavailable: {type(error).__name__}",
                        }
                    )
        proposal = client.generate(
            PlannerPatch,
            model="muse-wonder-1.0",
            instructions="Propose bounded rule-planner parameter edits only. Never change scoring/family/splits.",
            input=canonical(
                {
                    "current": current.to_dict(),
                    "bounds": current.BOUNDS,
                    "training": summarize(training),
                    "examples": [r.model_dump(exclude={"trace"}) for r in training if not r.success][:5],
                    "history": history,
                }
            ),
            stage="proposal",
        )
        row = {
            "round": round_index + 1,
            "patch": proposal.patch,
            "proposal_reason": proposal.reason,
            "current_hash": current.hash,
            "training": summarize(training),
            "critic": diagnoses,
        }
        try:
            candidate = current.patch(proposal.patch)
            old_validation = batch(validation, current, True)
            new_validation = batch(validation, candidate, True)
            row.update(acceptance(old_validation, new_validation, seed=42 + round_index))
            row.update(
                {
                    "candidate_hash": candidate.hash,
                    "baseline_validation": summarize(old_validation),
                    "candidate_validation": summarize(new_validation),
                }
            )
            if row["accepted"]:
                current = candidate
                rejected = 0
            else:
                rejected += 1
        except ValueError as error:
            row.update({"accepted": False, "reason": str(error)})
            rejected += 1
        history.append(row)
        save_json(run_dir / "rounds.json", history)
        current.save(run_dir / "final_params.json")
        print(f"Round {round_index + 1}: {row['reason']}; gain={row.get('gain')}", flush=True)
        if rejected >= 3:
            break
    result = {
        "history": history,
        "final_params": current.to_dict(),
        "validation_evaluations": validation_evaluations,
        "fresh_validation_simulations": fresh_validation,
    }
    save_json(run_dir / "loop.json", result)
    cache.close()
    return result
