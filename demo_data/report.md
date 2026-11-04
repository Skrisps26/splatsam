# Splatsam evaluation report

**Mode: synthetic.** Synthetic illustration only, not real-video evidence.

Validation-selected planner; held-out effect reported separately

## Sealed test

| Planner | Successes / n | Success rate | Wilson 95% CI |
| --- | ---: | ---: | --- |
| Simple default settings | 69 / 200 | 34.5% | [28.3%, 41.3%] |
| Final rule-based settings | 95 / 200 | 47.5% | [40.7%, 54.4%] |

Paired gain: 13.0%; paired bootstrap 95% CI [0.085, 0.175].

### Failure breakdown

| Mode | Baseline | Final |
| --- | ---: | ---: |
| instability | 65 | 65 |
| knock_over | 35 | 34 |
| no_grasp_found | 0 | 0 |
| occluder_collision | 0 | 0 |
| slip_or_drop | 31 | 6 |
| timeout | 0 | 0 |

## Validation selection

6 rounds; 3 accepted. 1200 planner-scenario validation evaluations, 700 fresh simulations. Validation is optimistic because used for selection.

| Round | Accepted | Patch | Reason |
| --- | --- | --- | --- |
| 1 | True | {'grip_safety_factor': 1.6} | accepted |
| 2 | True | {'grip_safety_factor': 2.0} | accepted |
| 3 | True | {'grip_safety_factor': 2.4} | accepted |
| 4 | False | {'grip_safety_factor': 2.8} | mean gain below 2pp |
| 5 | False | {'grip_safety_factor': 3.2} | mean gain below 2pp |
| 6 | False | {'grip_safety_factor': 3.6} | mean gain below 2pp |

## Reconstruction and physics

Scale source: paper_homography; independently measured synthetic error: 0.002507040529839729. Sheet self-consistency: {'aspect_ratio_error': 0.0002644009480050613, 'paper_area_error': 0.0013078562927505422, 'scene_origin_x': 0.2684993539005518, 'scene_origin_y': 0.33670205110684037}.

| ID | Shape/source | IoU | Mass kg/source | Friction/source | Height source |
| --- | --- | --- | --- | --- | --- |
| paper | box/primitive;clutter_dimension_gate | 0.9905044323164769 | 0.05047/estimate | 0.5/estimate | estimate |
| remote_1 | box/primitive | 0.966553436111837 | 0.1999/estimate | 0.5/estimate | estimate |
| remote_2 | box/primitive | 0.9572903225806452 | 0.2022/estimate | 0.5/estimate | estimate |
| ball | sphere/primitive | 0.9659090909090909 | 0.07373/estimate | 1/estimate | estimate |
| marker_1 | horizontal_cylinder/primitive | 0.9013854930725347 | 0.03395/estimate | 0.5/estimate | estimate |
| marker_2 | horizontal_cylinder/primitive | 0.9130966952264382 | 0.03483/estimate | 0.5/estimate | estimate |

## Spend

{'meta_usd': 0.0, 'gpu_usd': 0}

## Limits

- Simulation only; floating two-finger hand, no arm or inverse kinematics.
- Rule-based planner parameter tuning, not training a neural policy.
- No real-robot transfer is claimed.
- Object heights and hidden-side geometry are estimated unless explicitly measured.
- Mass and friction are estimates except fields labeled measured.
- Consistency IoU checks footprint appearance only, not height, mass, friction or collision geometry.
- Scale may be uncertain by a few percent due to paper curl, object height and camera tilt.
- Results are specific to the captured scene and this simulation/scoring configuration.
- Validation was used for accept/reject decisions and is therefore optimistic.
- Local hash seals guard accidental leakage/tampering, not a malicious repository owner.
