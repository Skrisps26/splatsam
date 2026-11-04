"""Render recorded MuJoCo states, never draw an illustrative grasp."""
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from splatsam import rendering
from splatsam.sim.scene import build_model
from splatsam.sim.episode import run_episode
import mujoco


def episode_frames(scene,scenario,params,*,all_frames=False):
    result=run_episode(scene,scenario,params,record=True)
    trace=result.trace
    if not trace: raise ValueError('episode produced no states to render')
    if all_frames:
        samples=trace
    else:
        samples=[]
        for phase in ['close','lift','hold']:
            candidates=[t for t in trace if t['phase']==phase]
            samples.append(candidates[-1] if candidates else trace[-1])
    model=build_model(scene,scenario); data=mujoco.MjData(model)
    cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    target=scenario.positions[scenario.target_id]
    cam.lookat[:]=(target[0],target[1],.04); cam.distance=.65; cam.azimuth=90; cam.elevation=-60
    frames=[]
    with rendering.renderer(model,height=360,width=480) as renderer:
        for sample in samples:
            data.qpos[:]=json.loads(sample['qpos_json'])
            data.mocap_pos[0]=[sample['hand_x'],sample['hand_y'],sample['hand_z']]
            yaw=float(sample['hand_yaw']); data.mocap_quat[0]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
            mujoco.mj_forward(model,data); renderer.update_scene(data,camera=cam)
            frames.append(renderer.render().copy())
    return frames,result


def save_replay(scene,scenario,params,path:Path):
    frames,result=episode_frames(scene,scenario,params,all_frames=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    imageio.mimwrite(str(path),frames,fps=25,codec='libx264',macro_block_size=1)
    return result
