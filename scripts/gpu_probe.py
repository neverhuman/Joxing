"""Optional exact-scene GPU feature/parity spike; never a production backend."""
import argparse
import importlib.metadata
from pathlib import Path
import time

import mujoco
import numpy as np

from joxing.provenance import manifest,write_json
from joxing.scene import build_scene


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--worlds",type=int,default=8);args=p.parse_args()
    report=manifest()|{"experiment":"exact-dual-model-gpu-feature-spike","worlds":args.worlds,"passed":False}
    started=time.perf_counter()
    try:
        import mujoco_warp as mjw
        import warp as wp
        report["mujoco_warp"]=importlib.metadata.version("mujoco-warp")
        report["warp_lang"]=importlib.metadata.version("warp-lang")
        scene=build_scene();model=scene.model;report["scene_hash"]=scene.report["scene_hash"]
        report["dimensions"]=scene.report["dimensions"]
        data=mujoco.MjData(model);data.qpos[:]=scene.report["initial_qpos"];data.ctrl[:]=.02;mujoco.mj_forward(model,data)
        wp.init()
        with wp.ScopedDevice("cuda:0"):
            device=wp.get_device();report["gpu"]=device.name
            gpu_model=mjw.put_model(model)
            gpu_data=mjw.put_data(model,data,nworld=args.worlds,nconmax=128,njmax=1024)
            report["uploaded"]=True
            warm_started=time.perf_counter()
            mjw.step(gpu_model,gpu_data);wp.synchronize()
            report["first_step_compile_wall_s"]=time.perf_counter()-warm_started
            mujoco.mj_step(model,data)
            for _ in range(9):
                mjw.step(gpu_model,gpu_data);mujoco.mj_step(model,data)
            wp.synchronize()
            states=gpu_data.qpos.numpy()
            error=float(np.max(np.abs(states-data.qpos)))
            report["ten_step_qpos_max_error"]=error
            report["finite"]=bool(np.isfinite(states).all())
            # This is only short-state parity. No contact/referee completeness claim.
            report["passed"]=bool(report["finite"] and error<1e-4)
            report["limitations"]=["10 steps only","8-world candidate capacity, not a scaling sweep","no force/event/referee parity","no end-to-end RL throughput"]
    except Exception as exc:
        report["failure_type"]=type(exc).__name__;report["failure"]=str(exc)
    report["wall_s"]=time.perf_counter()-started
    write_json(args.out,report)
    print(report,flush=True)


if __name__=="__main__": main()
