import os
import sys
import numpy as np
import torch
from os import path as osp
from pathlib import Path
# --- Visualization / Body Model Imports ---
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors as vis_colors
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from dataset_utils.amass_visualizer import AmassVisualizer
from src.data_processing import AMASSTAdapter
# root_orient: global orientation (rotation of the root joint)
# pose_body: body pose (joint rotations excluding hands)
# pose_hand: hand pose (finger articulations)
# trans: global translation (body position in world coordinates)
# betas: body shape coefficients (static PCA-based body shape)
# dmpls: dynamic soft-tissue deformation coefficients (per-frame motion corrections)

# ============================================================
# CONFIG SECTION
# ============================================================

CONFIG = {
    "motion_dir": "/workspace/data/amass/raw/Aude/INF_Basketball_S2_01_poses.npz",
    "support_dir": "motion_data/",
    "gender": "female",                # "male" | "female" | "neutral"
    "num_betas": 16,
    "num_dmpls": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "viewer": {"width": 800, "height": 800, "use_offscreen": True},
    "render": {
        "default_pose_t": 0,
        "video_name": "walk_cycle",
        "start_frame": 0,
        "end_frame": 300,
        "step": 1,
        "fps": 60,
    },
    "debug": {
        "enabled": False,
        "host": "0.0.0.0",
        "port": 5678,
    },
}


# ============================================================
# UTILITIES
# ============================================================

def parse_debug_flag(cfg: dict):
    """Enable debugpy if 'debug' argument is present."""
    if len(sys.argv) > 1 and sys.argv[1].lower() == "debug":
        try:
            import debugpy
            print(f"[DEBUG] Starting debugpy on {cfg['debug']['host']}:{cfg['debug']['port']} ...")
            debugpy.listen((cfg["debug"]["host"], cfg["debug"]["port"]))
            print("[DEBUG] Waiting for debugger to attach...")
            debugpy.wait_for_client()
            cfg["debug"]["enabled"] = True
            print("[DEBUG] Debugger attached. Continuing execution.")
        except ImportError:
            print("[WARN] debugpy not installed. Run `pip install debugpy` to use debug mode.")
    return cfg


def load_body_model(cfg: dict) -> tuple[BodyModel, str]:
    """Load BodyModel and DMPL files."""
    gender = cfg["gender"]
    support_dir = cfg["support_dir"]

    bm_path = osp.join(support_dir, f"body_models/smplh/{gender}/model.npz")
    dmpl_path = osp.join(support_dir, f"body_models/dmpls/{gender}/model.npz")

    bm = BodyModel(
        bm_fname=bm_path,
        num_betas=cfg["num_betas"],
        num_dmpls=cfg["num_dmpls"],
        dmpl_fname=dmpl_path,
    ).to(torch.device(cfg["device"]))
    return bm


def load_colors() -> dict:
    """Prepare color dictionary."""
    return {
        "grey": vis_colors["grey"],
        "red": vis_colors["red"],
        "green": vis_colors["green"],
    }


def load_amass_data(cfg: dict) -> str:
    """Load AMASS .npz dataset."""
    amass_npz = cfg["motion_dir"]
    if not osp.exists(amass_npz):
        raise FileNotFoundError(f"Dataset not found: {amass_npz}")
    return amass_npz

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    cfg = parse_debug_flag(CONFIG)

    # 1. Load dataset and model
    # amass_npz_fname = load_amass_data(cfg)
    # bm, dmpl_fname = load_body_model(cfg)

    # 2. Prepare viewer and visualization environment
    mv = MeshViewer(**cfg["viewer"])

    adapter = AMASSTAdapter()
    adapter._post_init(Path("/workspace/data/amass/raw/Aude"))

    faces = c2c(adapter.body_model.f)
    color_map = load_colors()

    # 3. Build the visualizer
    vis = AmassVisualizer.build_from_amass(
        amass_npz_fname="/workspace/data/amass/raw/Aude/INF_Basketball_S2_01_poses.npz",
        bm=adapter.body_model,
        mv=mv,
        faces=faces,
        colors=color_map,
        comp_device=cfg["device"],
    )

    # ========================================================
    # Visualization calls
    # ========================================================
    # render_joints
    vis.render_joints(0, joint_idx=1)
    # vis.render_joints(0,  joint_idx=2)
    # vis.render_joints(0,  joint_idx=3)
    # vis.render_joints(0,  joint_idx=20)
    # vis.render_joints(0,  joint_idx=21)
    # T_pose_offsets = vis.offsets(0) #(1, J, 3)
    # vis.render_offsets(0)
    # orientations = vis.orientations(all=True) #(T,n_joints,4)
    # print(T_pose_offsets.shape)
    # print(orientations.shape)
    # vis.render_default_pose(t=cfg["render"]["default_pose_t"])
    # vis.joints(t=25)
    # vis.render_joints(t=25)
    # vis.render_pose_body_betas(t=500)
    # vis.render_pose_body_betas_hands(t=0)

    # vis.export_motion_video(
    #     name=cfg["render"]["video_name"],
    #     start=cfg["render"]["start_frame"],
    #     step=cfg["render"]["step"],
    #     fps=cfg["render"]["fps"],
    # )
    # T = orientations.shape[0]
    # num_joints = orientations.shape[1]
    # eo_tiled = np.tile(T_pose_offsets, (T, 1, 1))
    # M_eo = np.concatenate([eo_tiled, orientations], axis=-1)

if __name__ == "__main__":
    main()
