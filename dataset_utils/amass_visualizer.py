from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
import trimesh
import tempfile
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Utility functions
def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_torch(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)

@dataclass
class AmassVisualizer:
    amass_npz_fname: str
    bm: Any
    kintree_table: Any
    mv: Any
    faces: np.ndarray
    colors: Dict[str, np.ndarray]
    body_parms: Dict[str, torch.Tensor]
    device: torch.device
    out_root: str = 'amass_vis_out'
    _full_output: Optional[Any] = field(default=None, init=False, repr=False)

    @property
    def dirs(self) -> Dict[str, str]:
        base = os.path.join(self.out_root, os.path.splitext(os.path.basename(self.amass_npz_fname))[0])
        paths = {
            'base': base,
            'images': os.path.join(base, 'images'),
            'videos': os.path.join(base, 'videos'),
            'meshes': os.path.join(base, 'meshes'),
            'data': os.path.join(base, 'data'),
        }
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
        return paths

    def _save_image(self, img: np.ndarray, name: str) -> str:
        fp = os.path.join(self.dirs['images'], f"{name}.png")
        Image.fromarray(img.astype(np.uint8)).save(fp)
        return fp

    def _save_mesh(self, vertices: np.ndarray, name: str) -> str:
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=self.faces,
            process=False,
            vertex_colors=np.tile(self.colors.get('grey', np.array([200, 200, 200], dtype=np.uint8)), (vertices.shape[0], 1))
        )
        fp = os.path.join(self.dirs['meshes'], f"{name}.ply")
        mesh.export(fp)
        return fp

    @property
    def T(self) -> int:
        for k in ['pose_body', 'pose_hand', 'root_orient', 'trans', 'dmpls']:
            if k in self.body_parms:
                return int(self.body_parms[k].shape[0])
        return 1

    def _forward_subset(self, keys: Iterable[str]) -> Any:
        subset = {k: v for k, v in self.body_parms.items() if k in set(keys)}
        return self.bm(**subset)

    def forward_all(self) -> Any:
        if self._full_output is None:
            self._full_output = self.bm(**self.body_parms)
        return self._full_output

    def joints(self, t: int, include_hands: bool = False) -> np.ndarray:
        out = self.forward_all()
        J = getattr(out, "Jtr", getattr(out, "J", None))
        if J is None:
            raise AttributeError("BodyModel output has no joints (expected Jtr or J).")

        J = _to_numpy(J)
        jt = J[t]

        # Remove hand joints by default
        if not include_hands:
            jt = jt[:22]  # body-only subset

        np.save(os.path.join(self.dirs["data"], f"joints_t{t}.npy"), jt)
        return jt

    def offsets(self, t: int, include_hands: bool = False):
        out = self.forward_all()
        J = getattr(out, "Jtr", getattr(out, "J", None))
        if J is None:
            raise AttributeError("BodyModel output has no joints (expected Jtr or J).")
        J = _to_numpy(J)
        parent_kintree = _to_numpy(self.kintree_table[0])
        n_joints = len(parent_kintree)
        
        jt = J[t]
        offsets = jt - jt[parent_kintree]

        indices = np.arange(1, n_joints)

        if not include_hands:
            indices = indices[indices < 22]
        offsets = offsets[indices]
        
        np.save(os.path.join(self.dirs["data"], f"offsets_t{t}.npy"), offsets)
        print(f"[INFO] Saved offsets → offsets_t{t}.npy")
        return offsets[np.newaxis, :, :]  # (1, n_joints, 3)
    
    def orientations(self, t: int = 0, include_hands: bool = False, all: bool = False) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R

        out = self.forward_all()
        full_pose = getattr(out, "full_pose", None)
        if full_pose is None:
            raise AttributeError("BodyModel output has no full_pose attribute.")

        full_pose = _to_numpy(full_pose)
        parents = self.kintree_table[0].long().cpu().numpy()
        T = full_pose.shape[0]  # total frames
        n_joints = len(parents)

        def compute_frame(frame_idx: int):
            aa = full_pose[frame_idx].reshape(-1, 3)
            Rmats = R.from_rotvec(aa).as_matrix()
            local_quats = []
            for i in range(1, n_joints):
                if not include_hands and (i >= 22 or parents[i] >= 22):
                    continue
                R_parent = Rmats[parents[i]]
                R_child = Rmats[i]
                R_rel = R_parent.T @ R_child
                q_rel = R.from_matrix(R_rel).as_quat()
                local_quats.append(q_rel)
            return np.array(local_quats)

        if all:
            all_quats = []
            for frame_idx in tqdm(range(T), desc="Saving orientations"):
                local_quats = compute_frame(frame_idx)
                np.save(os.path.join(self.dirs["data"], f"orientations_t{frame_idx}.npy"), local_quats)
                all_quats.append(local_quats)
            all_quats = np.stack(all_quats, axis=0)
            np.save(os.path.join(self.dirs["data"], "orientations_all.npy"), all_quats)
            print(f"[INFO] Saved all relative joint orientations → orientations_all.npy")
            return all_quats
        else:
            local_quats = compute_frame(t)
            np.save(os.path.join(self.dirs["data"], f"orientations_t{t}.npy"), local_quats)
            print(f"[INFO] Saved relative joint orientations → orientations_t{t}.npy")
            return local_quats
    
    def render_joints(
        self,
        t: int = 0,
        name: str = "body_joints",
        include_hands: bool = False
    ) -> str:
        from body_visualizer.mesh.sphere import points_to_spheres

        keys = ["pose_body", "betas"]
        if include_hands and "pose_hand" in self.body_parms:
            keys.append("pose_hand")

        out = self._forward_subset(keys)
        J = getattr(out, "Jtr", getattr(out, "J", None))
        if J is None:
            raise AttributeError("BodyModel output has no joints (expected Jtr or J).")

        J = _to_numpy(J)
        jt = J[t]

        # Remove hand joints by default
        if not include_hands:
            jt = jt[:22]

        joints_mesh = points_to_spheres(
            jt,
            point_color=self.colors.get("red", np.array([255, 0, 0], dtype=np.uint8)),
            radius=0.005,
        )

        self.mv.set_static_meshes([joints_mesh])
        img = self.mv.render(render_wireframe=False)
        img_path = os.path.join(self.dirs["images"], f"{name}_t{t}.png")
        Image.fromarray(img.astype(np.uint8)).save(img_path)
        print(f"[INFO] Saved joint visualization → {img_path}")
        return img_path

    def render_joints_and_parents(
        self,
        t: int = 0,
        child: int = 0,
        name: str = "body_joints",
        include_hands: bool = False
    ) -> str:
        from body_visualizer.mesh.sphere import points_to_spheres

        keys = ["pose_body", "betas"]
        if include_hands and "pose_hand" in self.body_parms:
            keys.append("pose_hand")

        out = self._forward_subset(keys)
        J = getattr(out, "Jtr", getattr(out, "J", None))
        if J is None:
            raise AttributeError("BodyModel output has no joints (expected Jtr or J).")

        J = _to_numpy(J)
        jt = J[t]
        child_joint = jt[child]
        parent_joint = jt[self.kintree_table[0,child]]

        # # Remove hand joints by default
        # if not include_hands:
        #     jt = jt[:22]
        points = np.vstack([parent_joint, child_joint])

        # colors per joint: parent=red, child=blue
        colors = np.array([
            [255, 0, 0],   # parent
            [0, 0, 255]    # child
        ], dtype=np.uint8)

        joints_mesh = points_to_spheres(points, point_color=colors, radius=0.005)

        self.mv.set_static_meshes([joints_mesh])
        img = self.mv.render(render_wireframe=False)
        img_path = os.path.join(self.dirs["images"], f"{name}_t{t}_child{child}.png")
        Image.fromarray(img.astype(np.uint8)).save(img_path)
        print(f"[INFO] Saved joint visualization → {img_path}")
        return img_path
    
    def render_offsets(
        self,
        t: int = 0,
        name: str = "offsets",
        include_hands: bool = False
        ) -> str:
        keys = ["pose_body", "betas"]
        if include_hands and "pose_hand" in self.body_parms:
            keys.append("pose_hand")

        out = self._forward_subset(keys)
        J = getattr(out, "Jtr", getattr(out, "J", None))
        if J is None:
            raise AttributeError("BodyModel output has no joints (expected Jtr or J).")

        J = _to_numpy(J)
        jt = J[t]

        # Remove hand joints by default
        if not include_hands:
            jt = jt[:22]

        parents = self.bm.kintree_table[0].long().cpu().numpy()

        # Create white background
        H, W = 512, 512
        img = np.ones((H, W, 3), dtype=np.uint8) * 255

        # Normalize with padding (zoom out)
        jt_min, jt_max = jt.min(axis=0), jt.max(axis=0)
        pad = 0.25 * (jt_max - jt_min)          # expand bounds by 25%
        jt_min -= pad
        jt_max += pad
        norm = (jt - jt_min) / (jt_max - jt_min + 1e-8)
        px = (norm[:, 0] * (W - 1)).astype(int)
        py = ((1 - norm[:, 1]) * (H - 1)).astype(int)  # flip vertically

        # Draw edges
        for i in range(1, len(parents)):
            if not include_hands and (i >= 22 or parents[i] >= 22):
                continue
            p = (px[parents[i]], py[parents[i]])
            c = (px[i], py[i])
            rr, cc = np.linspace(p[1], c[1], 50).astype(int), np.linspace(p[0], c[0], 50).astype(int)
            valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            img[rr[valid], cc[valid]] = [0, 0, 0]

        img_path = os.path.join(self.dirs["images"], f"{name}_t{t}.png")
        Image.fromarray(img).save(img_path)
        print(f"[INFO] Saved offset visualization → {img_path}")
        return img_path

    def _render_vertices(self, verts: torch.Tensor, name: str) -> str:
        verts_np = _to_numpy(verts)
        mesh = trimesh.Trimesh(
            vertices=verts_np,
            faces=self.faces,
            process=False,
            vertex_colors=np.tile(self.colors.get('grey', np.array([200, 200, 200], dtype=np.uint8)), (verts_np.shape[0], 1))
        )
        self.mv.set_static_meshes([mesh])
        img = self.mv.render(render_wireframe=False)
        return self._save_image(img, name)

    def render_default_pose(self, t: int = 0, name: str = 'default_pose') -> Tuple[str, str]:
        # Use the same subset as pose_body + betas (avoids bird's-eye bug)
        out = self._forward_subset(['pose_body', 'betas'])
        V = _to_numpy(out.v)
        img_fp = self._render_vertices(V[t], f"{name}_t{t}")
        ply_fp = self._save_mesh(V[t], f"{name}_t{t}")
        return img_fp, ply_fp

    def render_pose_body_betas(self, t: int = 0, name: str = 'pose_body_betas') -> Tuple[str, str]:
        out = self._forward_subset(['pose_body', 'betas'])
        V = out.v
        img_fp = self._render_vertices(V[t], f"{name}_t{t}")
        ply_fp = self._save_mesh(_to_numpy(V[t]), f"{name}_t{t}")
        return img_fp, ply_fp

    def render_pose_body_betas_hands(self, t: int = 0, name: str = 'pose_body_betas_hands') -> Tuple[str, str]:
        out = self._forward_subset(['pose_body', 'betas', 'pose_hand'])
        V = out.v
        img_fp = self._render_vertices(V[t], f"{name}_t{t}")
        ply_fp = self._save_mesh(_to_numpy(V[t]), f"{name}_t{t}")
        return img_fp, ply_fp

    def export_motion_video(
        self,
        name: str = 'motion',
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
        fps: int = 30
    ) -> str:
        """
        Render motion video from a consistent front-facing view.
        Uses only pose_body + betas (+hands if present) to avoid bird's-eye offset.
        """
        # --- Forward only pose-related params (ignore trans, root_orient) ---
        keys = ['pose_body', 'betas']
        if 'pose_hand' in self.body_parms:
            keys.append('pose_hand')
        out = self._forward_subset(keys)

        V = out.v
        T = V.shape[0]
        s = max(0, int(start))
        e = int(end) if end is not None else T
        e = min(e, T)
        idxs = list(range(s, e, max(1, int(step))))

        tmp_dir = tempfile.mkdtemp(prefix=f"{name}_frames_", dir=self.dirs["videos"])
        print(f"[INFO] Rendering {len(idxs)} frames into temporary directory: {tmp_dir}")

        frame_paths = []
        for i, t in enumerate(tqdm(idxs, desc="Rendering frames", unit="frame")):
            verts_np = _to_numpy(V[t])
            mesh = trimesh.Trimesh(
                vertices=verts_np,
                faces=self.faces,
                process=False,
                vertex_colors=np.tile(
                    self.colors.get("grey", np.array([200, 200, 200], dtype=np.uint8)),
                    (verts_np.shape[0], 1)
                ),
            )
            self.mv.set_static_meshes([mesh])
            img = self.mv.render(render_wireframe=False)
            frame_path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
            Image.fromarray(img.astype(np.uint8)).save(frame_path)
            frame_paths.append(frame_path)

        video_path = os.path.join(self.dirs["videos"], f"{name}.mp4")
        print(f"[INFO] Encoding video → {video_path}")

        with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
            for fpath in tqdm(frame_paths, desc="Encoding video", unit="frame"):
                frame = imageio.imread(fpath)
                writer.append_data(frame)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[CLEANUP] Removed temporary frame directory: {tmp_dir}")

        return video_path

    @staticmethod
    def build_from_amass(amass_npz_fname: str, bm: Any, mv: Any, faces: np.ndarray, colors: Dict[str, np.ndarray], dmpl_fname: Optional[str] = None, comp_device: Optional[str] = None) -> 'AmassVisualizer':
        device = torch.device(comp_device or 'cpu')
        bdata = np.load(amass_npz_fname)
        body_parms: Dict[str, torch.Tensor] = {}

        poses = bdata['poses']
        body_parms['root_orient'] = _to_torch(poses[:, :3], device)
        body_parms['pose_body'] = _to_torch(poses[:, 3:66], device)
        if poses.shape[1] > 66:
            body_parms['pose_hand'] = _to_torch(poses[:, 66:], device)

        if 'trans' in bdata:
            body_parms['trans'] = _to_torch(bdata['trans'], device)
        if 'betas' in bdata:
            betas = bdata['betas']
            if betas.ndim == 1:
                T = poses.shape[0]
                betas = np.repeat(betas[None, :], repeats=T, axis=0)
            body_parms['betas'] = _to_torch(betas, device)
        if 'dmpls' in bdata:
            body_parms['dmpls'] = _to_torch(bdata['dmpls'], device)
        elif dmpl_fname and os.path.exists(dmpl_fname):
            dmpl = np.load(dmpl_fname)
            if isinstance(dmpl, np.lib.npyio.NpzFile) and 'dmpls' in dmpl:
                body_parms['dmpls'] = _to_torch(dmpl['dmpls'], device)

        colors_packed = {k: np.asarray(v, dtype=np.uint8) for k, v in colors.items()}

        vis = AmassVisualizer(
            amass_npz_fname=amass_npz_fname,
            bm=bm,
            kintree_table=bm.kintree_table,
            mv=mv,
            faces=faces.astype(np.int32),
            colors=colors_packed,
            body_parms=body_parms,
            device=device,
        )

        manifest = {
            'amass_npz_fname': amass_npz_fname,
            'dmpl_fname': dmpl_fname,
            'T': vis.T,
            'keys': sorted(list(body_parms.keys())),
            'device': str(device)
        }
        with open(os.path.join(vis.dirs['base'], 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

        return vis