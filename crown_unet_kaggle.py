# %% [markdown]
# AI Crown Design — conditional 3D U-Net occupancy completion
# =============================================================
# Thesis: "Assessment of Accuracy of AI versus conventional Digital Design of
# fixed dental restoration." Reference / gold standard = exocad `design.stl`.
#
# Pipeline (run cells top to bottom on a Kaggle GPU notebook, internet ON):
#   1. config            2. constructionInfo parser   3. preprocess -> voxel cache
#   4. dataset/augment   5. 3D U-Net                   6. train
#   7. inference + mesh  8. evaluate (IoU / Hausdorff / RMS) -> metrics.csv
#
# Input layout expected at INPUT_ROOT, one folder per case (as inspected):
#   <case>/upper.stl  lower.stl  design.stl  constructionInfo  [tooth_model.obj ...]
# The prepared arch is chosen from the target tooth's FDI quadrant
# (1-2 -> upper, 3-4 -> lower); the other arch is the antagonist. Both arches are
# already articulated in one world frame, so the antagonist gives the real
# opposing profile the protocol asks the model to learn from.
#
# NOTE: this file was authored without a GPU/data sandbox; run cell-by-cell on
# Kaggle and expect to tune RES / channels / epochs. Nothing here needs the
# local `dcprep` package — it is self-contained.

# %% setup ---------------------------------------------------------------------
# !pip -q install trimesh scikit-image rtree manifold3d
import os, glob, json, math, random, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

SEED = 1337
random.seed(SEED); np.random.seed(SEED)

# %% 1. config -----------------------------------------------------------------
class CFG:
    INPUT_ROOT = "/kaggle/input/single-crown"      # mounts here when the dataset is attached
    CACHE      = "/kaggle/working/cache"           # voxel .npz cache
    CKPT       = "/kaggle/working/ckpt"
    OUT        = "/kaggle/working/out"

    ROI_MM     = 20.0        # cube side around the tooth centre (covers crown+context)
    RES        = 64          # voxels per side (64 fast pilot; bump to 96/128 for final)
    PITCH      = ROI_MM / RES

    # input channels: prep-arch shell, antagonist shell  (add neighbours later if wanted)
    IN_CH      = 2
    SPLIT      = (0.60, 0.10, 0.30)   # train / val / test  (by ORIGINAL case, no leakage)

    EPOCHS     = 120
    BATCH      = 4
    LR         = 1e-3
    BASE_CH    = 24          # U-Net width; ~6-8M params at 3 levels
    AMP        = True
    TOL_MM     = 0.10        # clinical equivalence margin for the stats write-up

for d in (CFG.CACHE, CFG.CKPT, CFG.OUT):
    os.makedirs(d, exist_ok=True)

# %% 2. constructionInfo parser ------------------------------------------------
# Minimal port of the validated dcprep parser. Pulls, per tooth: FDI number,
# Center, and the 4x4 ToothModelMatrix (row-vector convention: world = [x y z 1] @ M),
# whose inverse canonicalises the tooth into a centred, axis-aligned frame.

def _f(node, tag, default=None):
    e = node.find(tag)
    return float(e.text) if e is not None and e.text not in (None, "") else default

def _matrix(node, tag):
    m = node.find(tag)
    if m is None:
        return None
    M = np.eye(4)
    for r in range(4):
        for c in range(4):
            e = m.find(f"_{r}{c}")
            if e is not None and e.text is not None:
                M[r, c] = float(e.text)
    return M

def parse_construction_info(path):
    root = ET.parse(path).getroot()          # root tag is the misspelled <ConstuctionInfo>
    teeth = []
    for t in root.findall("./Teeth/Tooth"):
        num = t.find("Number")
        if num is None:
            continue
        fdi = int(num.text)
        ctr = t.find("Center")
        center = np.array([_f(ctr, "x", 0), _f(ctr, "y", 0), _f(ctr, "z", 0)]) if ctr is not None else None
        margin = np.array([[_f(v, "x", 0), _f(v, "y", 0), _f(v, "z", 0)]
                           for v in t.findall("./Margin/Vec3")], float)   # world coords
        teeth.append(dict(fdi=fdi, center=center,
                          tmm=_matrix(t, "ToothModelMatrix"),
                          margin=margin if len(margin) else None))
    return teeth

def world_to_canonical(verts, tmm):
    """world (N,3) -> canonical frame via inverse ToothModelMatrix (row-vector convention)."""
    Minv = np.linalg.inv(tmm)
    h = np.c_[verts, np.ones(len(verts))]
    return (h @ Minv)[:, :3]

def fdi_arch(fdi):
    """upper if quadrant 1/2, lower if 3/4 (and deciduous 5/6 upper, 7/8 lower)."""
    q = fdi // 10
    return "upper" if q in (1, 2, 5, 6) else "lower"

# %% 3. preprocess -> voxel cache ---------------------------------------------
import trimesh
from trimesh.voxel import creation as vcreate

def _load(path):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m

def _voxelise(verts_faces, fill):
    """Rasterise a mesh (already in canonical mm coords) onto the fixed ROI grid.
    Returns a (RES,RES,RES) float array. `fill` -> solid interior (for the crown)."""
    v, f = verts_faces
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    r = CFG.RES // 2
    vg = vcreate.local_voxelize(mesh, point=np.zeros(3), pitch=CFG.PITCH,
                                radius=r, fill=fill)
    if vg is None:
        return np.zeros((CFG.RES,) * 3, np.float32)
    grid = vg.matrix.astype(np.float32)            # (2r+1)^3 centred on origin
    # crop/pad to exactly RES^3
    out = np.zeros((CFG.RES,) * 3, np.float32)
    s = [min(CFG.RES, grid.shape[i]) for i in range(3)]
    out[:s[0], :s[1], :s[2]] = grid[:s[0], :s[1], :s[2]]
    return out

def preprocess():
    # discover cases by locating constructionInfo files. Try the configured root
    # first; if the mount slug differs, fall back to scanning all of /kaggle/input,
    # so we don't care what Kaggle named the mount.
    found = glob.glob(f"{CFG.INPUT_ROOT}/**/constructionInfo", recursive=True)
    if not found:
        found = glob.glob("/kaggle/input/**/constructionInfo", recursive=True)
    cidirs = sorted({os.path.dirname(p) for p in found})
    root = os.path.commonpath(cidirs) if cidirs else CFG.INPUT_ROOT
    print(f"found {len(cidirs)} case folders (root: {root})")
    manifest = []
    for cdir in cidirs:
        ci = os.path.join(cdir, "constructionInfo")
        dpath = os.path.join(cdir, "design.stl")
        if not (os.path.exists(ci) and os.path.exists(dpath)):
            continue
        teeth = parse_construction_info(ci)
        if not teeth:
            continue
        design = _load(dpath)
        dcen = design.vertices.mean(0)
        # match this folder's single design.stl to its tooth (split two-prep folders
        # share a constructionInfo listing both teeth -> pick the nearest by Center)
        tooth = min(teeth, key=lambda t: np.inf if t["center"] is None
                    else np.linalg.norm(t["center"] - dcen))
        if tooth["tmm"] is None:
            continue
        arch = fdi_arch(tooth["fdi"])
        prep_p = os.path.join(cdir, f"{arch}.stl")
        anta_p = os.path.join(cdir, f"{'lower' if arch == 'upper' else 'upper'}.stl")
        if not (os.path.exists(prep_p) and os.path.exists(anta_p)):
            continue

        tmm = tooth["tmm"]
        prep = _load(prep_p); anta = _load(anta_p)
        prep_c = (world_to_canonical(prep.vertices, tmm), prep.faces)
        anta_c = (world_to_canonical(anta.vertices, tmm), anta.faces)
        des_c  = (world_to_canonical(design.vertices, tmm), design.faces)

        x = np.stack([_voxelise(prep_c, fill=False),
                      _voxelise(anta_c, fill=False)], 0)        # (2,R,R,R) shells
        y = _voxelise(des_c, fill=True)[None]                   # (1,R,R,R) solid crown
        if y.sum() < 10:                                        # voxelisation failed
            continue

        case_id = os.path.basename(cdir)
        # group key = original case (strip split suffix) so split copies stay together
        group = case_id.split("-copy")[0]
        np.savez_compressed(os.path.join(CFG.CACHE, f"{case_id}.npz"),
                            x=x.astype(np.float32), y=y.astype(np.float32),
                            fdi=tooth["fdi"], tmm=tmm, group=group, cdir=cdir)
        manifest.append(dict(case=case_id, fdi=tooth["fdi"], group=group,
                             cdir=cdir, crown_vox=int(y.sum())))
        print(f"  {case_id:28s} fdi {tooth['fdi']:>2}  crown voxels {int(y.sum()):6d}")
    json.dump(manifest, open(os.path.join(CFG.CACHE, "manifest.json"), "w"), indent=2)
    print(f"\ncached {len(manifest)} cases -> {CFG.CACHE}")
    return manifest

# Run once; comment out on later sessions (cache persists in /kaggle/working).
manifest = preprocess()

# %% 4. dataset / augmentation -------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader

def split_by_group(manifest, frac=CFG.SPLIT, seed=SEED):
    groups = sorted({m["group"] for m in manifest})
    rng = random.Random(seed); rng.shuffle(groups)
    n = len(groups); a = int(frac[0] * n); b = a + int(frac[1] * n)
    tr, va, te = set(groups[:a]), set(groups[a:b]), set(groups[b:])
    pick = lambda S: [m["case"] for m in manifest if m["group"] in S]
    return pick(tr), pick(va), pick(te)

class CrownVox(Dataset):
    def __init__(self, cases, train=False):
        self.cases, self.train = cases, train
    def __len__(self): return len(self.cases)
    def __getitem__(self, i):
        d = np.load(os.path.join(CFG.CACHE, f"{self.cases[i]}.npz"))
        x, y = d["x"].copy(), d["y"].copy()
        if self.train:                      # light, fit-preserving augmentation
            for ax in (1, 2, 3):
                if random.random() < 0.5:
                    x = np.flip(x, ax); y = np.flip(y, ax)
            k = random.randint(0, 3)        # rotate about the insertion (Z) axis
            x = np.rot90(x, k, axes=(1, 2)); y = np.rot90(y, k, axes=(1, 2))
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)))

# %% 5. 3D U-Net ---------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.ReLU(inplace=True),
        nn.Conv3d(co, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.ReLU(inplace=True))

class UNet3D(nn.Module):
    def __init__(self, in_ch=CFG.IN_CH, base=CFG.BASE_CH):
        super().__init__()
        b = base
        self.e1 = conv_block(in_ch, b);     self.e2 = conv_block(b, b * 2)
        self.e3 = conv_block(b * 2, b * 4)
        self.bott = conv_block(b * 4, b * 8)
        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, 2); self.d3 = conv_block(b * 8, b * 4)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, 2); self.d2 = conv_block(b * 4, b * 2)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, 2);     self.d1 = conv_block(b * 2, b)
        self.head = nn.Conv3d(b, 1, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        z = self.bott(self.pool(e3))
        d = self.d3(torch.cat([self.up3(z), e3], 1))
        d = self.d2(torch.cat([self.up2(d), e2], 1))
        d = self.d1(torch.cat([self.up1(d), e1], 1))
        return self.head(d)                 # logits (B,1,R,R,R)

# %% 6. train ------------------------------------------------------------------
def dice_bce(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((2, 3, 4)); den = p.sum((2, 3, 4)) + y.sum((2, 3, 4))
    dice = 1 - ((2 * inter + eps) / (den + eps)).mean()
    return dice + F.binary_cross_entropy_with_logits(logits, y)

@torch.no_grad()
def iou(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    i = (p * y).sum((2, 3, 4)); u = ((p + y) >= 1).float().sum((2, 3, 4))
    return ((i + 1) / (u + 1)).mean().item()

def train():
    tr, va, te = split_by_group(manifest)
    json.dump(dict(train=tr, val=va, test=te), open(f"{CFG.OUT}/splits.json", "w"), indent=2)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl_tr = DataLoader(CrownVox(tr, True), CFG.BATCH, shuffle=True, num_workers=2, drop_last=True)
    dl_va = DataLoader(CrownVox(va), CFG.BATCH)
    net = UNet3D().to(dev)
    opt = torch.optim.Adam(net.parameters(), CFG.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG.EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.AMP)
    best = -1
    for ep in range(CFG.EPOCHS):
        net.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=CFG.AMP):
                loss = dice_bce(net(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        net.eval(); vs = []
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                vs.append(iou(net(x), y))
        v = float(np.mean(vs)) if vs else 0.0
        print(f"epoch {ep:3d}  val IoU {v:.4f}")
        if v > best:
            best = v; torch.save(net.state_dict(), f"{CFG.CKPT}/best.pt")
    print(f"best val IoU {best:.4f}  -> {CFG.CKPT}/best.pt")
    return te

test_cases = train()

# %% 7. inference + mesh -------------------------------------------------------
from skimage import measure

@torch.no_grad()
def predict_mesh(net, case, dev):
    d = np.load(os.path.join(CFG.CACHE, f"{case}.npz"))
    x = torch.from_numpy(d["x"][None]).to(dev)
    prob = torch.sigmoid(net(x))[0, 0].cpu().numpy()
    pred = (prob > 0.5).astype(np.float32)
    # keep the largest connected component, then surface it
    lbl = measure.label(pred)
    if lbl.max() > 0:
        pred = (lbl == np.argmax(np.bincount(lbl.flat)[1:]) + 1).astype(np.float32)
    return pred, d["y"][0], d["tmm"]


def crown_world_mesh(vox, tmm):
    """Marching-cubes the predicted occupancy and place it back in world coords
    (voxel idx -> canonical mm -> world via forward ToothModelMatrix). Returns a
    trimesh that overlays the original scans / design.stl directly, or None."""
    if vox.sum() < 4:
        return None
    verts, faces, *_ = measure.marching_cubes(vox, level=0.5)
    verts_mm = (verts - CFG.RES // 2) * CFG.PITCH
    world = (np.c_[verts_mm, np.ones(len(verts_mm))] @ tmm)[:, :3]
    return trimesh.Trimesh(world, faces, process=False)


# --- deterministic margin-snap -------------------------------------------------
# Trim the predicted crown exactly at the preparation finish line so its cervical
# edge coincides with the real margin polyline. Done in the canonical frame
# (insertion axis = +Z), by subtracting a solid "plug" swept apically from the
# margin: crown - plug removes any flash apical to the finish line and cuts the
# crown along the true (3D, Z-varying) margin curve. Robust boolean via manifold3d;
# falls back to the raw crown if the boolean can't run on a given case.

def _margin_plug(margin_canon, depth=8.0):
    """Closed solid spanning from the margin curve `depth` mm apically (-Z)."""
    M = np.asarray(margin_canon, float)
    n = len(M)
    Mlow = M - np.array([0, 0, depth])
    V = np.vstack([M, Mlow, M.mean(0), Mlow.mean(0)])     # +2 cap centroids
    top_c, low_c = 2 * n, 2 * n + 1
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]           # side wall
        F += [[top_c, j, i]]                              # top cap (margin)
        F += [[low_c, n + i, n + j]]                      # bottom cap
    return trimesh.Trimesh(np.asarray(V), np.asarray(F), process=True)

def snap_to_margin(crown_world, margin_world, tmm):
    """Return the crown trimmed to the finish line (world coords), or the raw
    crown if anything fails. Never raises."""
    if margin_world is None or len(margin_world) < 8:
        return crown_world
    try:
        cc = crown_world.copy()
        cc.vertices = world_to_canonical(cc.vertices, tmm)
        plug = _margin_plug(world_to_canonical(margin_world, tmm))
        cut = trimesh.boolean.difference([cc, plug], engine="manifold")
        if cut is None or cut.is_empty or len(cut.vertices) < 10:
            return crown_world
        cut.vertices = (np.c_[cut.vertices, np.ones(len(cut.vertices))] @ tmm)[:, :3]
        return cut
    except Exception:
        return crown_world

# %% 8. clinical metrics: deviation heatmap + occlusal + margin -----------------
from scipy.spatial import cKDTree
import matplotlib.cm as _cm
import csv

def _sample(mesh, n=20000):
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts)

def _cmap(vals, vmin, vmax):
    t = np.clip((np.asarray(vals) - vmin) / (vmax - vmin + 1e-9), 0, 1)
    return (_cm.coolwarm(t)[:, :3] * 255).astype(np.uint8)

def vox_iou(pred, gt):
    i = (pred * gt).sum(); u = ((pred + gt) >= 1).sum()
    return float((i + 1) / (u + 1))

def surface_metrics(ai, design, n=20000):
    """Symmetric Hausdorff + RMS surface distance (mm) between two world meshes."""
    P, G = _sample(ai, n), _sample(design, n)
    dPG = cKDTree(G).query(P)[0]; dGP = cKDTree(P).query(G)[0]
    hd = max(dPG.max(), dGP.max())
    rms = math.sqrt((np.r_[dPG, dGP] ** 2).mean())
    return hd, rms

def deviation_heatmap(ai, design, out_ply, clip=0.5):
    """Per-vertex signed deviation of the AI crown from the exocad design.
    + = AI proud/over-contoured, - = deficient. Writes a colour-mapped PLY
    (open in MeshLab/CloudCompare) and returns (mean|dev|, %within TOL, %within 0.2mm)."""
    try:
        sd = trimesh.proximity.ProximityQuery(design).signed_distance(ai.vertices)
        dev = -np.asarray(sd)
    except Exception:                                   # design not watertight -> unsigned
        dev = cKDTree(_sample(design, 60000)).query(ai.vertices)[0]
    rgb = _cmap(dev, -clip, clip)
    m = ai.copy(); m.visual.vertex_colors = np.c_[rgb, np.full(len(rgb), 255, np.uint8)]
    m.export(out_ply)
    a = np.abs(dev)
    return float(a.mean()), float((a <= CFG.TOL_MM).mean() * 100), float((a <= 0.2).mean() * 100)

def occlusal_gap(crown, anta, c0):
    """Nearest distance from the crown surface to the opposing (antagonist) arch.
    Returns the minimum gap (occlusal clearance/contact, mm). Restricts the
    antagonist to the local ROI so far-away arch points don't pollute the query."""
    apts = _sample(anta, 60000)
    apts = apts[np.linalg.norm(apts - c0, axis=1) < CFG.ROI_MM]
    if len(apts) < 10:
        return float("nan")
    d = cKDTree(apts).query(_sample(crown, 20000))[0]
    return float(d.min())

def margin_discrepancy(crown, margin_pts):
    """Distance from the true preparation finish line (constructionInfo Margin,
    world coords) to the crown surface -> marginal discrepancy (mm)."""
    if margin_pts is None or len(margin_pts) < 3:
        return float("nan"), float("nan")
    d = cKDTree(_sample(crown, 40000)).query(margin_pts)[0]
    return float(d.mean()), float(d.max())

def _case_world_refs(case):
    """Reload the case's world-space design crown, antagonist arch and margin
    polyline (cheap; the raw scans are already in one world frame)."""
    cdir = str(np.load(os.path.join(CFG.CACHE, f"{case}.npz"))["cdir"])
    teeth = parse_construction_info(os.path.join(cdir, "constructionInfo"))
    design = _load(os.path.join(cdir, "design.stl"))
    dcen = design.vertices.mean(0)
    tooth = min(teeth, key=lambda t: np.inf if t["center"] is None
                else np.linalg.norm(t["center"] - dcen))
    arch = fdi_arch(tooth["fdi"])
    anta = _load(os.path.join(cdir, f"{'lower' if arch == 'upper' else 'upper'}.stl"))
    return design, anta, tooth["margin"], tooth["fdi"]

# %% 9. evaluate -> metrics.csv + STL + heatmaps -------------------------------
FIELDS = ["case", "fdi", "IoU", "Hausdorff_mm", "RMS_mm",
          "occlusal_gap_AI_mm", "occlusal_gap_design_mm",
          "margin_mean_mm", "margin_max_mm", "margin_fit_max_mm", "dev_within_tol_pct"]

def evaluate(test_cases):
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet3D().to(dev_); net.load_state_dict(torch.load(f"{CFG.CKPT}/best.pt", map_location=dev_)); net.eval()
    stl_dir = os.path.join(CFG.OUT, "stl"); heat_dir = os.path.join(CFG.OUT, "heatmap")
    os.makedirs(stl_dir, exist_ok=True); os.makedirs(heat_dir, exist_ok=True)
    rows = []
    for c in test_cases:
        pred, gt, tmm = predict_mesh(net, c, dev_)
        ai = crown_world_mesh(pred, tmm)
        if ai is None:
            continue
        ai.export(os.path.join(stl_dir, f"{c}_AI_crown.stl"))
        design, anta, margin, fdi = _case_world_refs(c)
        c0 = np.asarray(ai.vertices).mean(0)
        hd, rms = surface_metrics(ai, design)
        og_ai = occlusal_gap(ai, anta, c0)
        og_de = occlusal_gap(design, anta, c0)            # exocad's occlusal gap = reference
        mg_mean, mg_max = margin_discrepancy(ai, margin)  # model's raw marginal accuracy
        fitted = snap_to_margin(ai, margin, tmm)          # seatable crown
        fitted.export(os.path.join(stl_dir, f"{c}_AI_crown_fitted.stl"))
        _, mg_fit_max = margin_discrepancy(fitted, margin)   # seating residual (QC, ~0)
        _, pct_tol, _ = deviation_heatmap(ai, design, os.path.join(heat_dir, f"{c}_dev.ply"))
        rows.append(dict(case=c, fdi=fdi, IoU=round(vox_iou(pred, gt), 4),
                         Hausdorff_mm=round(hd, 4), RMS_mm=round(rms, 4),
                         occlusal_gap_AI_mm=round(og_ai, 4), occlusal_gap_design_mm=round(og_de, 4),
                         margin_mean_mm=round(mg_mean, 4), margin_max_mm=round(mg_max, 4),
                         margin_fit_max_mm=round(mg_fit_max, 4),
                         dev_within_tol_pct=round(pct_tol, 1)))
        print(f"  {c:26s} IoU {rows[-1]['IoU']:.3f} HD {hd:.2f} RMS {rms:.2f} "
              f"occl(AI/exo {og_ai:.2f}/{og_de:.2f}) margin {mg_mean:.2f}/{mg_max:.2f} "
              f"within{CFG.TOL_MM} {pct_tol:.0f}%")
    with open(f"{CFG.OUT}/metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    print("\n=== test summary (metrics.csv -> MedCalc) ===")
    for k in FIELDS[2:]:
        col = np.array([r[k] for r in rows], float); col = col[~np.isnan(col)]
        if len(col):
            print(f"  {k:22s} mean {col.mean():.4f}  RMS {math.sqrt((col**2).mean()):.4f}  SD {col.std(ddof=1):.4f}")
    occl_acc = np.abs(np.array([r["occlusal_gap_AI_mm"] - r["occlusal_gap_design_mm"] for r in rows], float))
    occl_acc = occl_acc[~np.isnan(occl_acc)]
    if len(occl_acc):
        print(f"  occlusal accuracy (|AI-exocad gap|): mean {occl_acc.mean():.4f} mm")
    print(f"\n  equivalence margin for write-up: RMS / margin deviation <= {CFG.TOL_MM} mm")
    print(f"  STL crowns : {stl_dir}/   (<case>_AI_crown.stl = morphology; "
          f"<case>_AI_crown_fitted.stl = margin-snapped, seatable)")
    print(f"  heatmaps   : {heat_dir}/   (PLY, signed deviation; open in MeshLab/CloudCompare)")
    print("  READING THE MARGIN COLUMNS:")
    print("   - margin_mean/max_mm  = the MODEL's raw marginal accuracy (report this;")
    print(f"     floored by voxel pitch {CFG.PITCH:.3f} mm, so use RES>=128 for the final run).")
    print("   - margin_fit_max_mm   = residual AFTER margin-snap (~0 = the delivered crown")
    print("     seats on the finish line). The _fitted.stl is the clinically usable crown.")
    print("  Occlusal & morphology metrics are likewise sharpened by RES>=128; "
          "prove the pipeline at 64, report at 128.")
    return rows

metrics = evaluate(test_cases)
