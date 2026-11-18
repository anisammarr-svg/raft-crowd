"""Build the standalone Colab notebook and source archive from canonical sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    paths = [*sorted((ROOT / "src/raftcrowd").glob("*.py")),
             *sorted((ROOT / "tests").glob("test_raftcrowd_*.py")),
             ROOT / "scripts/raftcrowd.py", ROOT / "configs/raftcrowd.json",
             ROOT / "configs/raftcrowd_manifest.example.csv", ROOT / "requirements-raftcrowd.txt",
             ROOT / "RAFTCROWD_README.md", ROOT / "docs/raftcrowd-method.md",
             ROOT / "plan/raftcrowd-experiment-protocol.md", ROOT / "tables/raftcrowd-table-schema.md",
             ROOT / "figures/raftcrowd-data-manifest.md"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): p.read_bytes().decode("utf-8") for p in paths}


def cell(kind, text):
    result = {"id": hashlib.sha256((kind + "\n" + text).encode()).hexdigest()[:16],
              "cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(output=None):
    output = Path(output) if output else ROOT / "output/jupyter-notebook/RAFTCrowdFlow_from_scratch.ipynb"
    files = source_files()
    bundle = {name: {"sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text} for name, text in files.items()}
    notebook = json.loads((ROOT / "scripts/templates/raftcrowd_experiment.ipynb").read_text(encoding="utf-8"))
    notebook["metadata"].update({"colab": {"name": output.name}, "accelerator": "GPU",
                                  "raftcrowd_source_sha256": {k: v["sha256"] for k, v in bundle.items()}})
    notebook["cells"] = [
        cell("markdown", """# RAFT-CrowdFlow from the beginning

This notebook contains the full starting implementation. It does **not** reproduce the rejected paper's tables. Defaults implement motion-only CFI; externally supplied pedestrian density is optional and is never inferred from blurred flow. No crowd-crush prediction or real-time performance is established.

Select **Runtime > Change runtime type > GPU** in Colab. Run cells in order. No original notebook or private manuscript is required. All embedded source files are checksum-pinned and regenerated from the local repository builder.
"""),
        cell("code", """from pathlib import Path
import hashlib, json, os, sys, subprocess

ROOT = Path.cwd() / 'raftcrowd_start'
ROOT.mkdir(exist_ok=True)
""" + "FILES = " + repr(bundle) + "\n" + """
for name, item in FILES.items():
    target = ROOT / name
    data = item['text'].encode('utf-8')
    assert hashlib.sha256(data).hexdigest() == item['sha256'], name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise RuntimeError(f'Existing source differs: {target}; choose a fresh ROOT')
    else:
        target.write_bytes(data)
sys.path.insert(0, str(ROOT / 'src'))
os.environ['PYTHONPATH'] = str(ROOT / 'src')
print(f'Checked {len(FILES)} source files in {ROOT}')
"""),
        cell("markdown", """## Environment and synthetic checks

The following cell installs the declared dependencies in your runtime. Colab normally provides a compatible PyTorch/Torchvision GPU pair; check the displayed versions. These broad dependency bounds are not a tested lockfile. Synthetic checks do not run RAFT or establish dataset performance.
"""),
        cell("code", """subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-raftcrowd.txt'), 'pytest'])
import torch, torchvision, numpy as np
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('CUDA:', torch.cuda.is_available())
subprocess.check_call([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests')], cwd=ROOT)
"""),
        cell("markdown", """## Supply benchmark frames and actual labels

Prepare a ZIP with `manifest.csv` at its root and frame/label folders. See the embedded `RAFTCROWD_README.md` for exact formats. A row example is `val001,val,camera_val,val001/frames,val001/labels.npy,`.

Use separate train/validation/test source groups. Train must be normal-only. Labels are binary `.npy` vectors of length N, one for each chronologically named input frame. Both classes must occur among scored frames in the pooled validation and pooled test sets. Do not invent labels. Run separate manifests/caches for each dataset.

The cell below uploads and extracts a local ZIP. Alternatively skip it and set `MANIFEST` to a file you have placed in this runtime or mounted storage. No dataset is downloaded by the code. For video inputs, first use `python scripts/raftcrowd.py video-frames --video ... --out ...`, then align true annotations explicitly.
"""),
        cell("code", """# Colab upload: alternatively skip and set MANIFEST in the next cell.
from google.colab import files
import stat, zipfile
uploaded = files.upload()
zip_names = [name for name in uploaded if name.lower().endswith('.zip')]
if len(zip_names) != 1:
    raise ValueError('Upload one dataset ZIP with manifest.csv at its root')
DATA = ROOT / 'input_data'
DATA.mkdir(exist_ok=False)
with zipfile.ZipFile(zip_names[0]) as archive:
    for member in archive.infolist():
        target = (DATA / member.filename).resolve()
        if not target.is_relative_to(DATA.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError('Archive has an unsafe path or link')
    archive.extractall(DATA)
MANIFEST = DATA / 'manifest.csv'
assert MANIFEST.is_file(), 'manifest.csv must be at the ZIP root'
"""),
        cell("markdown", """## Freeze configuration before extraction

Defaults: RAFT-large C_T_V1, 360 x 640, 12 updates, FP32, four motion groups, fixed equal weights, training-only 99th-percentile cue scaling, top-1% mean frame score. The output flow is cropped after padding. No temporal region warping is implemented.

Optional density: set `density_mode='external'` and supply a density folder in **every** manifest row. Maps must be independently produced pedestrian-density arrays, already aligned to the resized source frame, named with its stem. Record their estimator/checkpoint/units separately. The default has no pressure term.

Caching several numeric maps for every frame pair can use substantial storage. Start with a separate small pilot manifest to check inputs and resources. This implementation has no extraction resume; keep failed outputs for inspection and use a new run directory.
"""),
        cell("code", """from datetime import datetime, timezone
from raftcrowd.io import load_config, load_manifest

# For mounted/pre-existing data, set MANIFEST = Path('/path/to/manifest.csv') here.
config = load_config(ROOT / 'configs/raftcrowd.json')
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
# Make deliberate prospective changes here, before running this experiment.
RUN = ROOT / 'results' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
RUN.mkdir(parents=True, exist_ok=False)
CONFIG = RUN / 'config.json'
CONFIG.write_text(json.dumps(config, indent=2), encoding='utf-8')
records = load_manifest(MANIFEST, load_config(CONFIG))
print([(r['id'], r['split'], len(r['frames'])) for r in records])
print('Run:', RUN, 'Device:', config['device'])
"""),
        cell("markdown", """## Extract frozen RAFT flow and numeric cues

This is the costly step. Pretrained weights download on first use. Every panel, baseline and CFI cue uses the same cached field; a model is never silently switched between panels. Original frames and labels are hashed. No benchmark metrics or thresholds are selected here.
"""),
        cell("code", """def run_cli(*args):
    subprocess.check_call([sys.executable, str(ROOT / 'scripts/raftcrowd.py'), *map(str, args)], cwd=ROOT)

CACHE = RUN / 'cache'
run_cli('extract', '--manifest', MANIFEST, '--config', CONFIG, '--out', CACHE)
"""),
        cell("markdown", """## Calibrate using training and validation only

Normal training frames fit cue scales. Validation labels choose each method's F1 threshold. The model JSON saves these fixed choices and validation predictions. A no-segmentation control has separately fitted training variability scales. Test labels are not used for selection.
"""),
        cell("code", """MODEL = RUN / 'model.json'
run_cli('calibrate', '--cache', CACHE, '--out', MODEL)
"""),
        cell("markdown", """## Evaluate the held-out test set

The output includes full active CFI, magnitude-only, individual cue, leave-one-cue-out and no-segmentation controls on identical pairs. It reports pooled frame ROC-AUC, interpolated EER and fixed-threshold precision/recall/F1, plus per-video metrics and per-frame scores. Values are fractions, not percentages. Pair `(t,t+1)` is evaluated at label `t+1`; frame zero is excluded.

Do not tune this configuration after reading test results and describe the same test set as untouched. These anomaly metrics do not establish early warning, prevention or spatial localization of crowd crushes.
"""),
        cell("code", """EVALUATION = RUN / 'evaluation'
run_cli('evaluate', '--cache', CACHE, '--model', MODEL, '--out', EVALUATION)
report = json.loads((EVALUATION / 'metrics.json').read_text())
for name, item in report['methods'].items():
    print(name, item['pooled'])
"""),
        cell("markdown", """## Six panels from one flow field

Panels show source RGB, flow color, actual CFI score, magnitude, compression and motion groups. Colors represent normalized motion cues, not verified safety risk. The preview requires the unchanged original frame files as well as the cache.
"""),
        cell("code", """from IPython.display import Image, display
TEST_VIDEO = next(r['id'] for r in records if r['split'] == 'test')
PANEL = RUN / 'six_panel.png'
run_cli('visualize', '--cache', CACHE, '--model', MODEL, '--video-id', TEST_VIDEO,
        '--frame-index', 1, '--out', PANEL)
display(Image(filename=str(PANEL), width=1000))
"""),
        cell("markdown", """## Preserve evidence

Keep the original manifest, frame/annotation identities, density provenance if used, source pins, cache receipts, model JSON, prediction CSV and metric JSON. Extraction timing is diagnostic, has no warm-up and excludes parts of the pipeline; it is not a Jetson or end-to-end deployment benchmark.

The source package includes method choices, a prospective protocol and table/figure provenance schemas. Historical scores from the rejected paper remain unverified. No new claim of novelty, improvement or safety effectiveness follows from the presence of this code.
"""),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    archive_path = ROOT / "output/raftcrowd_starting_code.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text.encode())
        # Include the builder/template as well so edits regenerate the notebook.
        for path in [Path(__file__), ROOT / "scripts/check_raftcrowd_notebook.py",
                     ROOT / "scripts/templates/raftcrowd_experiment.ipynb"]:
            archive.write(path, str(path.relative_to(ROOT)))
        archive.write(output, "output/jupyter-notebook/" + output.name)
    return output, archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    for path in build(args.out):
        print(path)
