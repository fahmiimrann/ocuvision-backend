"""Inference entry point called by python-runner.js.

Usage:
    python python_predict.py --engine tabular|cnn --image <path> [--models-dir <dir>]

Output:
    Prints a single line that delimits the JSON payload so the Node parent
    can extract it with the same kind of marker-based regex used by
    matlab-runner.js:

        PY_JSON_BEGIN { ...json... } PY_JSON_END

    On error:

        PY_ERROR: <message>

The returned JSON has the same schema as the MATLAB pipeline so the
existing /api/analyze response normalisation works unchanged:

    {
        diagnosis, severity, score, notes,
        rawDiagnosis,
        features { ... display-name -> float ... },
        probs    { class -> float },
        engine: "python-tabular" | "python-cnn"
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _emit_json(obj: dict) -> None:
    sys.stdout.write("PY_JSON_BEGIN" + json.dumps(obj) + "PY_JSON_END\n")
    sys.stdout.flush()


def _emit_error(msg: str) -> None:
    sys.stdout.write(f"PY_ERROR:{msg}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Diagnosis copy (mirrors ocu_predict.m section 5 so UI severity logic stays
# consistent between engines).
# ---------------------------------------------------------------------------

def map_label_to_diagnosis(raw: str) -> dict:
    low = raw.lower()
    if "normal" in low or "healthy" in low:
        return {
            "diagnosis": "Healthy",
            "severity": "Healthy",
            "notes": "Fundus pattern is within baseline range. Continue routine annual screening.",
        }
    if "dr" in low or "diabetic" in low or "retinopathy" in low:
        return {
            "diagnosis": "Diabetic Retinopathy",
            "severity": "Critical",
            "notes": "Multiple microaneurysms and intraretinal hemorrhages detected. Expert validation recommended.",
        }
    if "early" in low and "glaucoma" in low:
        return {
            "diagnosis": "Early Glaucoma indicators detected",
            "severity": "Moderate",
            "notes": "Cup-to-disc ratio and vessel morphology should be reviewed by an ophthalmologist.",
        }
    if "glaucoma" in low:
        return {
            "diagnosis": "Glaucoma",
            "severity": "Critical",
            "notes": "Optic-disc cupping and neuroretinal-rim changes consistent with glaucoma. Refer for ophthalmic evaluation.",
        }
    if "amd" in low or "macular" in low:
        return {
            "diagnosis": "Age-related Macular Degeneration",
            "severity": "Moderate",
            "notes": "Drusen and macular pigmentary changes detected. Specialist follow-up recommended.",
        }
    if "cataract" in low:
        return {
            "diagnosis": "Cataract indicators detected",
            "severity": "Moderate",
            "notes": "Lens opacification reduces image clarity; consider slit-lamp evaluation.",
        }
    return {
        "diagnosis": raw,
        "severity": "Critical",
        "notes": "Abnormal fundus pattern detected. Expert validation recommended.",
    }


# ---------------------------------------------------------------------------
# Tabular engine (Option B)
# ---------------------------------------------------------------------------

def predict_tabular(image_path: str, models_dir: Path) -> dict:
    import joblib
    import pandas as pd
    import torch
    from feature_extraction import FEATURE_COLUMNS, extract_features

    bundle_path = models_dir / "tabular_clf.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Tabular classifier not found at {bundle_path}. "
            "Train it first with python train_tabular_classifier.py."
        )

    # Optional U-Nets: used to upgrade the disc/cup/vessel masks if their
    # weights are present. The classifier still uses CSV-trained features,
    # but the returned masks (for the UI overlay) come from the U-Nets when
    # they're available.
    vessel_unet = None
    optic_unet = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vessel_path = models_dir / "vessel_unet_model.pth"
    optic_path = models_dir / "optic_disc_cup_model.pth"
    if vessel_path.exists() or optic_path.exists():
        from unet import load_unet
        if vessel_path.exists():
            vessel_unet = load_unet(str(vessel_path), out_channels=1, device=device)
        if optic_path.exists():
            optic_unet = load_unet(str(optic_path), out_channels=2, device=device)

    feats = extract_features(
        image_path,
        vessel_unet=vessel_unet,
        optic_unet=optic_unet,
        device=device,
    )

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    classes = bundle["label_encoder_classes"]
    cols = bundle["feature_columns"]

    # Pass the DataFrame (not row.values) so LGBM/XGBoost preserve feature
    # names and don't emit "X does not have valid feature names" warnings.
    row = pd.DataFrame([{c: feats["features"].get(c, 0.0) for c in cols}],
                       columns=cols)
    pred_idx = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    probs: Dict[str, float] = {classes[i]: float(proba[i]) for i in range(len(classes))}
    raw_label = classes[pred_idx]

    diag = map_label_to_diagnosis(raw_label)
    confidence = float(np.max(proba))

    return {
        "engine": "python-tabular",
        "modelKind": bundle.get("model_kind", "tabular"),
        "rawDiagnosis": raw_label,
        "diagnosis": diag["diagnosis"],
        "severity": diag["severity"],
        "notes": diag["notes"],
        "score": round(confidence * 100, 1),
        "probs": probs,
        "features": feats["display"],
    }


# ---------------------------------------------------------------------------
# CNN engine (Option C)
# ---------------------------------------------------------------------------

def predict_cnn(image_path: str, models_dir: Path) -> dict:
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import models as tvm
    from torchvision import transforms

    bundle_path = models_dir / "cnn_clf.pth"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"CNN classifier not found at {bundle_path}. "
            "Train it first with python train_cnn_classifier.py --data-root ..."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(bundle_path, map_location=device)
    classes = bundle["classes"]
    img_size = bundle.get("img_size", 384)
    mean = bundle.get("normalize_mean", [0.485, 0.456, 0.406])
    std = bundle.get("normalize_std", [0.229, 0.224, 0.225])

    kind = bundle.get("model_kind", "resnet50")
    if kind == "resnet50":
        model = tvm.resnet50(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feats, len(classes)))
    else:
        raise RuntimeError(f"Unsupported CNN kind in bundle: {kind}")
    model.load_state_dict(bundle["state_dict"])
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(proba))
    raw_label = classes[pred_idx]
    diag = map_label_to_diagnosis(raw_label)

    # Optional: still extract the 19 display features so the UI report's
    # "Extracted Features" panel can show something. Falls back silently
    # if feature_extraction can't import torch already.
    features_for_ui: Dict[str, float] = {}
    try:
        from feature_extraction import extract_features
        vessel_unet = optic_unet = None
        vessel_path = models_dir / "vessel_unet_model.pth"
        optic_path = models_dir / "optic_disc_cup_model.pth"
        if vessel_path.exists() or optic_path.exists():
            from unet import load_unet
            if vessel_path.exists():
                vessel_unet = load_unet(str(vessel_path), out_channels=1, device=device)
            if optic_path.exists():
                optic_unet = load_unet(str(optic_path), out_channels=2, device=device)
        feats = extract_features(image_path, vessel_unet=vessel_unet,
                                 optic_unet=optic_unet, device=device)
        features_for_ui = feats["display"]
    except Exception as exc:
        features_for_ui = {"_warning": f"feature extraction skipped: {exc}"}

    return {
        "engine": "python-cnn",
        "modelKind": kind,
        "rawDiagnosis": raw_label,
        "diagnosis": diag["diagnosis"],
        "severity": diag["severity"],
        "notes": diag["notes"],
        "score": round(float(proba[pred_idx]) * 100, 1),
        "probs": {classes[i]: float(proba[i]) for i in range(len(classes))},
        "features": features_for_ui,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_models = here.parent / "models"

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["tabular", "cnn"], required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--models-dir", type=Path, default=default_models,
                    help="Folder that contains tabular_clf.pkl / cnn_clf.pth "
                         "(and optional vessel_unet_model.pth, optic_disc_cup_model.pth)")
    return ap.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # so 'unet', 'feature_extraction' resolve
    try:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"image not found: {args.image}")
        if args.engine == "tabular":
            result = predict_tabular(args.image, args.models_dir)
        else:
            result = predict_cnn(args.image, args.models_dir)
        _emit_json(result)
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit_error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
