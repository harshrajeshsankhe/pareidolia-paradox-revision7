import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter, sobel

import torch
import torch.nn as nn
from torchvision import models

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


ROOT = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(ROOT, "train_images")
EVAL_DIR = os.path.join(ROOT, "eval_images")
META_PATH = os.path.join(ROOT, "test_metadata.csv")

OOF_PROBS = os.path.join(
    ROOT, "models", "revision7", "oof_probs.npy"
)

OOF_TRUE = os.path.join(
    ROOT, "models", "revision7", "oof_true.npy"
)

MODEL_PATH = os.path.join(
    ROOT, "models", "revision7", "final", "revision7_full.pt"
)

OUTPUT = os.path.join(ROOT, "submission.csv")

IMG_SIZE = 224
THRESHOLD = 0.377
SEED = 42

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def find_image(folder, image_id):
    for root, _, files in os.walk(folder):
        if image_id in files:
            return os.path.join(root, image_id)
    raise FileNotFoundError(image_id)


def make_features(path, azimuth):
    img = np.array(
        Image.open(path).convert("L"),
        dtype=np.float32
    ) / 255.0

    physics = rotate(
        img,
        angle=-float(azimuth),
        reshape=False,
        order=1,
        mode="reflect"
    )

    blur = gaussian_filter(physics, sigma=3)
    contrast = physics - blur

    cmin = contrast.min()
    cmax = contrast.max()

    if cmax > cmin:
        contrast = (contrast - cmin) / (cmax - cmin)

    gx = sobel(physics, axis=0)
    gy = sobel(physics, axis=1)

    grad = np.sqrt(gx * gx + gy * gy)

    gmax = grad.max()
    if gmax > 0:
        grad /= gmax

    rad = np.deg2rad(float(azimuth))

    sin_a = np.full_like(
        physics,
        np.sin(rad),
        dtype=np.float32
    )

    cos_a = np.full_like(
        physics,
        np.cos(rad),
        dtype=np.float32
    )

    channels = [
        physics,
        contrast,
        grad,
        sin_a,
        cos_a
    ]

    tensors = []

    for channel in channels:
        im = Image.fromarray(
            np.clip(channel * 255, 0, 255).astype(np.uint8)
        )

        im = im.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.BILINEAR
        )

        tensors.append(
            np.asarray(im, dtype=np.float32) / 255.0
        )

    return np.stack(tensors).astype(np.float32)


def build_model():
    model = models.resnet18(
        weights=None
    )

    old = model.conv1

    model.conv1 = nn.Conv2d(
        5,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False
    )

    model.fc = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(model.fc.in_features, 2)
    )

    return model


def load_model():
    model = build_model()

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    state = {
        k.replace("module.", "", 1): v
        for k, v in state.items()
    }

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


def azimuth_features(azimuth):
    azimuth = np.asarray(azimuth, dtype=float)
    rad = np.deg2rad(azimuth)

    return np.column_stack([
        np.sin(rad),
        np.cos(rad),
        np.sin(2 * rad),
        np.cos(2 * rad),
        (azimuth >= 270).astype(float),
        (azimuth >= 315).astype(float),
    ])


def main():

    print("=" * 70)
    print("PAREIDOLIA PARADOX — REVISION 7 HYBRID INFERENCE")
    print("=" * 70)

    print("Device:", DEVICE)

    # ------------------------------------------------------------
    # Load OOF predictions
    # ------------------------------------------------------------

    oof_probs = np.load(OOF_PROBS)
    oof_true = np.load(OOF_TRUE)

    if len(oof_probs) != len(oof_true):
        raise ValueError("OOF probability/label length mismatch")

    # ------------------------------------------------------------
    # Reconstruct exact Revision 7 meta-model
    # ------------------------------------------------------------

    train_meta = pd.read_csv(
        os.path.join(ROOT, "train_metadata.csv")
    )

    if len(train_meta) != len(oof_probs):
        raise ValueError("OOF length does not match training metadata")

    X_meta = np.column_stack([
        oof_probs,
        azimuth_features(
            train_meta["sun_azimuth_angle"].values
        )
    ])

    meta_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.3,
            max_iter=3000,
            random_state=SEED
        )
    )

    meta_model.fit(
        X_meta,
        oof_true
    )

    print("Meta-model reconstructed.")

    # ------------------------------------------------------------
    # Load test metadata
    # ------------------------------------------------------------

    test = pd.read_csv(META_PATH)

    if len(test) != 2000:
        raise ValueError(
            f"Expected 2,000 test rows, got {len(test)}"
        )

    required = {
        "image_id",
        "sun_azimuth_angle"
    }

    if not required.issubset(test.columns):
        raise ValueError(
            f"Missing columns: {required - set(test.columns)}"
        )

    # ------------------------------------------------------------
    # Build image index
    # ------------------------------------------------------------

    image_index = {}

    for root, _, files in os.walk(EVAL_DIR):
        for filename in files:
            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                image_index[filename] = os.path.join(
                    root,
                    filename
                )

    missing = [
        image_id
        for image_id in test["image_id"]
        if image_id not in image_index
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} evaluation images. "
            f"Example: {missing[:5]}"
        )

    print("Evaluation images:", len(image_index))

    # ------------------------------------------------------------
    # Load full Revision 7 visual model
    # ------------------------------------------------------------

    model = load_model()

    print("Visual model loaded.")

    # ------------------------------------------------------------
    # Visual inference
    # ------------------------------------------------------------

    visual_probs = []

    with torch.no_grad():

        for i, row in test.iterrows():

            path = image_index[row["image_id"]]

            x = make_features(
                path,
                row["sun_azimuth_angle"]
            )

            x = torch.from_numpy(
                x
            ).unsqueeze(0).to(DEVICE)

            logits = model(x)

            probability = torch.softmax(
                logits,
                dim=1
            )[0, 1].item()

            visual_probs.append(probability)

            if (i + 1) % 100 == 0:
                print(
                    f"Processed {i + 1}/2000"
                )

    visual_probs = np.asarray(
        visual_probs,
        dtype=np.float64
    )

    # ------------------------------------------------------------
    # Exact Revision 7 hybrid features
    # ------------------------------------------------------------

    X_test_meta = np.column_stack([
        visual_probs,
        azimuth_features(
            test["sun_azimuth_angle"].values
        )
    ])

    hybrid_probs = meta_model.predict_proba(
        X_test_meta
    )[:, 1]

    predictions = (
        hybrid_probs >= THRESHOLD
    ).astype(int)

    # ------------------------------------------------------------
    # Create submission
    # ------------------------------------------------------------

    submission = pd.DataFrame({
        "image_id": test["image_id"],
        "label": predictions
    })

    if len(submission) != 2000:
        raise ValueError("Submission must contain 2,000 rows")

    if not submission["image_id"].is_unique:
        raise ValueError("Duplicate image IDs")

    if submission["label"].isna().any():
        raise ValueError("Null labels")

    if not set(
        submission["label"].unique()
    ).issubset({0, 1}):
        raise ValueError("Invalid labels")

    submission.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("=" * 70)
    print("REVISION 7 HYBRID INFERENCE COMPLETE")
    print("=" * 70)
    print("Output:", OUTPUT)
    print("Rows:", len(submission))
    print(
        "Class 0:",
        int((predictions == 0).sum())
    )
    print(
        "Class 1:",
        int((predictions == 1).sum())
    )
    print(
        "Threshold:",
        THRESHOLD
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
