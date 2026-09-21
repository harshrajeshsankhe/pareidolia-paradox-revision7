# The Pareidolia Paradox — Revision 7

## Final Competition Solution

Binary classification of 256x256 grayscale lunar surface crops into Depth (Class 0) and Rise (Class 1).

Metric: Balanced Accuracy.

## Physics Normalization

Each image is rotated by the negative sun_azimuth_angle to normalize illumination direction before feature extraction.

## Five-Channel Visual Model

- Physics-normalized grayscale
- Local contrast
- Sobel gradient magnitude
- sin(azimuth)
- cos(azimuth)

The visual backbone is a pretrained ResNet18 modified for five-channel input.

Training uses class-weighted cross-entropy, AdamW, horizontal flipping and small ±7 degree rotations.

## Meta Model

Out-of-fold visual predictions are combined with azimuth-derived features: sin(azimuth), cos(azimuth), sin(2*azimuth), cos(2*azimuth), azimuth >= 270 degrees, and azimuth >= 315 degrees.

A standardized logistic-regression meta-model is used.

Final threshold: 0.404

## Validation

Normal cross-fitted Balanced Accuracy: 0.78350

Shift-aware weighted Balanced Accuracy: 0.69894

The shift-aware evaluation accounts for the difference between training and evaluation azimuth distributions.

## Submission

submission.csv contains exactly 2,000 predictions with columns image_id,label.

## Reproduction

Install dependencies:

    pip install -r requirements.txt

The validated final artifacts are stored under models/revision7/ and outputs/submissions/.

## Author

Harsh Sankhe

Computer Engineering | Cyber Security | AI & Data Analytics | Computer Vision
