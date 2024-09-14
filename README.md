# IEEE TGRS Paper: Semantic Change Detection Based on Supervised Contrastive Learning for High-Resolution Remote Sensing Imagery - Official Code

The experimental setup for this paper uses the following software versions:
- PyTorch: 1.11.0
- segmentation_models_pytorch: 0.3.4

For more details, see `env.yaml`.

The experiment was conducted using an Nvidia GeForce RTX 3090 graphics card.

Taking the experiment on the Hi-UCD dataset as an example, first set the data path at line 14 in `datasets/datasets_catalog.py`. Then, execute `train_0.sh` to automatically train and infer.

The download address for the Hi-UCD data is https://github.com/Daisy-7/Hi-UCD-S.
