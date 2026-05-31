# CholecT50 query images (visual cross-attention)

Place one reference image per instrument class, named after the label (underscores OK):

- `grasper.png`, `hook.png`, `bipolar.png`, `clipper.png`, `scissors.png`, `irrigator.png`
- `cystic-artery.png` or `cystic_artery.png`, etc.

Used by `visual_cross_attention_cholect50.py` (`--query-dir`, default: this folder).

If query files are missing, pass `--query-from-gt-crop` to use the GT bbox crop from the test frame as the query image.
