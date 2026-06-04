# CholecT50 query images (visual cross-attention)

Place one reference image per instrument class, named after the label (underscores OK):

- `grasper.png`, `hook.png`, `bipolar.png`, `clipper.png`, `scissors.png`, `irrigator.png`
- `cystic-artery.png` or `cystic_artery.png`, etc.

Used by `visual_cross_attention_cholect50.py` (`--query-dir`, default: `assets/cholect50_query/`).

Place PNGs under `assets/cholect50_query/` (create the folder if missing).

If query files are missing, pass `--query-from-gt-crop` to use the GT bbox crop from the test frame as the query image.

# EndoVis 2017 query images (visual cross-attention)

Place one reference image per instrument slug (underscores), e.g.:

- `bipolar_forceps.png`, `prograsp_forceps.png`, `large_needle_driver.png`, `monopolar_curved_scissors.png`

Used by `visual_cross_attention_endovis2017.py` (`--query-dir`, default: `assets/endovis2017_query/`).

GT bboxes come from segmentation masks under `eval/endovis2017/val*/label/`. Use `--query-from-gt-crop` when query PNGs are absent.
