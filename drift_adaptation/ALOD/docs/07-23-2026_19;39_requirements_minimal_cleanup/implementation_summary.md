# Requirements Minimal Cleanup

## Scope

The dependency files were reduced to the current ALOD runtime shape instead of
the copied MMDetection packaging layout.

## Kept

- `requirements.txt`: default pip-installable ALOD runtime entrypoint.
- `requirements/runtime.txt`: core runtime packages used by local ALOD +
  MMDetection code.
- `requirements/tests.txt`: stable test install target; ALOD tests use stdlib
  `unittest`, so it only includes runtime dependencies.
- `requirements/pal_embeddings.txt`: optional Google ViT embedding extraction
  dependency.

## Removed

- `requirements/build.txt`
- `requirements/docs.txt`
- `requirements/mminstall.txt`
- `requirements/optional.txt`
- `requirements/readthedocs.txt`

These were upstream MMDetection packaging/documentation extras and are not part
of the current ALOD user workflow.

## Notes

PyTorch, torchvision, and `mmcv-full` remain documented as separate framework
prerequisites because their wheels depend on the CUDA/PyTorch build. The local
MMDetection copy expects `mmcv-full>=1.3.17,<=1.5.0`; the previously validated
stack used PyTorch 1.10, torchvision 0.11, and mmcv-full 1.4.8.

`numpy==1.23.5` is pinned because the local MMDetection 2.x code still contains
deprecated aliases such as `np.int` and `np.bool`, which break on NumPy 1.24+.

## Verification

- Existing 24 unit tests passed.
- PAL runner dry-run passed.
- PPAL runner dry-run passed.
