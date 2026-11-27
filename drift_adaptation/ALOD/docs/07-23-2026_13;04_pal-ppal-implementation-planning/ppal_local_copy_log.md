# PPAL Local Copy Log

## Scope

Phase 1 copied the PPAL reference implementation from `code_refs/PPAL` into editable ALOD-owned paths. The reference directory remains read-only and was not modified.

## Copied Paths

- `mmdet/` from `code_refs/PPAL/mmdet/`
- `configs/` from `code_refs/PPAL/configs/`
- `requirements/` from `code_refs/PPAL/requirements/`
- `tools/train.py` from `code_refs/PPAL/tools/train.py`
- `tools/test.py` from `code_refs/PPAL/tools/test.py`
- `setup.py` from `code_refs/PPAL/setup.py`
- `setup.cfg` from `code_refs/PPAL/setup.cfg`
- `MANIFEST.in` from `code_refs/PPAL/MANIFEST.in`
- `README.md` from `code_refs/PPAL/README.md`
- `model-index.yml` from `code_refs/PPAL/model-index.yml`

`tools/` was created with only `train.py` and `test.py`; other PPAL tools were intentionally not copied in this phase.

## Intentionally Excluded

- `code_refs/PPAL/mmdet.egg-info/`
- Any `__pycache__/` directories
- Any `*.pyc` files
- `code_refs/PPAL/al_configs/`
- `code_refs/PPAL/tests/`
- `code_refs/PPAL/resources/`
- Root files outside the requested write scope, including `requirements.txt`, `LICENSE`, `CITATION.cff`, `pytest.ini`, and the paper PDF

The copied `setup.py` still has the original PPAL/MMDetection behavior. It reads local `README.md` successfully. Its `extras_require["all"]` calls `parse_requirements("requirements.txt")`; because the original parser returns an empty list when that file is absent, this does not block setup metadata generation, but the `all` extra is not meaningful unless a root `requirements.txt` is added later.

## Path And Import Concerns

- Running Python from the ALOD root resolves `mmdet` to `D:\DataDrift\GitHub\ALOD\mmdet\__init__.py` via `importlib.util.find_spec`.
- Direct `import mmdet` imports `mmcv` in `mmdet/__init__.py`. In this non-interactive Codex shell, both direct `envs\alod\python.exe` and `conda.exe run -n alod` timed out while importing `mmcv`, before `mmdet` could finish importing. The user previously validated `mmcv 1.4.8` in an activated `alod` PowerShell, so this appears to be an environment invocation issue in the tool shell rather than a copied-source path issue.
- `python -m py_compile tools/train.py tools/test.py` passed using `C:\Users\Yeseongjin\anaconda3\envs\alod\python.exe`.

## Validation Commands

```powershell
Get-ChildItem -Force mmdet,tools,configs,requirements | Select-Object FullName, PSIsContainer
Get-ChildItem -Recurse -Directory mmdet,configs,requirements | Where-Object { $_.Name -eq '__pycache__' -or $_.Name -eq 'mmdet.egg-info' } | Select-Object -ExpandProperty FullName
& 'C:\Users\Yeseongjin\anaconda3\envs\alod\python.exe' -m py_compile tools\train.py tools\test.py
& 'C:\Users\Yeseongjin\anaconda3\envs\alod\python.exe' -c "import importlib.util, pathlib; spec=importlib.util.find_spec('mmdet'); print(pathlib.Path(spec.origin).resolve())"
& 'C:\Users\Yeseongjin\anaconda3\envs\alod\python.exe' -c "import mmcv; print(mmcv.__version__)"
& 'C:\Users\Yeseongjin\anaconda3\Scripts\conda.exe' run -n alod python -c "import mmcv; print(mmcv.__version__)"
```

Results:

- Directory inventory completed.
- Generated cache/package directories were not present after cleanup.
- `py_compile` passed.
- `find_spec("mmdet")` returned `D:\DataDrift\GitHub\ALOD\mmdet\__init__.py`.
- Both `mmcv` import commands timed out in the Codex tool shell.
