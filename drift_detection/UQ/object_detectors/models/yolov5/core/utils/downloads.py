import logging
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def curl_download(url, filename, *, silent=False):
    silent_option = "sS" if silent else ""
    proc = subprocess.run(
        ["curl", "-#", f"-{silent_option}L", url, "--output", str(filename), "--retry", "9", "-C", "-"]
    )
    return proc.returncode == 0


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    from models.yolov5.core.utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg
    except Exception as e:
        if file.exists():
            file.unlink()
        LOGGER.info(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:
            if file.exists():
                file.unlink()
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info("")


def _github_assets(repository, version="latest"):
    if version != "latest":
        version = f"tags/{version}"
    response = requests.get(f"https://api.github.com/repos/{repository}/releases/{version}").json()
    return response["tag_name"], [x["name"] for x in response["assets"]]


def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
    from models.yolov5.core.utils.general import LOGGER

    file = Path(str(file).strip().replace("'", ""))
    if file.exists():
        return str(file)

    name = Path(urllib.parse.unquote(str(file))).name
    if str(file).startswith(("http:/", "https:/")):
        url = str(file).replace(":/", "://")
        target = Path(name.split("?")[0])
        if target.is_file():
            LOGGER.info(f"Found {url} locally at {target}")
        else:
            safe_download(file=target, url=url, min_bytes=1e5)
        return str(target)

    assets = [f"yolov5{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "6", "-cls", "-seg")]
    try:
        tag, assets = _github_assets(repo, release)
    except Exception:
        try:
            tag, assets = _github_assets(repo)
        except Exception:
            tag = release

    if name in assets:
        file.parent.mkdir(parents=True, exist_ok=True)
        safe_download(
            file,
            url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
            min_bytes=1e5,
            error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
        )
    return str(file)
