"""
Module description:

"""


from typing import List, Generator, Union
from os import PathLike
import re
import shutil
from pathlib import Path

regexp = re.compile(r'[\D][\w-]+\.[\w-]+')


_CTX = {}

def set_config_folder(path):
    _CTX["config_folder"] = path

def get_config_folder():
    return _CTX["config_folder"]


def check_dir(
    path: Union[str, PathLike[str]],
    replace: bool = False
) -> str:
    _path = Path(path)

    if _path.exists():
        if replace:
            shutil.rmtree(_path, ignore_errors=True)
            _path.mkdir(parents=True, exist_ok=True)
    else:
        _path.mkdir(parents=True, exist_ok=True)

    return path_absolute(_path)


def list_dir(
    path: Union[str, PathLike[str]]
) -> Generator[str, None, None]:
    _dirs = [d for d in Path(path).iterdir()]
    for d in _dirs:
        yield str(d)


def parent_dir(
    path: Union[str, PathLike[str]]
) -> str:
    return str(Path(path).parent)


def is_dir(
    path: Union[str, PathLike[str]]
) -> bool:
    return Path(path).is_dir()


def check_path(
    path: Union[str, PathLike[str]]
) -> bool:
    return Path(path).exists()


def is_file(
    path: Union[str, PathLike[str]]
) -> bool:
    return Path(path).is_file()


def path_joiner(
    *args: Union[str, PathLike[str]]
) -> str:
    return str(Path(*args))


def path_absolute(
    path: Union[str, PathLike[str]]
) -> str:
    return str(Path(path).resolve())


def path_relative(
    path: Union[str, PathLike[str]],
    start: Union[str, PathLike[str]]
) -> str:
    return str(Path(path).relative_to(start))


def path_resolver(local_path: str, values: Union[str, List[str]] = "") -> str:
    if local_path.startswith((".", "..")) or regexp.search(local_path):
        local_path = path_absolute(
            path_joiner(get_config_folder(), local_path)
        )
    if values:
        try:
            local_path = local_path.format(values)
        except (KeyError, IndexError, ValueError, TypeError, AttributeError):
            pass
    return local_path


def file_ext(
    path: Union[str, PathLike[str]],
) -> str:
    return str(Path(path).suffix)


def file_name(
    path: Union[str, PathLike[str]],
) -> str:
    return str(Path(path).stem)
