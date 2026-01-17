"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from typing import Generator, Union
from os import PathLike
import shutil
from pathlib import Path


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
