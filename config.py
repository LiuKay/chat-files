import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from utils import merge

logger = logging.getLogger(__name__)
FILENAME = "config.yml"


def _get_config(path: Union[Path, str]) -> Dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / FILENAME
    with open(path) as f:
        return yaml.safe_load(f)


def get_config(path: Optional[Union[Path, str]] = None, merge_default: bool = False) -> Dict[str, Any]:
    default_config = _get_config(Path(__file__).parent / "config")
    if path is None:
        path = Path() / FILENAME
        if not path.is_file():
            return default_config
    logging.info(f"load config file path:{path}")
    user_config = _get_config(path)
    if merge_default:
        return merge(default_config, user_config)
    return user_config
