from typing import Any, Mapping
from omegaconf import DictConfig, OmegaConf
from .registry import Registry


def to_dict(cfg: Any) -> dict:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    return cfg


def build_from_cfg(cfg: Mapping, registry: Registry) -> Any:
    cfg_dict = to_dict(cfg) or {}
    typ = cfg_dict.get("type")
    params = cfg_dict.get("params", {})
    if typ is None:
        raise ValueError(f"Config missing 'type': {cfg_dict}")
    cls = registry.get(typ)
    return cls(**params)
