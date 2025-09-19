from typing import Any, Dict, Optional, Callable


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._obj_map: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Any], Any]:
        def _wrap(obj: Any):
            key = name or getattr(obj, "__name__", None) or str(obj)
            if key in self._obj_map:
                raise KeyError(f"{self._name} already has key: {key}")
            self._obj_map[key] = obj
            return obj

        return _wrap

    def get(self, name: str) -> Any:
        if name not in self._obj_map:
            raise KeyError(
                f"{name} not found in registry {self._name}. Available: {list(self._obj_map)}"
            )
        return self._obj_map[name]

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def keys(self):
        return list(self._obj_map.keys())
