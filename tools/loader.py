"""Shared model loading for the dev tools.

A model source is either:
  - a path to an .xml file, or
  - "package.module:function", where function() returns an MjSpec or MjModel.

The default source is "model.generator:build", so the generator just needs to
expose a zero-argument build() that returns an MjSpec (load your config inside it).
"""

import importlib
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = ROOT / "model" / "generated"
DEFAULT_SOURCE = "model.generator:build"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(source: str = DEFAULT_SOURCE, reload_modules: bool = False) -> mujoco.MjModel:
    """Build and compile the model. Writes the XML to model/generated/ when built from a spec."""
    if source.endswith(".xml"):
        return mujoco.MjModel.from_xml_path(source)

    module_name, _, func_name = source.partition(":")
    func_name = func_name or "build"

    if reload_modules:
        _reload_project_modules()
    module = importlib.import_module(module_name)
    result = getattr(module, func_name)()

    if isinstance(result, mujoco.MjModel):
        return result
    if isinstance(result, mujoco.MjSpec):
        model = result.compile()
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        (GENERATED_DIR / "robot.xml").write_text(result.to_xml())
        return model
    raise TypeError(f"{source} returned {type(result).__name__}, expected MjSpec or MjModel")


def _reload_project_modules() -> None:
    """Drop cached project modules (anything under ROOT, except tools/) so edits are picked up."""
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if not path:
            continue
        p = Path(path).resolve()
        if ROOT in p.parents and (ROOT / "tools") not in p.parents and ROOT / ".venv" not in p.parents:
            del sys.modules[name]
