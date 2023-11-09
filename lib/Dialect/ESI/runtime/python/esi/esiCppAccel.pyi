# Originally imported via:
#    pybind11-stubgen -o lib/Dialect/ESI/runtime/python/ esi.esiCppAccel
# Local modifications:
#    None yet. Though we're assuming that we will have some at some point.

from __future__ import annotations
import typing

__all__ = [
    'Accelerator', 'AppID', 'Design', 'Instance', 'MMIO', 'Manifest',
    'ModuleInfo', 'SysInfo', 'Type'
]


class Accelerator:

  def __init__(self, arg0: str, arg1: str) -> None:
    ...

  def get_service_mmio(self) -> MMIO:
    ...

  def sysinfo(self) -> SysInfo:
    ...


class AppID:

  def __repr__(self) -> str:
    ...

  @property
  def idx(self) -> typing.Any:
    ...

  @property
  def name(self) -> str:
    ...


class Design:

  @property
  def children(self) -> list[Instance]:
    ...

  @property
  def info(self) -> ModuleInfo | None:
    ...


class Instance(Design):

  @property
  def id(self) -> AppID:
    ...


class MMIO:

  def read(self, arg0: int) -> int:
    ...

  def write(self, arg0: int, arg1: int) -> None:
    ...


class Manifest:

  def __init__(self, arg0: str) -> None:
    ...

  def build_design(self, arg0: Accelerator) -> Design:
    ...

  @property
  def api_version(self) -> int:
    ...

  @property
  def type_table(self) -> list[Type]:
    ...


class ModuleInfo:

  def __repr__(self) -> str:
    ...

  @property
  def commit_hash(self) -> str | None:
    ...

  @property
  def name(self) -> str | None:
    ...

  @property
  def repo(self) -> str | None:
    ...

  @property
  def summary(self) -> str | None:
    ...

  @property
  def version(self) -> str | None:
    ...


class SysInfo:

  def esi_version(self) -> int:
    ...

  def json_manifest(self) -> str:
    ...


class Type:

  def __repr__(self) -> str:
    ...

  @property
  def id(self) -> str:
    ...
