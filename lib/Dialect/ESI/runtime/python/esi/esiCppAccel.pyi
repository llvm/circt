# Originally imported via:
#    pybind11-stubgen -o lib/Dialect/ESI/runtime/python/ esi.esiCppAccel
# Local modifications:
#    None yet. Though we're assuming that we will have some at some point.

from __future__ import annotations
import typing

__all__ = [
    'Accelerator', 'AppID', 'BundlePort', 'ChannelPort', 'Design', 'Instance',
    'MMIO', 'Manifest', 'ModuleInfo', 'ReadChannelPort', 'SysInfo', 'Type',
    'WriteChannelPort'
]


class Accelerator:

  def __init__(self, arg0: str, arg1: str) -> None:
    ...

  def get_service_mmio(self) -> MMIO:
    ...

  def sysinfo(self) -> SysInfo:
    ...


class AppID:

  def __eq__(self, arg0: AppID) -> bool:
    ...

  def __hash__(self) -> int:
    ...

  def __init__(self, name: str, idx: int | None = None) -> None:
    ...

  def __repr__(self) -> str:
    ...

  @property
  def idx(self) -> typing.Any:
    ...

  @property
  def name(self) -> str:
    ...


class BundlePort:

  def getRead(self, arg0: str) -> ReadChannelPort:
    ...

  def getWrite(self, arg0: str) -> WriteChannelPort:
    ...

  @property
  def channels(self) -> dict[str, ChannelPort]:
    ...

  @property
  def id(self) -> AppID:
    ...


class ChannelPort:

  def connect(self) -> None:
    ...


class Design:

  @property
  def children(self) -> dict[AppID, Instance]:
    ...

  @property
  def info(self) -> ModuleInfo | None:
    ...

  @property
  def ports(self) -> dict[AppID, BundlePort]:
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


class ReadChannelPort(ChannelPort):

  def read(self, arg0: int) -> list[int]:
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


class WriteChannelPort(ChannelPort):

  def write(self, arg0: list[int]) -> None:
    ...
