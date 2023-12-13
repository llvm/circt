# Originally imported via:
#    pybind11-stubgen -o lib/Dialect/ESI/runtime/python/ esi.esiCppAccel
# Local modifications:
#    None yet. Though we're assuming that we will have some at some point.

from __future__ import annotations
import typing

__all__ = [
    'Accelerator', 'AcceleratorConnection', 'AnyType', 'AppID', 'ArrayType',
    'BitVectorType', 'BitsType', 'BundlePort', 'BundleType', 'ChannelPort',
    'ChannelType', 'Direction', 'From', 'HWModule', 'Instance', 'IntegerType',
    'MMIO', 'Manifest', 'ModuleInfo', 'ReadChannelPort', 'SIntType',
    'StructType', 'SysInfo', 'To', 'Type', 'UIntType', 'WriteChannelPort'
]


class Accelerator(HWModule):
  pass


class AcceleratorConnection:

  def __init__(self, arg0: str, arg1: str) -> None:
    ...

  def get_service_mmio(self) -> MMIO:
    ...

  def sysinfo(self) -> SysInfo:
    ...


class AnyType(Type):
  pass


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


class ArrayType(Type):

  @property
  def element(self) -> Type:
    ...

  @property
  def size(self) -> int:
    ...


class BitVectorType(Type):

  @property
  def width(self) -> int:
    ...


class BitsType(BitVectorType):
  pass


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


class BundleType(Type):

  @property
  def channels(self) -> list[tuple[str, Direction, Type]]:
    ...


class ChannelPort:

  def connect(self) -> None:
    ...

  @property
  def type(self) -> Type:
    ...


class ChannelType(Type):

  @property
  def inner(self) -> Type:
    ...


class Direction:
  """
    Members:
    
      To
    
      From
    """
  From: typing.ClassVar[Direction]  # value = <Direction.From: 1>
  To: typing.ClassVar[Direction]  # value = <Direction.To: 0>
  __members__: typing.ClassVar[dict[
      str,
      Direction]]  # value = {'To': <Direction.To: 0>, 'From': <Direction.From: 1>}

  def __eq__(self, other: typing.Any) -> bool:
    ...

  def __getstate__(self) -> int:
    ...

  def __hash__(self) -> int:
    ...

  def __index__(self) -> int:
    ...

  def __init__(self, value: int) -> None:
    ...

  def __int__(self) -> int:
    ...

  def __ne__(self, other: typing.Any) -> bool:
    ...

  def __repr__(self) -> str:
    ...

  def __setstate__(self, state: int) -> None:
    ...

  def __str__(self) -> str:
    ...

  @property
  def name(self) -> str:
    ...

  @property
  def value(self) -> int:
    ...


class HWModule:

  @property
  def children(self) -> dict[AppID, Instance]:
    ...

  @property
  def info(self) -> ModuleInfo | None:
    ...

  @property
  def ports(self) -> dict[AppID, BundlePort]:
    ...


class Instance(HWModule):

  @property
  def id(self) -> AppID:
    ...


class IntegerType(BitVectorType):
  pass


class MMIO:

  def read(self, arg0: int) -> int:
    ...

  def write(self, arg0: int, arg1: int) -> None:
    ...


class Manifest:

  def __init__(self, arg0: str) -> None:
    ...

  def build_accelerator(self, arg0: AcceleratorConnection) -> Accelerator:
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


class SIntType(IntegerType):
  pass


class StructType(Type):

  @property
  def fields(self) -> list[tuple[str, Type]]:
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


class UIntType(IntegerType):
  pass


class WriteChannelPort(ChannelPort):

  def write(self, arg0: list[int]) -> None:
    ...


From: Direction  # value = <Direction.From: 1>
To: Direction  # value = <Direction.To: 0>
