from __future__ import annotations

from .core import Value
from .circt import ir
from .index import index
from .rtg import rtg
from .integers import Integer
from .resources import Immediate

from typing import Union


class Memory(Value):

  def __init__(self, value: ir.Value):
    """
    For library internal usage only.
    """

    self._value = value

  def declare(size: Union[Integer, int], align: Union[Integer, int],
              address_width: int) -> Memory:
    """
    Declare a new memory with the specified parameters.

    Args:
      size: The size of the memory.
      align: The alignment of the memory in bytes.
      address_width: The width of the memory addresses in bits.
    """

    if isinstance(size, int):
      size = index.ConstantOp(size)
    if isinstance(align, int):
      align = index.ConstantOp(align)
    return rtg.MemoryAllocOp(rtg.MemoryType.get(address_width), size, align)

  def size(self) -> Integer:
    """
    Get the size of the memory in bytes.
    """

    return rtg.MemorySizeOp(self._value)

  def base_address(self) -> Immediate:
    """
    Get the base address of the memory as an immediate matching the memories
    address width.
    """

    return rtg.MemoryGetBaseAddressOp(self._value)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> ir.Type:
    return self._value.type

  def type(address_width: int) -> ir.Type:
    return rtg.MemoryType.get(address_width)
