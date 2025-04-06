from __future__ import annotations

from .core import Value
from .base import ir
from .index import index
from .rtg import rtg
from .integers import Integer
from .resources import Immediate

from typing import Union


class MemoryBlock(Value):

  def __init__(self, value: ir.Value):
    """
    For library internal usage only.
    """

    self._value = value

  def declare(base_address: int, end_address: int,
              address_width: int) -> MemoryBlock:
    """
    Declare a new memory block with the specified parameters.

    Args:
      base_address: The first valid address of the memory
      end_address: The last valid address of the memory
      address_width: The width of the memory block addresses in bits.
    """

    return rtg.MemoryBlockDeclareOp(
        rtg.MemoryBlockType.get(address_width),
        ir.IntegerAttr.get(ir.IntegerType.get_signless(address_width),
                           base_address),
        ir.IntegerAttr.get(ir.IntegerType.get_signless(address_width),
                           end_address))

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> ir.Type:
    return self._value.type

  def type(address_width: int) -> ir.Type:
    return rtg.MemoryBlockType.get(address_width)


class Memory(Value):

  def __init__(self, value: ir.Value):
    """
    For library internal usage only.
    """

    self._value = value

  def alloc(mem_block: MemoryBlock, size: Union[Integer, int],
            align: Union[Integer, int]) -> Memory:
    """
    Allocate a new memory from a memory block with the specified parameters.

    Args:
      size: The size of the memory in bytes.
      align: The alignment of the memory in bytes.
    """

    if isinstance(size, int):
      size = index.ConstantOp(size)
    if isinstance(align, int):
      align = index.ConstantOp(align)
    return rtg.MemoryAllocOp(mem_block, size, align)

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

    return rtg.MemoryBaseAddressOp(self._value)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> ir.Type:
    return self._value.type

  def type(address_width: int) -> ir.Type:
    return rtg.MemoryType.get(address_width)
