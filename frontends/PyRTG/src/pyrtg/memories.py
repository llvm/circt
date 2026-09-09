from __future__ import annotations

from .core import Value, Type
from .base import ir
from .index import index
from .rtg import rtg
from .integers import Integer
from .immediates import Immediate
from .support import _FromCirctType

from typing import Union


class MemoryBlock(Value):

  def __init__(self, value: ir.Value, type: MemoryBlockType = None):
    """
    For library internal usage only.
    """

    self._value = value
    self._type = type

  def declare(base_address: int, end_address: int,
              address_width: int) -> MemoryBlock:
    """
    Declare a new memory block with the specified parameters.

    Args:
      base_address: The first valid address of the memory
      end_address: The last valid address of the memory
      address_width: The width of the memory block addresses in bits.
    """

    op = ir.Operation.create(
        "rtg.isa.memory_block_declare",
        attributes={
            "baseAddress": ir.IntegerAttr.get(
                ir.IntegerType.get_signless(address_width), base_address),
            "endAddress": ir.IntegerAttr.get(
                ir.IntegerType.get_signless(address_width), end_address),
        },
        results=[rtg.MemoryBlockType.get(address_width)],
        regions=0)
    return MemoryBlock(op.result, MemoryBlockType(address_width))

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    if self._type is not None:
      return self._type
    return _FromCirctType(self._value.type)


class MemoryBlockType(Type):
  """
  Represents the type of memory blocks.

  Fields:
    address_width: int
  """

  def __init__(self, address_width: int):
    self.address_width = address_width

  def __eq__(self, other) -> bool:
    return isinstance(
        other, MemoryBlockType) and self.address_width == other.address_width

  def _codegen(self):
    return rtg.MemoryBlockType.get(self.address_width)

  def _wrap(self, value: ir.Value) -> MemoryBlock:
    return MemoryBlock(value, self)


class Memory(Value):

  def __init__(self, value: ir.Value, type: MemoryType = None):
    """
    For library internal usage only.
    """

    self._value = value
    self._type = type

  def alloc(mem_block: MemoryBlock, size: Union[Integer, int],
            align: Union[Integer, int]) -> Memory:
    """
    Allocate a new memory from a memory block with the specified parameters.

    Args:
      size: The size of the memory in bytes.
      align: The alignment of the memory in bytes.
    """

    if isinstance(size, int):
      size = ir.Operation.create(
          "index.constant",
          attributes={"value": ir.IntegerAttr.get(ir.IndexType.get(), size)},
          results=[ir.IndexType.get()],
          regions=0).result
    if isinstance(align, int):
      align = ir.Operation.create(
          "index.constant",
          attributes={"value": ir.IntegerAttr.get(ir.IndexType.get(), align)},
          results=[ir.IndexType.get()],
          regions=0).result
    op = ir.Operation.create(
        "rtg.isa.memory_alloc",
        operands=[mem_block._get_ssa_value(),
                  size._get_ssa_value() if isinstance(size, Integer) else size,
                  align._get_ssa_value() if isinstance(align, Integer) else align],
        results=[rtg.MemoryType.get(mem_block.get_type().address_width)],
        regions=0)
    return Memory(op.result, MemoryType(mem_block.get_type().address_width))

  def size(self) -> Integer:
    """
    Get the size of the memory in bytes.
    """

    return Integer(ir.Operation.create(
        "rtg.isa.memory_size",
        operands=[self._value],
        results=[ir.IndexType.get()],
        regions=0).result)

  def base_address(self) -> Immediate:
    """
    Get the base address of the memory as an immediate matching the memories
    address width.
    """

    return Immediate(self.get_type().address_width,
                     ir.Operation.create(
                         "rtg.isa.memory_base_address",
                         operands=[self._value],
                         results=[ir.IntegerType.get_signless(
                             self.get_type().address_width)],
                         regions=0).result)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    if self._type is not None:
      return self._type
    return _FromCirctType(self._value.type)


class MemoryType(Type):
  """
  Represents the type of memory allocations.

  Fields:
    address_width: int
  """

  def __init__(self, address_width: int):
    self.address_width = address_width

  def __eq__(self, other) -> bool:
    return isinstance(other,
                      MemoryType) and self.address_width == other.address_width

  def _codegen(self):
    return rtg.MemoryType.get(self.address_width)

  def _wrap(self, value: ir.Value) -> Memory:
    return Memory(value, self)
