#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .support import get_user_loc

import circt.support as support

import mlir.ir as ir

from typing import Union
import re


class Value:

  @staticmethod
  def get(value, type=None):
    from .pycde_types import PyCDEType

    if value is None or isinstance(value, Value):
      return value
    resvalue = support.get_value(value)
    if resvalue is None:
      return None

    if type is None:
      type = resvalue.type
    type = PyCDEType(type)
    return type.get_value(value)

  _reg_name = re.compile(r"^(.*)__reg(\d+)$")

  def reg(self, clk, rst=None, name=None):
    from .dialects import seq
    if name is None:
      name = self.name
      if name is not None:
        m = Value._reg_name.match(name)
        if m:
          basename = m.group(1)
          reg_num = m.group(2)
          name = f"{basename}__reg{int(reg_num)+1}"
        else:
          name = name + "__reg1"
    with get_user_loc():
      return seq.CompRegOp(self.value.type,
                           input=self.value,
                           clk=clk,
                           reset=rst,
                           name=name,
                           sym_name=name)

  @property
  def _namehint_attrname(self):
    if self.value.owner.name == "seq.compreg":
      return "name"
    return "sv.namehint"

  @property
  def name(self):
    owner = self.value.owner
    if hasattr(owner,
               "attributes") and self._namehint_attrname in owner.attributes:
      return ir.StringAttr(owner.attributes[self._namehint_attrname]).value
    from circt.dialects import msft
    if isinstance(owner, ir.Block) and isinstance(owner.owner,
                                                  msft.MSFTModuleOp):
      mod = owner.owner
      return ir.StringAttr(
          ir.ArrayAttr(mod.attributes["argNames"])[self.value.arg_number]).value
    if hasattr(self, "_name"):
      return self._name

  @name.setter
  def name(self, new: str):
    owner = self.value.owner
    if hasattr(owner, "attributes"):
      owner.attributes[self._namehint_attrname] = ir.StringAttr.get(new)
    else:
      self._name = new


class RegularValue(Value):

  def __init__(self, value, type):
    self.value = value
    self.type = type


class BitVectorValue(Value):

  def __init__(self, value, type):
    self.value = value
    self.type = type

  def __getitem__(self, idxOrSlice: Union[int, slice]):
    if isinstance(idxOrSlice, int):
      s = slice(idxOrSlice, idxOrSlice + 1)
    elif isinstance(idxOrSlice, slice):
      s = idxOrSlice
    else:
      raise TypeError("Expected int or slice")
    idxs = s.indices(len(self))
    if idxs[2] != 1:
      raise ValueError("Integer / bitvector slices do not support steps")

    from .pycde_types import types
    from .dialects import comb
    ret_type = types.int(idxs[1] - idxs[0])

    with get_user_loc():
      ret = comb.ExtractOp(idxs[0], ret_type, self.value)
      if self.name is not None:
        ret.name = f"{self.name}_{idxs[0]}upto{idxs[1]}"
      return ret

  def __len__(self):
    return self.type.width


class ListValue(Value):

  def __init__(self, value, type):
    self.value = value
    self.type = type

  def __getitem__(self, sub):
    if isinstance(sub, int):
      idx = int(sub)
      if idx >= self.type.size:
        raise ValueError("Subscript out-of-bounds")
    else:
      idx = support.get_value(sub)
      if idx is None or not isinstance(support.type_to_pytype(idx.type),
                                       ir.IntegerType):
        raise TypeError("Subscript on array must be either int or MLIR int"
                        f" Value, not {type(sub)}.")
    from .dialects import hw
    with get_user_loc():
      v = hw.ArrayGetOp(self.value, idx)
      if self.name and isinstance(idx, int):
        v.name = self.name + f"__{idx}"
      return v

  def __len__(self):
    return self.type.strip.size


class StructValue(Value):

  def __init__(self, value, type):
    self.value = value
    self.type = type

  def __getitem__(self, sub):
    if sub not in [name for name, _ in self.type.strip.fields]:
      raise ValueError(f"Struct field '{sub}' not found in {self.type}")
    from .dialects import hw
    with get_user_loc():
      return hw.StructExtractOp(self.value, sub)

  def __getattr__(self, attr):
    ty = self.type.strip
    if attr in [name for name, _ in ty.fields]:
      from .dialects import hw
      with get_user_loc():
        v = hw.StructExtractOp(self.value, attr)
        if self.name:
          v.name = f"{self.name}__{attr}"
        return v
    raise AttributeError(f"'Value' object has no attribute '{attr}'")


def wrap_opviews_with_values(dialect, module_name):
  """Wraps all of a dialect's OpView classes to have their create method return
     a PyCDE Value instead of an OpView. The wrapped classes are inserted into
     the provided module."""
  import sys
  module = sys.modules[module_name]

  for attr in dir(dialect):
    cls = getattr(dialect, attr)

    if isinstance(cls, type) and issubclass(cls, ir.OpView):

      class ValueOpView(Value):
        _opview_cls = cls

        def __init__(self, *args, **kwargs):
          from .pycde_types import PyCDEType
          from .dialects import hw

          created = self._opview_cls.create(*args, **kwargs)
          if isinstance(created, support.NamedValueOpView):
            created = created.opview

          assert len(created.results) == 1
          value = created.results[0]
          type = PyCDEType(value.type)
          self._inst = type.get_value(value)
          if hasattr(self._inst, "__len__"):
            setattr(self, "__len__", lambda x: len(x._inst))

        def __getitem__(self, sub):
          return self._inst[sub]

        def __getattr__(self, attr):
          return getattr(self._inst, attr)

      wrapped_class = ValueOpView
      setattr(module, attr, wrapped_class)
    else:
      setattr(module, attr, cls)
