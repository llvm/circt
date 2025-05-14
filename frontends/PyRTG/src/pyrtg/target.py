#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .core import CodeGenRoot, Value
from .base import ir
from .rtg import rtg


class Entry:
  """
  Represents an RTG Target Entry. Stores the entry function and location.
  """

  def __init__(self, entry_func) -> Entry:
    self.entry_func = entry_func

  @property
  def name(self) -> str:
    return self.entry_func.__name__


def entry(func):
  """
  Decorator for target entry functions. It computes one value returned from the
  target. The name of the function is used as the key in the target dictionary
  and the values returned from the target will be sorted by name.
  """

  return Entry(func)


def target(cls):
  """
  Represents an RTG Target. Constructs an instance of the decorated class which
  registers it as an RTG target.
  """

  def new_init(self):
    self._name = self.__class__.__name__
    self._dict = cls.__dict__

  cls = type(cls.__name__, (Target,) + cls.__bases__, dict(cls.__dict__))
  cls.__init__ = new_init
  instance = cls()
  return instance


class Target(CodeGenRoot):
  """
  An RTG Target is a collection of entry functions that define the capabilities
  and characteristics of a specific test target. Each entry function computes
  and returns a value that represents a particular feature or property of the
  target.
  """

  def _codegen(self) -> None:
    entries = []
    names = []

    #  Collect entries from the class dictionary.
    for attr_name, attr in self.__class__.__dict__.items():
      if isinstance(attr, Entry):
        entries.append(attr)
        names.append(attr_name)

    # Construct the target operation.
    target_op = rtg.TargetOp(self._name, ir.TypeAttr.get(rtg.DictType.get()))
    entry_block = ir.Block.create_at_start(target_op.bodyRegion, [])
    with ir.InsertionPoint(entry_block):
      results: list[Value] = []

      for entry in entries:
        results.append(entry.entry_func())

      rtg.YieldOp(results)

      dict_entries = [(ir.StringAttr.get(name), val.get_type())
                      for (name, val) in zip(names, results)]
      target_op.target = ir.TypeAttr.get(rtg.DictType.get(dict_entries))
