#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .core import CodeGenRoot
from .base import ir
from .rtg import rtg


class Param:
  """
  This class represents a configuration parameter. An instance of this class
  should be added as an attribute to a config class (i.e., a class that
  inherits from 'Config' and has a '@config' decorator).

  There are two ways to do this:
  * Declare the attribute in the class directly or in the constructor. In this
    case, the 'loader' keyword argument has to be used to pass a callable
    (e.g., a lambda) which will construct the value to be assigned to this
    parameter.
  * Declare the attribute in the 'load' method. Here either the 'loader'
    callback can be used or the 'value' keyword argument can be used to assign
    the value directly.

  Note that any parameter assignments/declarations in the 'load' method take
  precedence.

  In both cases, the 'name' keyword argument can be used to assign a name to
  this parameter. You have to make sure that no two parameters in a config have
  the same name. By default, the name of the attribute within the config class
  will be used as the name.
  """

  def __init__(self, **kwargs) -> Param:
    if "value" in kwargs:
      self._value = kwargs["value"]

    if "loader" in kwargs:
      self._loader = kwargs["loader"]

    if "name" in kwargs:
      self._name = kwargs["name"]

    if hasattr(self, "_value") == hasattr(self, "_loader"):
      raise AttributeError(
          "either the 'value' or 'loader' argument must be used but not both")

  def get_type(self) -> ir.Type:
    if not hasattr(self, "_value"):
      raise AttributeError(
          "type can only be accesses after the config has been loaded")

    return self._value.get_type()

  def get_name(self) -> str:
    if not hasattr(self, "_name"):
      raise AttributeError(
          "name can only be accesses after the config has been loaded")

    return self._name


def config(cls):
  """
  Represents an RTG Configuration. Constructs an instance of the decorated
  class which registers it as an RTG config.
  """

  inst = cls()
  for attr_name, attr in cls.__dict__.items():
    if isinstance(attr, Param):
      setattr(inst, attr_name, attr)
  inst._name = cls.__name__
  return inst


class Config(CodeGenRoot):
  """
  An RTG Config is a collection of parameters that define the capabilities and
  characteristics of a specific test target.
  """

  def get_params(self) -> list[Param]:
    if not self._already_generated:
      raise RuntimeError(
          "can only get params once the config has been generated")

    params = []
    for attr_name, attr in self.__dict__.items():
      if isinstance(attr, Param):
        if not hasattr(attr, "_name"):
          attr._name = attr_name
        params.append(attr)

    return params

  def _codegen(self) -> None:
    self._already_generated = True

    # Construct the target operation.
    target_op = rtg.TargetOp(self._name, ir.TypeAttr.get(rtg.DictType.get()))
    entry_block = ir.Block.create_at_start(target_op.bodyRegion, [])
    with ir.InsertionPoint(entry_block):
      if hasattr(self, "load"):
        self.load()

      params = self.get_params()
      for param in params:
        if not hasattr(param, "_value"):
          param._value = param._loader()

      params.sort(key=lambda param: param.get_name())
      rtg.YieldOp([param._value for param in params])

      dict_entries = [(ir.StringAttr.get(param.get_name()), param.get_type())
                      for param in params]
      target_op.target = ir.TypeAttr.get(rtg.DictType.get(dict_entries))
