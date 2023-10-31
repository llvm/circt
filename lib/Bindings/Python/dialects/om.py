#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from ._om_ops_gen import *
from .._mlir_libs._circt._om import Evaluator as BaseEvaluator, Object as BaseObject, List as BaseList, Tuple as BaseTuple, Map as BaseMap, BasePath as BaseBasePath, BasePathType, Path, PathType, ClassType, ReferenceAttr, ListAttr, MapAttr, OMIntegerAttr

from ..ir import Attribute, Diagnostic, DiagnosticSeverity, Module, StringAttr, IntegerAttr, IntegerType
from ..support import attribute_to_var, var_to_attribute

import sys
import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Sequence, TypeVar

if TYPE_CHECKING:
  from _typeshed.stdlib.dataclass import DataclassInstance


# Wrap a base mlir object with high-level object.
def wrap_mlir_object(value):
  # For primitives, return a Python value.
  if isinstance(value, Attribute):
    return attribute_to_var(value)

  if isinstance(value, BaseList):
    return List(value)

  if isinstance(value, BaseTuple):
    return Tuple(value)

  if isinstance(value, BaseMap):
    return Map(value)

  if isinstance(value, BaseBasePath):
    return BasePath(value)

  if isinstance(value, Path):
    return value

  # For objects, return an Object, wrapping the base implementation.
  assert isinstance(value, BaseObject)
  return Object(value)


def om_var_to_attribute(obj, none_on_fail: bool = False) -> ir.Attrbute:
  if isinstance(obj, int):
    return OMIntegerAttr.get(IntegerAttr.get(IntegerType.get_signless(64), obj))
  return var_to_attribute(obj, none_on_fail)


def unwrap_python_object(value):
  # Check if the value is a Primitive.
  try:
    return om_var_to_attribute(value)
  except:
    pass

  if isinstance(value, List):
    return BaseList(value)

  if isinstance(value, Tuple):
    return BaseTuple(value)

  if isinstance(value, Map):
    return BaseMap(value)

  if isinstance(value, BasePath):
    return BaseBasePath(value)

  if isinstance(value, Path):
    return value

  # Otherwise, it must be an Object. Cast to the mlir object.
  assert isinstance(value, Object)
  return BaseObject(value)


class List(BaseList):

  def __init__(self, obj: BaseList) -> None:
    super().__init__(obj)

  def __getitem__(self, i):
    val = super().__getitem__(i)
    return wrap_mlir_object(val)

  # Support iterating over a List by yielding its elements.
  def __iter__(self):
    for i in range(0, self.__len__()):
      yield self.__getitem__(i)


class Tuple(BaseTuple):

  def __init__(self, obj: BaseTuple) -> None:
    super().__init__(obj)

  def __getitem__(self, i):
    val = super().__getitem__(i)
    return wrap_mlir_object(val)

  # Support iterating over a Tuple by yielding its elements.
  def __iter__(self):
    for i in range(0, self.__len__()):
      yield self.__getitem__(i)


class Map(BaseMap):

  def __init__(self, obj: BaseMap) -> None:
    super().__init__(obj)

  def __getitem__(self, key):
    val = super().__getitem__(key)
    return wrap_mlir_object(val)

  def keys(self):
    return [wrap_mlir_object(arg) for arg in super().keys()]

  def items(self):
    for i in self:
      yield i

  def values(self):
    for (_, v) in self:
      yield v

  # Support iterating over a Map
  def __iter__(self):
    for i in super().keys():
      yield (wrap_mlir_object(i), self.__getitem__(i))


class BasePath(BaseBasePath):

  @staticmethod
  def get_empty(context=None) -> "BasePath":
    return BasePath(BaseBasePath.get_empty(context))


# Define the Object class by inheriting from the base implementation in C++.
class Object(BaseObject):

  def __init__(self, obj: BaseObject) -> None:
    super().__init__(obj)

  def __getattr__(self, name: str):
    # Call the base method to get a field.
    field = super().__getattr__(name)
    return wrap_mlir_object(field)

  def get_field_loc(self, name: str):
    # Call the base method to get the loc.
    loc = super().get_field_loc(name)
    return loc

  # Support iterating over an Object by yielding its fields.
  def __iter__(self):
    for name in self.field_names:
      yield (name, getattr(self, name))


# Define the Evaluator class by inheriting from the base implementation in C++.
class Evaluator(BaseEvaluator):

  def __init__(self, mod: Module) -> None:
    """Instantiate an Evaluator with a Module."""

    # Call the base constructor.
    super().__init__(mod)

    # Set up logging for diagnostics.
    logging.basicConfig(
        format="[%(asctime)s] %(name)s (%(levelname)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )
    self._logger = logging.getLogger("Evaluator")

    # Attach our Diagnostic handler.
    mod.context.attach_diagnostic_handler(self._handle_diagnostic)

  def instantiate(self, cls: str, *args: Any) -> Object:
    """Instantiate an Object with a class name and actual parameters."""

    # Convert the class name and actual parameters to Attributes within the
    # Evaluator's context.
    with self.module.context:
      # Get the class name from the class name.
      class_name = StringAttr.get(cls)

      # Get the actual parameter Values from the supplied variadic
      # arguments.
      actual_params = [unwrap_python_object(arg) for arg in args]

    # Call the base instantiate method.
    obj = super().instantiate(class_name, actual_params)

    # Return the Object, wrapping the base implementation.
    return Object(obj)

  def _handle_diagnostic(self, diagnostic: Diagnostic) -> bool:
    """Handle MLIR Diagnostics by logging them."""

    # Log the diagnostic message at the appropriate level.
    if diagnostic.severity == DiagnosticSeverity.ERROR:
      self._logger.error(diagnostic.message)
    elif diagnostic.severity == DiagnosticSeverity.WARNING:
      self._logger.warning(diagnostic.message)
    else:
      self._logger.info(diagnostic.message)

    # Log any diagnostic notes at the info level.
    for note in diagnostic.notes:
      self._logger.info(str(note))

    # Flush the stdout stream to ensure logs appear when expected.
    sys.stdout.flush()

    # Return True, indicating this diagnostic has been fully handled.
    return True
