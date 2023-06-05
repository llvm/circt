#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from ._om_ops_gen import *
from .._mlir_libs._circt._om import Evaluator as BaseEvaluator, Object

from circt.ir import Attribute, Diagnostic, DiagnosticSeverity, Module, StringAttr
from circt.support import attribute_to_var, var_to_attribute

import sys
import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
  from _typeshed.stdlib.dataclass import DataclassInstance


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

  def instantiate(self, cls: type["DataclassInstance"],
                  *args: Any) -> "DataclassInstance":
    """Instantiate an Object with a dataclass type and actual parameters."""

    # Convert the class name and actual parameters to Attributes within the
    # Evaluator's context.
    with self.module.context:
      # Get the class name from the provided dataclass name.
      class_name = StringAttr.get(cls.__name__)

      # Get the actual parameter Attributes from the supplied variadic
      # arguments. This relies on the circt.support helpers to convert from
      # Python objects to Attributes.
      actual_params = var_to_attribute(list(args))

    # Call the base instantiate method.
    obj = super().instantiate(class_name, actual_params)

    # Convert the field names of the class we are instantiating to StringAttrs
    # within the Evaluator's context.
    with self.module.context:
      field_names = [StringAttr.get(field.name) for field in fields(cls)]

    # Convert the instantiated Object fields from Attributes to Python objects.
    # This will be generalized to support Objects in fields soon.
    object_fields = {}
    for field_name in field_names:
      # Get the field from the object.
      field = obj.get_field(field_name)

      # Convert the field value to a Python object. This relies on the
      # circt.support helpers to convert from Attribute to Python objects.
      field_value = attribute_to_var(field)

      # Save this field in the keyword argument dictionary that will be passed
      # to the dataclass constructor.
      object_fields[field_name.value] = field_value

    # Instantiate a Python object of the requested class.
    return cls(**object_fields)

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
