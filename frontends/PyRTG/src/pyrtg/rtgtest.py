#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import ir
from .base.dialects import rtgtest as _rtgtest
from .core import Value


class _RTGTestDialect:

  def __getattr__(self, name):
    attr = getattr(_rtgtest, name)
    if isinstance(attr, type) and issubclass(attr, ir.OpView):

      def create(*args, **kwargs):
        def convert(value):
          if isinstance(value, Value):
            return value._get_ssa_value()
          if isinstance(value, (list, tuple)):
            return [convert(element) for element in value]
          return value

        return ir.Operation.create(
            attr.OPERATION_NAME,
            operands=[convert(arg) for arg in args],
            attributes=kwargs,
            regions=0)

      return create
    return attr


rtgtest = _RTGTestDialect()
