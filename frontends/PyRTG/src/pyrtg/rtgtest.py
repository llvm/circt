#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import ir
from .base.dialects import rtgtest as _rtgtest
from .support import _create


class _RTGTestDialect:
  """Frontend view of rtgtest with explicit operand conversion.

  The underlying OpView classes are left untouched.  This proxy only converts
  PyRTG operands at the public frontend boundary and returns the normal
  binding OpView.
  """

  def __getattr__(self, name):
    attr = getattr(_rtgtest, name)
    if isinstance(attr, type) and issubclass(attr, ir.OpView):
      return lambda *args, **kwargs: _create(attr, *args, **kwargs)
    return attr


rtgtest = _RTGTestDialect()
