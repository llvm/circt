#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect

from .circt import ir
from .circt.dialects import rtg


class Test:
  """
  Represents an RTG Test. Stores the test function and location.
  """

  type: ir.Type

  def __init__(self, test_func):
    self.test_func = test_func

    sig = inspect.signature(test_func)
    assert len(sig.parameters) == 0, "test arguments not supported yet"

    self.type = rtg.DictType.get([])

  @property
  def name(self) -> str:
    return self.test_func.__name__

  def codegen(self):
    test = rtg.TestOp(self.name, ir.TypeAttr.get(self.type))
    block = ir.Block.create_at_start(test.bodyRegion, [])
    with ir.InsertionPoint(block):
      self.test_func(*block.arguments)


def test(func):
  """
  Decorator for RTG test functions.
  """

  return Test(func)
