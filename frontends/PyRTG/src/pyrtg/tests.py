#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import ir
from .core import CodeGenRoot
from .rtg import rtg
from .support import _FromCirctValue


class Test(CodeGenRoot):
  """
  Represents an RTG Test. Stores the test function and location.
  """

  def __init__(self, test_func, args: list[tuple[str, ir.Type]]):
    self.test_func = test_func
    self.arg_names = [name for name, _ in args]
    self.arg_types = [ty for _, ty in args]

  @property
  def name(self) -> str:
    return self.test_func.__name__

  def _codegen(self):
    test = rtg.TestOp(
        self.name, self.name,
        ir.TypeAttr.get(
            rtg.DictType.get([
                (ir.StringAttr.get(name), ty)
                for (name, ty) in zip(self.arg_names, self.arg_types)
            ])))
    block = ir.Block.create_at_start(test.bodyRegion, self.arg_types)
    with ir.InsertionPoint(block):
      self.test_func(*[_FromCirctValue(arg) for arg in block.arguments])


def test(*args, **kwargs):
  """
  Decorator for RTG test functions.
  """

  def wrapper(func):
    return Test(func, list(args))

  return wrapper


def embed_comment(comment: str) -> None:
  """
  Embeds a comment in the instruction stream.
  """

  rtg.CommentOp(comment)
