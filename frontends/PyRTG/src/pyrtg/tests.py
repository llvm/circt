#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import ir
from .core import CodeGenRoot
from .rtg import rtg
from .support import _FromCirctValue

from types import SimpleNamespace


class Test(CodeGenRoot):
  """
  Represents an RTG Test. Stores the test function and location.
  """

  def __init__(self, test_func, config):
    self.test_func = test_func
    self.config = config

  @property
  def name(self) -> str:
    return self.test_func.__name__

  def _codegen(self):
    # The config the test is referring to must be codegen'd first because only then we have guaranteed access to the param types.
    if not self.config._already_generated:
      self.config._codegen()

    self._already_generated = True

    # Sort arguments by name
    params_sorted = self.config.get_params()
    params_sorted.sort(key=lambda param: param.get_name())

    test = rtg.TestOp(
        self.name, self.name,
        ir.TypeAttr.get(
            rtg.DictType.get([(ir.StringAttr.get(param.get_name()),
                               param.get_type()) for param in params_sorted])))
    block = ir.Block.create_at_start(
        test.bodyRegion, [param.get_type() for param in params_sorted])
    new_config = []
    for param, arg in zip(params_sorted, block.arguments):
      new_config.append((param.get_name(), _FromCirctValue(arg)))

    with ir.InsertionPoint(block):
      self.test_func(SimpleNamespace(new_config))


def test(config):
  """
  Decorator for RTG test functions.
  """

  def wrapper(func):
    return Test(func, config)

  return wrapper


def embed_comment(comment: str) -> None:
  """
  Embeds a comment in the instruction stream.
  """

  rtg.CommentOp(comment)
