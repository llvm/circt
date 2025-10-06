#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .modes import Mode
from .labels import Label

from typing import Callable


class Handler:
  def __init__(self, mode: Mode, func: Callable):
    self.mode = mode
    self.func = func


def handler(mode: Mode):
  def wrapper(func):
    return Handler(mode, func)
  
  return wrapper


def enable_handler(handler: Handler) -> None:
  illegal_instruction_handler_lbl = Label.declare_unique(
      "illegal_instruction_handler")
  test_lbl = Label.declare_unique("test")

  # # Handler setup
  # curr_mtvec = rv32i.csrr_r(CSR.mtvec())
  # handler_addr = rv32i.la_r(illegal_instruction_handler_lbl)
  # rv32i.csrw(CSR.mtvec(), handler_addr)
  # test_addr = rv32i.la_r(test_lbl)
  # rv32i.jalr(IntegerRegister.zero(), test_addr, Immediate(12, 0))

  # # Handler
  # illegal_instruction_handler_lbl.place()
  # handler.func()
  # rv32i.mret()

  # Test
  test_lbl.place()


def disable_handler(handler: Handler) -> None:
  # rv32i.csrw(CSR.mtvec(), reg)
  pass
