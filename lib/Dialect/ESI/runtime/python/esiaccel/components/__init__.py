# ===- __init__.py - high-performance ESI component library ---------------===//
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//
#
#  A library of reusable, high-performance ESI hardware components built with
#  PyCDE. These are timing-optimized building blocks (arbiters, muxes, buffers,
#  ...) intended to be instantiated from BSPs and accelerator designs. Each
#  component is documented under `docs/`.
#
# ===-----------------------------------------------------------------------===//

from .channel_arbiter import ChannelArbiter, ChannelArbiterMod

__all__ = ["ChannelArbiter", "ChannelArbiterMod"]
