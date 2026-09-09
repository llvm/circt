#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .circt import ir, support, dialects

from functools import lru_cache


@lru_cache(maxsize=None)
def _cached_index_type(context):
  return ir.IndexType.get(context)


def _get_index_type():
  return _cached_index_type(ir.Context.current)


@lru_cache(maxsize=None)
def _cached_signless_integer_type(context, width):
  return ir.IntegerType.get_signless(width, context)


def _get_signless_integer_type(width):
  return _cached_signless_integer_type(ir.Context.current, width)


@lru_cache(maxsize=None)
def _cached_string_type(context):
  return dialects.rtg.StringType.get(context)


def _get_string_type():
  return _cached_string_type(ir.Context.current)

import sys

sys.modules[__name__ + '.dialects'] = dialects
