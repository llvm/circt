#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._rtg_ops_gen import *
from .._mlir_libs._circt._rtg import *
from ..ir import *
from .hw import ConstantOp

def sequence(cpu):
  def decorator(fn):
    def wrapper(ctx):
      snp = SequenceOp(SequenceType.get(ctx, []))
      block = Block.create_at_start(snp.bodyRegion, [])
      with InsertionPoint(block):
        fn()
      return snp 

    wrapper._is_sequence = True
    return wrapper
  return decorator

def init(cpu):
  def decorator(fn):
    def wrapper(ctx):
      snp = SequenceOp(SequenceType.get(ctx, []))
      block = Block.create_at_start(snp.bodyRegion, [])
      with InsertionPoint(block):
        fn()
      return snp
      
    wrapper._is_sequence = True
    wrapper._is_init = True
    return wrapper
  return decorator


def scope(cls):
  cls._is_scope = True
  cls._materialized_sequences = dict()
  cls._ctx = None
  cls._seq_ip = None

  def new_init(self, module):
    decorated_methods = []
    cls._ctx = module.context
    cls._seq_ip = InsertionPoint(module.body)

    for attr_name in dir(cls):
      attr = getattr(cls, attr_name)
      if callable(attr) and getattr(attr, '_is_sequence', False):
        decorated_methods.append(attr)
      if attr_name == 'register':
        attr(module.context)

    for fn in decorated_methods:
      cls._get_or_create_materialized_sequence(fn)

  def _get_or_create_materialized_sequence(fn):
    if fn not in cls._materialized_sequences:
      with cls._seq_ip:
        cls._materialized_sequences[fn] = fn(cls._ctx)
    return cls._materialized_sequences[fn]

  def select_random_sequence(sequences, ratios):
    materialized_sequences = []
    seq_arg_operand_segments = []
    for seq in sequences:
      materialized_sequences.append(cls._get_or_create_materialized_sequence(seq))
      seq_arg_operand_segments.append(0) #FIXME

    materialized_ratios = []
    for ratio in ratios:
      if isinstance(ratio, Value):
        materialized_ratios.append(ratio)
      else:
        materialized_ratios.append(ConstantOp.create(IntegerType.get_signless(32), ratio).result)
      
    select_random(materialized_sequences, materialized_ratios, [], seq_arg_operand_segments)

  cls.__init__ = new_init
  cls._get_or_create_materialized_sequence = _get_or_create_materialized_sequence
  cls.select_random_sequence = select_random_sequence
  return cls

def resource(cls):
  cls._is_resource = True
  return cls
