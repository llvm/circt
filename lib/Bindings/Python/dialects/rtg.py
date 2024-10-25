#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._rtg_ops_gen import *
from .._mlir_libs._circt._rtg import *
from ..ir import *

def sequence(cpu):
  def decorator(fn):
    def wrapper(ctx):
      snp = SequenceOp(SequenceType.get(ctx, []))
      block = Block.create_at_start(snp.bodyRegion, [])
      with InsertionPoint(block):
        fn()
    
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
      
    wrapper._is_sequence = True
    wrapper._is_init = True
    return wrapper
  return decorator

def scope(cls):
  cls._is_scope = True

  def new_init(self, module):
    decorated_methods = []

    for attr_name in dir(cls):
      attr = getattr(cls, attr_name)
      if callable(attr) and getattr(attr, '_is_sequence', False):
        decorated_methods.append(attr)
      if attr_name == 'register':
        attr(module.context)

    with InsertionPoint(module.body):
      for fn in decorated_methods:
        fn(module.context)
  cls.__init__ = new_init
  return cls


def resource(cls):
  cls._is_resource = True
  return cls
