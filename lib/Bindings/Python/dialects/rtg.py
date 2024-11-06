#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._rtg_ops_gen import *
from .._mlir_libs._circt._rtg import *
from ..ir import *
from .arith import ConstantOp

# def value_add(self, other: Value):
#   return addi(self, other)

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

def target(entries):
  def decorator(fn):
    def wrapper(ctx):
      names = []
      types = []
      for (name, (_, t)) in entries:
        names.append(StringAttr.get(name, ctx))
        types.append(t(ctx))
      tgt = TargetOp(fn.__name__, TypeAttr.get(TargetType.get(ctx, names, types)))
      block = Block.create_at_start(tgt.bodyRegion, [])
      with InsertionPoint(block):
        els = fn()
        operands = []
        for e in els:
          if isinstance(e, Set):
            operands.append(e.set)
          else:
            operands.append(e)
        YieldOp(operands)
      return tgt
      
    wrapper._is_target = True
    return wrapper
  return decorator

def test(entries):
  def decorator(fn):
    def wrapper(ctx):
      names = []
      types = []
      wrappers = []
      for (name, (w, t)) in entries:
        names.append(StringAttr.get(name, ctx))
        types.append(t(ctx))
        wrappers.append(w)
      tst = TestOp(fn.__name__, TypeAttr.get(TargetType.get(ctx, names, types)))
      block = Block.create_at_start(tst.bodyRegion, types)
      with InsertionPoint(block):
        args = []
        for wrap, arg in zip(wrappers, block.arguments):
            args.append(wrap(arg))
        fn(*args)
      return tst
      
    wrapper._is_test = True
    return wrapper
  return decorator

class ContextResource:
  def __init__(self, resource):
    self.resource = resource

  def __enter__(self):
    res = self.resource
    if isinstance(res, Set):
      res = res.set
    c = OnContextOp(res)
    body = Block.create_at_start(c.bodyRegion, [])
    self.ip = InsertionPoint(body)
    self.ip.__enter__()
  
  def __exit__(self, exc_type, exc_value, traceback):
    self.ip.__exit__(exc_type, exc_value, traceback)

def context(resource):
  return ContextResource(resource)

def label(name, args=[]):
  decl = LabelDeclOp(IntegerType.get_signless(32), name, args)
  LabelOp(decl.label)


def context_resource():
  def _context_resource(ctx):
    return ContextResourceType.get(ctx)

  return _context_resource

def set_of(elementTypeFn):
  def _set_of(ctx):
    return SetType.get(ctx, elementTypeFn(ctx))
  def _wrap(val):
    set = Set()
    set.set = val
    return set

  return _wrap, _set_of


class Set:
  def create(elements, elementType=None):
    set = Set()
    if elementType == None:
      elementType = elements[0].type
    set.elementType = elementType
    set.set = set_create(SetType.get(None, elementType), elements)
    return set
  
  def __sub__(self, other):
    if isinstance(other, Set):
      s = Set()
      s.set = set_difference(self.set, other.set)
      return s

    other_set = Set.create([other])
    s = Set()
    s.set = set_difference(self.set, other_set.set)
    return s

  def get_random(self):
    return set_select_random(self.set)
  
  def get_random_and_exclude(self):
    r = self.get_random()
    self = self - r
    return r
