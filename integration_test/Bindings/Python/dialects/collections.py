# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import om
from circt.ir import Context, Location, Module, Type

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  module = Module.parse("""
  module {
    om.class @Source() -> (items: !om.collection<!om.integer>) {
      %items = om.collection : !om.collection<!om.integer>
      %one = om.constant #om.integer<1 : si8> : !om.integer
      om.collection_push %items, %one : !om.collection<!om.integer>, !om.integer
      om.collection_push %items, %one : !om.collection<!om.integer>, !om.integer
      om.class.fields %items : !om.collection<!om.integer>
    }
    om.class @Append(%items: !om.collection<!om.integer>) -> (items: !om.collection<!om.integer>) {
      %two = om.constant #om.integer<2 : si8> : !om.integer
      om.collection_push %items, %two : !om.collection<!om.integer>, !om.integer
      om.class.fields %items : !om.collection<!om.integer>
    }
    om.class @Unknowns() -> (items: !om.collection<!om.integer>, missing: !om.collection<!om.integer>) {
      %items = om.collection : !om.collection<!om.integer>
      %element = om.unknown : !om.integer
      %missing = om.unknown : !om.collection<!om.integer>
      om.collection_push %items, %element : !om.collection<!om.integer>, !om.integer
      om.class.fields %items, %missing : !om.collection<!om.integer>, !om.collection<!om.integer>
    }
  }
  """)
  evaluator = om.Evaluator(module)
  source = evaluator.instantiate("Source", [])
  assert isinstance(source.items, om.MutableCollection)
  assert not hasattr(source.items, "append")
  assert len(source.items) == 2
  assert sorted(source.items) == [1, 1]
  assert source.items[-1] == 1
  try:
    source.items[2]
    raise AssertionError("expected IndexError")
  except IndexError:
    pass

  appended = evaluator.instantiate("Append", [source.items])
  assert sorted(appended.items) == [1, 1, 2]
  assert sorted(source.items) == [1, 1]
  unknowns = evaluator.instantiate("Unknowns", [])
  assert len(unknowns.items) == 1
  assert isinstance(unknowns.items[0], om.Unknown)
  assert isinstance(unknowns.missing, om.Unknown)
  ty = om.CollectionType(Type.parse("!om.collection<!om.integer>"))
  assert str(ty.element_type) == "!om.integer"
  # CHECK: MutableCollection tests passed
  print("MutableCollection tests passed")
