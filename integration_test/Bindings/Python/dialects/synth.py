# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw, comb, synth, seq
from circt.ir import Context, Location, Module, InsertionPoint, IntegerType, ArrayAttr
from circt.passmanager import PassManager
from circt.dialects.synth import LongestPathAnalysis, LongestPathCollection, DataflowPath

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  m = Module.create()
  with InsertionPoint(m.body):
    i2 = IntegerType.get_signless(4)

    # Create a module with comb.mul
    def build_module(module):
      a, b = module.entry_block.arguments
      hw.OutputOp([comb.mul([a, b])])

    hw.HWModuleOp(
        name="foo",
        input_ports=[("a", i2), ("b", i2)],
        output_ports=[("out", i2)],
        body_builder=build_module,
    )

  # Check that the synthesis pipeline is registered.
  pm = PassManager.parse(
      "builtin.module(hw.module(synth-comb-lowering-pipeline, "
      "synth-optimization-pipeline))")
  pm.run(m.operation)
  # CHECK: hw.module @foo(
  # CHECK-NOT: comb.mul
  # CHECK: synth.aig.and_inv
  print(m.operation)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  m = Module.create()
  with InsertionPoint(m.body):
    # Test basic AIG dialect functionality
    i32 = IntegerType.get_signless(32)

    # Create a simple hardware module with AIG operations
    def build_child(module):
      a, b = module.entry_block.arguments

      result1 = synth.AndInverterOp([a, b], [False, True])
      result2 = synth.AndInverterOp([a, result1], [True, False])

      hw.OutputOp([result1, result2])

    hw.HWModuleOp(name="test_child",
                  input_ports=[("a", i32), ("b", i32)],
                  output_ports=[("out1", i32), ("out2", i32)],
                  body_builder=build_child)

    def build_top(module):
      a, b = module.entry_block.arguments
      out1, out2 = hw.instance(
          [i32, i32],
          "child",
          "test_child",
          [a, b],
          ["a", "b"],
          ["out1", "out2"],
          parameters=ArrayAttr.get([]),
      )
      hw.OutputOp([out1, out2])

    hw.HWModuleOp(name="test_aig",
                  input_ports=[("c", i32), ("d", i32)],
                  output_ports=[("out1", i32), ("out2", i32)],
                  body_builder=build_top)

    # CHECK-LABEL: Synth dialect registration and basic operations successful!
    print("Synth dialect registration and basic operations successful!")
    # Test synth.aig.and_inv operation
    # CHECK: %[[RESULT1:.*]] = synth.aig.and_inv %a, not %b : i32
    # CHECK: %{{.*}} = synth.aig.and_inv not %a, %[[RESULT1]] : i32
    print(m)

    # Test LongestPathAnalysis Python bindings
    analysis = LongestPathAnalysis(m.operation)

    collection = analysis.get_all_paths("test_aig")
    # CHECK-LABEL:      LongestPathAnalysis created successfully!
    print("LongestPathAnalysis created successfully!")

    # CHECK:      Total paths: 128
    # CHECK-NEXT: Max delay: 2
    # CHECK-NEXT: Min delay: 1
    # CHECK-NEXT: 50th percentile delay: 1
    # CHECK-NEXT: 90th percentile delay: 2
    # CHECK-NEXT: 95th percentile delay: 2

    # CHECK-NEXT: 99th percentile delay: 2
    # CHECK-NEXT: 99.9th percentile delay: 2
    collection.print_summary()

    # Check index.
    # CHECK-NEXT: index 1 delay: 2
    # CHECK-NEXT: index -1 delay: 1
    print("index 1 delay:", collection[1].delay)
    print("index -1 delay:", collection[-1].delay)
    # Check that len and get_size are the same
    # CHECK-NEXT: 128 128
    print(len(collection), collection.collection.get_size())

    try:
      print(collection[128])
    except IndexError:
      # CHECK-NEXT: IndexError correctly raised
      print("IndexError correctly raised")

    # Check that iterator works
    # CHECK-NEXT: sum: 192
    print("sum: ", sum(p.delay for p in collection))

    for p in collection[:2]:
      # CHECK-NEXT: delay 2 : out2[{{[0-9]+}}]
      # CHECK-NEXT: delay 2 : out2[{{[0-9]+}}]
      print("delay", p.delay, ":", p.end_point)

    # CHECK-NEXT: minus index slice: True
    print("minus index slice:", len(collection[:-2]) == len(collection) - 2)

    # Test framegraph emission.
    # CHECK: top:test_aig;c[0] 0
    # CHECK: top:test_aig;child:test_child;a[0] 2
    print(collection.longest_path.to_flamegraph())

    original_length = len(collection)
    collection.drop_non_critical_paths(per_end_point=True)
    after_per_end_point = len(collection)
    collection.drop_non_critical_paths(per_end_point=False)
    after_per_start_point = len(collection)
    # CHECK-NEXT: drop_non_critical_paths: True True
    print("drop_non_critical_paths:", original_length > after_per_end_point,
          after_per_end_point > after_per_start_point)

    test_child = m.body.operations[0]
    body_block = test_child.regions[0].blocks[0]
    result0 = body_block.operations[0].results[0]
    result1 = body_block.operations[1].results[0]

    analysis = LongestPathAnalysis(test_child,
                                   collect_debug_info=True,
                                   keep_only_max_delay_paths=True,
                                   lazy_computation=True,
                                   top_module_name="test_child")
    c1 = analysis.get_paths(result0, 0)
    c2 = analysis.get_paths(result1, 0)
    # CHECK-NEXT: len(c1) = 1
    # CHECK-NEXT: len(c2) = 1
    s1, s2 = len(c1), len(c2)
    print("len(c1) =", s1)
    print("len(c2) =", s2)
    c1.merge(c2)
    # CHECK-NEXT: merge: True
    print("merge:", len(c1) == s1 + s2)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  m = Module.create()
  with InsertionPoint(m.body):
    i2 = IntegerType.get_signless(2)

    # Create a module with comb.mul
    def build_module(module):
      clock, a, b = module.entry_block.arguments
      and_inv = synth.AndInverterOp([a, b], [False, True])
      r = seq.CompRegOp(i2, and_inv, clock)
      hw.OutputOp([r])

    module = hw.HWModuleOp(
        name="foo",
        input_ports=[("clock", seq.ClockType.get(ctx)), ("a", i2), ("b", i2)],
        output_ports=[("out", i2)],
        body_builder=build_module,
    )

    analysis = LongestPathAnalysis(m.operation,
                                   collect_debug_info=True,
                                   keep_only_max_delay_paths=True,
                                   lazy_computation=False,
                                   top_module_name="foo")
    internal = analysis.get_internal_paths("foo")
    from_input = analysis.get_paths_from_input_ports_to_internal("foo")
    to_output = analysis.get_paths_from_internal_to_output_ports("foo")

    # CHECK-NEXT: len(internal) = 0
    # Not path between r -> r
    print("len(internal) =", len(internal))

    # CHECK-NEXT: len(from_input) = 4
    # 2 paths from a -> r, 2 paths from b -> r
    print("len(from_input) =", len(from_input))

    # CHECK-NEXT: len(to_output) = 2
    # 2 path from r -> out
    print("len(to_output) =", len(to_output))
