# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw, comb, synth
from circt.ir import Context, Location, Module, InsertionPoint, IntegerType, ArrayAttr
from circt.passmanager import PassManager
from circt.dialects.synth import LongestPathAnalysis, LongestPathCollection, DataflowPath

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  m = Module.create()
  with InsertionPoint(m.body):
    i4 = IntegerType.get_signless(4)

    # Create a module with comb.mul
    def build_module(module):
      a, b = module.entry_block.arguments
      hw.OutputOp([comb.mul([a, b])])

    hw.HWModuleOp(
        name="foo",
        input_ports=[("a", i4), ("b", i4)],
        output_ports=[("out", i4)],
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
      print("delay", p.delay, ":", p.fan_out)

    # CHECK-NEXT: minus index slice: True
    print("minus index slice:", len(collection[:-2]) == len(collection) - 2)

    # Test framegraph emission.
    # CHECK: top:test_aig;c[0] 0
    # CHECK: top:test_aig;child:test_child;a[0] 2
    print(collection.longest_path.to_flamegraph())

    test_child = m.body.operations[0]
    body_block = test_child.regions[0].blocks[0]
    result0 = body_block.operations[0].results[0]
    result1 = body_block.operations[1].results[0]

    analysis = LongestPathAnalysis(test_child,
                                   collect_debug_info=True,
                                   keep_only_max_delay_paths=True,
                                   lazy_computation=True)
    c1 = analysis.get_paths(result0, 0)
    c2 = analysis.get_paths(result1, 0)
    # CHECK-NEXT: len(c1) = 1
    # CHECK-NEXT: len(c2) = 1
    print("len(c1) =", len(c1))
    print("len(c2) =", len(c2))

    s1, s2 = len(c1), len(c2)
    c1.merge(c2)
    # CHECK-NEXT: merge: True
    print("merge:", len(c1) == s1 + s2)
