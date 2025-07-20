# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import aig, hw
from circt.ir import Context, Location, Module, InsertionPoint, IntegerType
from circt.dialects.aig import LongestPathAnalysis, LongestPathCollection, DataflowPath

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  m = Module.create()
  with InsertionPoint(m.body):
    # Test basic AIG dialect functionality
    i32 = IntegerType.get_signless(32)

    # Create a simple hardware module with AIG operations
    def build_module(module):
      a, b = module.entry_block.arguments

      result1 = aig.and_inv([a, b], [False, True])
      result2 = aig.and_inv([a, result1], [True, False])

      hw.OutputOp([result1, result2])

    hw.HWModuleOp(name="test_aig",
                  input_ports=[("a", i32), ("b", i32)],
                  output_ports=[("out1", i32), ("out2", i32)],
                  body_builder=build_module)
    # CHECK-LABEL: AIG dialect registration and basic operations successful!
    print("AIG dialect registration and basic operations successful!")
    # Test aig.and_inv operation
    # CHECK: %[[RESULT1:.*]] = aig.and_inv %a, not %b : i32
    # CHECK: %{{.*}} = aig.and_inv not %a, %[[RESULT1]] : i32
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
    # CHECK-NEXT: collection.get_path(5) == collection[5]: True
    print(
        "collection.get_path(5) == collection[5]:",
        DataflowPath.from_json_string(
            collection.collection.get_path(5)) == collection[5])
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
    # CHECK:      top:test_aig;a[0] 0
    # CHECK-NEXT: top:test_aig;out2[0] 2
    print(collection.longest_path.to_flamegraph())
