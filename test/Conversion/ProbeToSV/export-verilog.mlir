// RUN: circt-opt %s --lower-probe-to-sv --export-verilog | FileCheck %s

hw.module private @Producer(in %in: i8, out p: !probe.ref<i8>) {
  %p = probe.send %in : i8
  hw.output %p : !probe.ref<i8>
}

hw.module @Top(in %in: i8, out out: i8) {
  %p = hw.instance "producer" @Producer(in: %in: i8) ->
      (p: !probe.ref<i8>)
  %read = probe.read %p : <i8>
  hw.output %read : i8
}

// CHECK-LABEL: module Producer
// CHECK: wire [7:0] probe = in;
// CHECK: endmodule

// CHECK-LABEL: module Top
// CHECK: Producer producer (
// CHECK: assign out = Top.producer.probe;
// CHECK: endmodule
