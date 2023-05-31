// RUN: circt-opt %s --arc-sink-pure-arcs | FileCheck %s

arc.define @SimpleA(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 : i4
  arc.output %0 : i4
}

arc.define @SimpleB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.or %arg0, %arg1 : i4
  arc.output %0 : i4
}

hw.module @test(%clk: i1, %arg0: i4, %arg1: i4, %arg2: i4) -> (out0: i4, out1: i4) {
  %c = hw.constant 0 : i4
  %0 = arc.call @SimpleA(%arg0, %arg1) : (i4, i4) -> i4
  %1 = arc.call @SimpleA(%arg0, %arg1) : (i4, i4) -> i4
  %4 = arc.call @SimpleA(%0, %1) : (i4, i4) -> i4
  %3 = arc.state @SimpleA(%4, %arg0) clock %clk lat 1 : (i4, i4) -> i4
  %2 = arc.state @SimpleB(%4, %arg0) clock %clk lat 1 : (i4, i4) -> i4
  hw.output %2, %3 : i4, i4
}
