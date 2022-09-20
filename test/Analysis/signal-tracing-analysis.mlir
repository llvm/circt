// RUN: circt-opt -test-signal-tracing-analysis %s | FileCheck %s

// CHECK: %true = hw.constant true : SignalState(0)
// CHECK: %0 = comb.or %in, %true : i1 : SignalState(1)
// CHECK: %wire = sv.wire  : !hw.inout<i1> : SignalState(1)
// CHECK: %0 = sv.read_inout %wire : !hw.inout<i1> : SignalState(1)
// CHECK: %input = sv.wire  {source} : !hw.inout<i1> : SignalState(1)
// CHECK: %true = hw.constant true : SignalState(0)
// CHECK: %0 = sv.read_inout %input : !hw.inout<i1> : SignalState(1)
// CHECK: %a.out = hw.instance "a" @A(in: %0: i1) -> (out: i1) : SignalState(1)
// CHECK: %b.out = hw.instance "b" @B(in: %a.out: i1) -> (out: i1) : SignalState(1)
// CHECK: %output = sv.wire  {sink} : !hw.inout<i1> : SignalState(1)

hw.module private @A(%in: i1) -> (out: i1) {
  %0 = hw.constant 1 : i1
  %1 = comb.or %in, %0 : i1
  hw.output %1 : i1
}

hw.module private @B(%in: i1) -> (out: i1) {
  %wire = sv.wire : !hw.inout<i1>
  sv.assign %wire, %in : i1
  %0 = sv.read_inout %wire : !hw.inout<i1>
  hw.output %0 : i1
}

hw.module @Top() {
  %input = sv.wire {source} : !hw.inout<i1>
  %0 = hw.constant 1 : i1
  sv.assign %input, %0 : i1
  %1 = sv.read_inout %input : !hw.inout<i1>

  %2 = hw.instance "a" @A(in: %1: i1) -> (out: i1)

  %3 = hw.instance "b" @B(in: %2: i1) -> (out: i1)

  %output = sv.wire {sink} : !hw.inout<i1>
  sv.assign %output, %3 : i1
}
