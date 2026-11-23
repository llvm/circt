// RUN: circt-opt --llhd-undriven-to-symbolic %s | FileCheck %s

// CHECK-LABEL: @Probed
hw.module @Probed(out o : i32) {
  // CHECK-NEXT: %free = verif.symbolic_value : i32
  // CHECK-NOT: llhd.
  // CHECK-NEXT: hw.output %free
  %free = llhd.sig : <i32>
  %0 = llhd.prb %free : i32
  hw.output %0 : i32
}

// CHECK-LABEL: @Projections
hw.module @Projections(in %i : i1, in %j : i2, out a : i8, out b : i4, out c : i2, out d : !hw.array<2xi8>) {
  %c2_i3 = hw.constant 2 : i3
  // CHECK: [[SYM:%s]] = verif.symbolic_value : !hw.struct<x: !hw.array<4xi8>, y: i8>
  %s = llhd.sig : <!hw.struct<x: !hw.array<4xi8>, y: i8>>
  // CHECK-DAG: [[X:%.+]] = hw.struct_extract [[SYM]]["x"]
  // CHECK-DAG: [[GET:%.+]] = hw.array_get [[X]][%j]
  %x = llhd.sig.struct_extract %s["x"] : <!hw.struct<x: !hw.array<4xi8>, y: i8>>
  %elem = llhd.sig.array_get %x[%j] : <!hw.array<4xi8>>
  %0 = llhd.prb %elem : i8
  // CHECK-DAG: [[Y:%.+]] = hw.struct_extract [[SYM]]["y"]
  // CHECK-DAG: [[BITS:%.+]] = comb.extract [[Y]] from 2 : (i8) -> i4
  %y = llhd.sig.struct_extract %s["y"] : <!hw.struct<x: !hw.array<4xi8>, y: i8>>
  %bits = llhd.sig.extract %y from %c2_i3 : <i8> -> <i4>
  %1 = llhd.prb %bits : i4
  // CHECK-DAG: [[SLICE:%.+]] = hw.array_slice [[X]][%j]
  %slice = llhd.sig.array_slice %x at %j : <!hw.array<4xi8>> -> <!hw.array<2xi8>>
  %3 = llhd.prb %slice : !hw.array<2xi8>
  // CHECK-DAG: [[U:%u]] = verif.symbolic_value : !hw.union<p: i2, q: i2>
  // CHECK-DAG: [[UQ:%.+]] = hw.union_extract [[U]]["q"]
  %u = llhd.sig : <!hw.union<p: i2, q: i2>>
  %q = llhd.sig.struct_extract %u["q"] : <!hw.union<p: i2, q: i2>>
  %2 = llhd.prb %q : i2
  // CHECK-NOT: llhd.
  // CHECK: hw.output [[GET]], [[BITS]], [[UQ]], [[SLICE]]
  hw.output %0, %1, %2, %3 : i8, i4, i2, !hw.array<2xi8>
}

// Probes inside processes read the symbolic value as well.
// CHECK-LABEL: @ProbedInProcess
hw.module @ProbedInProcess(out o : i8) {
  // CHECK: [[SYM:%free]] = verif.symbolic_value : i8
  %free = llhd.sig : <i8>
  // CHECK: llhd.process
  %0 = llhd.process -> i8 {
    // CHECK-NOT: llhd.prb
    // CHECK: llhd.halt [[SYM]]
    %1 = llhd.prb %free : i8
    llhd.halt %1 : i8
  }
  hw.output %0 : i8
}

// CHECK-LABEL: @NotReplaced
hw.module @NotReplaced(in %v : i8, out o : !llhd.ref<i8>, out p : i8) {
  // CHECK-NOT: verif.symbolic_value
  %t = llhd.constant_time <0ns, 1d>
  // Driven.
  // CHECK: %driven = llhd.sig
  %driven = llhd.sig : <i8>
  llhd.drv %driven, %v after %t : i8
  // Initialized.
  // CHECK: %init = llhd.sig %v
  %init = llhd.sig %v : <i8>
  %0 = llhd.prb %init : i8
  // Passed out of the module, where it may be driven.
  // CHECK: %escapes = llhd.sig
  %escapes = llhd.sig : <i8>
  // Driven through a projection.
  // CHECK: %partial = llhd.sig
  %partial = llhd.sig : <!hw.array<2xi8>>
  %c0_i1 = hw.constant 0 : i1
  %1 = llhd.sig.array_get %partial[%c0_i1] : <!hw.array<2xi8>>
  llhd.drv %1, %v after %t : i8
  // Declared inside a process.
  // CHECK: llhd.process
  llhd.process {
    // CHECK: %local = llhd.sig
    %local = llhd.sig : <i8>
    %2 = llhd.prb %local : i8
    llhd.halt
  }
  hw.output %escapes, %0 : !llhd.ref<i8>, i8
}
