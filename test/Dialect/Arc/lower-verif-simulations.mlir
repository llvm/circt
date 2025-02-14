// RUN: circt-opt --arc-lower-verif-simulations %s | FileCheck %s

// CHECK: func.func private @exit(i32)

// CHECK-LABEL: hw.module @verif.simulation.impl.Foo(
// CHECK-SAME: in %clock : !seq.clock
// CHECK-SAME: in %init : i1
// CHECK-SAME: out done : i1
// CHECK-SAME: out exit_code : i32
verif.simulation @Foo {} {
^bb0(%clock: !seq.clock, %init: i1):
  // CHECK: [[TMP1:%.+]] = hw.constant true
  // CHECK: [[TMP2:%.+]] = hw.constant 0 : i32
  // CHECK: hw.output [[TMP1]], [[TMP2]] : i1, i32
  %true = hw.constant true
  %c0_i32 = hw.constant 0 : i32
  verif.yield %true, %c0_i32 : i1, i32
}

// CHECK-LABEL: func.func @Foo()
// CHECK: [[I0:%.+]] = hw.constant false
// CHECK: [[I1:%.+]] = hw.constant true
// CHECK: [[C0:%.+]] = seq.to_clock [[I0]]
// CHECK: [[C1:%.+]] = seq.to_clock [[I1]]
// CHECK: arc.sim.instantiate @verif.simulation.impl.Foo as [[A:%.+]] {
// CHECK:   scf.execute_region {
// CHECK:     arc.sim.set_input [[A]], "clock" = [[C0]]
// CHECK:     arc.sim.set_input [[A]], "init" = [[I1]]
// CHECK:     arc.sim.step [[A]]
// CHECK:     arc.sim.set_input [[A]], "clock" = [[C1]]
// CHECK:     arc.sim.step [[A]]
// CHECK:     arc.sim.set_input [[A]], "clock" = [[C0]]
// CHECK:     arc.sim.set_input [[A]], "init" = [[I0]]
// CHECK:     arc.sim.step [[A]]
// CHECK:     cf.br [[LOOP:\^.+]]
// CHECK:   [[LOOP]]:
// CHECK:     [[DONE:%.+]] = arc.sim.get_port [[A]], "done" : i1
// CHECK:     [[CODE:%.+]] = arc.sim.get_port [[A]], "exit_code" : i32
// CHECK:     arc.sim.set_input [[A]], "clock" = [[C1]]
// CHECK:     arc.sim.step [[A]]
// CHECK:     arc.sim.set_input [[A]], "clock" = [[C0]]
// CHECK:     arc.sim.step [[A]]
// CHECK:     cf.cond_br [[DONE]], [[EXIT:\^.+]], [[LOOP]]
// CHECK:   [[EXIT]]:
// CHECK:     [[TMP1:%.+]] = hw.constant 0 : i32
// CHECK:     [[TMP2:%.+]] = arith.cmpi ne, [[CODE]], [[TMP1]] : i32
// CHECK:     [[TMP3:%.+]] = arith.extui [[TMP2]] : i1 to i32
// CHECK:     [[TMP4:%.+]] = arith.ori [[CODE]], [[TMP3]] : i32
// CHECK:     func.call @exit([[TMP4]]) : (i32) -> ()
// CHECK:     scf.yield
// CHECK:   }
// CHECK: }
// CHECK: return

// CHECK-LABEL: func.func @NarrowExit()
// CHECK: [[CODE:%.+]] = arc.sim.get_port {{%.+}}, "exit_code" : i19
// CHECK: [[TMP1:%.+]] = hw.constant 0 : i19
// CHECK: [[TMP2:%.+]] = arith.cmpi ne, [[CODE]], [[TMP1]] : i19
// CHECK: [[TMP3:%.+]] = arith.extui [[TMP2]] : i1 to i32
// CHECK: [[TMP4:%.+]] = arith.extui [[CODE]] : i19 to i32
// CHECK: [[TMP5:%.+]] = arith.ori [[TMP4]], [[TMP3]] : i32
// CHECK: func.call @exit([[TMP5]]) : (i32) -> ()
verif.simulation @NarrowExit {} {
^bb0(%clock: !seq.clock, %init: i1):
  %true = hw.constant true
  %c0_i19 = hw.constant 0 : i19
  verif.yield %true, %c0_i19 : i1, i19
}

// CHECK-LABEL: func.func @WideExit()
// CHECK: [[CODE:%.+]] = arc.sim.get_port {{%.+}}, "exit_code" : i42
// CHECK: [[TMP1:%.+]] = hw.constant 0 : i42
// CHECK: [[TMP2:%.+]] = arith.cmpi ne, [[CODE]], [[TMP1]] : i42
// CHECK: [[TMP3:%.+]] = arith.extui [[TMP2]] : i1 to i32
// CHECK: [[TMP4:%.+]] = arith.trunci [[CODE]] : i42 to i32
// CHECK: [[TMP5:%.+]] = arith.ori [[TMP4]], [[TMP3]] : i32
// CHECK: func.call @exit([[TMP5]]) : (i32) -> ()
verif.simulation @WideExit {} {
^bb0(%clock: !seq.clock, %init: i1):
  %true = hw.constant true
  %c0_i42 = hw.constant 0 : i42
  verif.yield %true, %c0_i42 : i1, i42
}
