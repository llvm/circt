// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @proc_if
// CHECK: sv.alwayscomb {
// CHECK-NEXT:   sv.if %cond {
// CHECK:          sv.fwrite %{{.+}}, "hello"
// CHECK-NEXT:   }
// CHECK-NEXT: }
hw.module @proc_if(in %cond : i1) {
  %msg = sim.fmt.literal "hello"
  sv.alwayscomb {
    scf.if %cond {
      sim.proc.print %msg
    }
  }
  hw.output
}
