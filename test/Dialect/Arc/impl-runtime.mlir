// RUN: circt-opt --arc-impl-runtime %s | FileCheck %s

// CHECK-LABEL: func.func @Simple(
func.func @Simple() {
  %0 = sim.fmt.lit "Hello"
  sim.proc.print %0
  // call @use_fstring(%0) : (!sim.fstring) -> ()
  return
}

// func.func private @use_fstring(%arg0: !sim.fstring)
