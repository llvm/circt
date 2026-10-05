// RUN: circt-opt --lower-hw-to-sv --allow-unregistered-dialect %s | FileCheck %s

// Currently the sv.macro.decl is emitted in LowerToHW, so just assumed that the 
// declaration is existed here.
sv.macro.decl @SYNTHESIS
sv.macro.decl @VERILATOR
hw.module @attach_ports(inout %a : i1, inout %b : i1, inout %c : i1) {
  hw.attach %a, %b, %c : !hw.inout<i1>, !hw.inout<i1>, !hw.inout<i1>
  hw.output
}

// CHECK: sv.macro.decl @SYNTHESIS
// CHECK: sv.macro.decl @VERILATOR
// CHECK-LABEL: hw.module @attach_ports
// CHECK: sv.ifdef @SYNTHESIS {
// CHECK:   [[R0:%.*]] = sv.read_inout %a : !hw.inout<i1>
// CHECK:   [[R1:%.*]] = sv.read_inout %b : !hw.inout<i1>
// CHECK:   [[R2:%.*]] = sv.read_inout %c : !hw.inout<i1>
// CHECK:   sv.assign %a, [[R1]] : i1
// CHECK:   sv.assign %a, [[R2]] : i1
// CHECK:   sv.assign %b, [[R0]] : i1
// CHECK:   sv.assign %b, [[R2]] : i1
// CHECK:   sv.assign %c, [[R0]] : i1
// CHECK:   sv.assign %c, [[R1]] : i1
// CHECK: } else {
// CHECK:   sv.ifdef @VERILATOR {
// CHECK:     sv.verbatim "`error {{.*}}bidirectional wires and ports{{.*}}"
// CHECK:   } else {
// CHECK:     sv.alias %a, %b, %c : !hw.inout<i1>, !hw.inout<i1>, !hw.inout<i1>
// CHECK:   }
// CHECK: }
