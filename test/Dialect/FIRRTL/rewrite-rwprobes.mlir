// RUN: circt-opt %s --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-rewrite-rwprobes)))' --split-input-file | FileCheck %s --implicit-check-not=firrtl.ref.rwprobe.ssa

// CHECK-LABEL: circuit "RWProbeInst"
firrtl.circuit "RWProbeInst" {
  firrtl.module private @Child(in %x: !firrtl.vector<uint<5>, 2>, out %y: !firrtl.uint<5>) {
    %0 = firrtl.subindex %x[1] : !firrtl.vector<uint<5>, 2>
    %1 = firrtl.subindex %x[0] : !firrtl.vector<uint<5>, 2>
    %2 = firrtl.and %1, %0 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
    firrtl.strictconnect %y, %2 : !firrtl.uint<5>
  }
  // CHECK-LABEL: module @RWProbeInst(
  firrtl.module @RWProbeInst(in %x: !firrtl.vector<uint<5>, 2>,
                             out %rw_x0: !firrtl.rwprobe<uint<5>>,
                             out %rw_x1: !firrtl.rwprobe<uint<5>>,
                             out %rw_y: !firrtl.rwprobe<uint<5>>) attributes {convention = #firrtl<convention scalarized>} {
    // Instance results become wires with symbols.
    // CHECK-DAG: %[[C_X:.+]] = firrtl.wire sym [<@[[CX0_SYM:.+]],1,public>, <@[[CX1_SYM:.+]],2,public>] interesting_name
    // CHECK-DAG: %[[C_Y:.+]] = firrtl.wire sym @[[CY_SYM:.+]] interesting_name {{.*}} : !firrtl.uint<5>
    // CHECK-DAG: firrtl.strictconnect %c_x, %[[C_X]]
    // CHECK-DAG: firrtl.strictconnect %[[C_Y]], %c_y

    // Replacement ops.
    // CHECK-DAG: %[[RW_CY:.+]] = firrtl.ref.rwprobe <@RWProbeInst::@[[CY_SYM]]>
    // CHECK-DAG: %[[RW_CX0:.+]] = firrtl.ref.rwprobe <@RWProbeInst::@[[CX0_SYM]]>
    // CHECK-DAG: %[[RW_CX1:.+]] = firrtl.ref.rwprobe <@RWProbeInst::@[[CX1_SYM]]>

    // ref.define uses sanity check.
    // CHECK-DAG: firrtl.ref.define %rw_x0, %[[RW_CX0]]
    // CHECK-DAG: firrtl.ref.define %rw_x1, %[[RW_CX1]]
    // CHECK-DAG: firrtl.ref.define %rw_y, %[[RW_CY]]
    %c_x, %c_y = firrtl.instance c interesting_name @Child(in x: !firrtl.vector<uint<5>, 2>, out y: !firrtl.uint<5>)
    %1 = firrtl.subindex %c_x[0] : !firrtl.vector<uint<5>, 2>
    %0 = firrtl.subindex %c_x[1] : !firrtl.vector<uint<5>, 2>
    firrtl.strictconnect %c_x, %x : !firrtl.vector<uint<5>, 2>
    %2 = firrtl.ref.rwprobe.ssa %1 : !firrtl.uint<5>
    firrtl.ref.define %rw_x0, %2 : !firrtl.rwprobe<uint<5>>
    %3 = firrtl.ref.rwprobe.ssa %0 : !firrtl.uint<5>
    firrtl.ref.define %rw_x1, %3 : !firrtl.rwprobe<uint<5>>
    %4 = firrtl.ref.rwprobe.ssa %c_y : !firrtl.uint<5>
    firrtl.ref.define %rw_y, %4 : !firrtl.rwprobe<uint<5>>
  }
}

// -----

// CHECK-LABEL: circuit "RWProbePort" {
firrtl.circuit "RWProbePort" {
   // CHECK-LABEL: module @RWProbePort(
   // CHECK-SAME: in %in: !firrtl.vector<uint<1>, 2> sym [<@[[PORT_SYM:.+]],2,public>]
  firrtl.module @RWProbePort(in %in: !firrtl.vector<uint<1>, 2>, out %p: !firrtl.rwprobe<uint<1>>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.subindex %in[1] : !firrtl.vector<uint<1>, 2>
    // CHECK: %[[RW:.+]] = firrtl.ref.rwprobe <@RWProbePort::@[[PORT_SYM]]>
    %1 = firrtl.ref.rwprobe.ssa %0 : !firrtl.uint<1>
    // CHECK: ref.define %p, %[[RW]]
    firrtl.ref.define %p, %1 : !firrtl.rwprobe<uint<1>>
  }
}
