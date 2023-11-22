// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-materialize-debug-info)))' %s | FileCheck %s

firrtl.circuit "Ports" {

// CHECK-LABEL: firrtl.module @Ports
firrtl.module @Ports(
  in %inA: !firrtl.uint<42>,
  in %inB: !firrtl.bundle<a: sint<19>, b: clock>,
  in %inC: !firrtl.vector<asyncreset, 2>,
  in %inD: !firrtl.bundle<clocks: vector<clock, 4>>,
  out %outA: !firrtl.uint<42>
) {
  // CHECK-NEXT: dbg.variable "inA", %inA

  // CHECK-NEXT: [[TMP0:%.+]] = firrtl.subfield %inB[a]
  // CHECK-NEXT: [[TMP1:%.+]] = firrtl.subfield %inB[b]
  // CHECK-NEXT: [[TMP:%.+]] = dbg.struct {"a": [[TMP0]], "b": [[TMP1]]}
  // CHECK-NEXT: dbg.variable "inB", [[TMP]]

  // CHECK-NEXT: [[TMP0:%.+]] = firrtl.subindex %inC[0]
  // CHECK-NEXT: [[TMP1:%.+]] = firrtl.subindex %inC[1]
  // CHECK-NEXT: [[TMP:%.+]] = dbg.array [[[TMP0]], [[TMP1]]]
  // CHECK-NEXT: dbg.variable "inC", [[TMP]]

  // CHECK-NEXT: [[TMP1:%.+]] = firrtl.subfield %inD[clocks]
  // CHECK-NEXT: [[TMP2:%.+]] = firrtl.subindex [[TMP1]][0]
  // CHECK-NEXT: [[TMP3:%.+]] = firrtl.subindex [[TMP1]][1]
  // CHECK-NEXT: [[TMP4:%.+]] = firrtl.subindex [[TMP1]][2]
  // CHECK-NEXT: [[TMP5:%.+]] = firrtl.subindex [[TMP1]][3]
  // CHECK-NEXT: [[TMP6:%.+]] = dbg.array [[[TMP2]], [[TMP3]], [[TMP4]], [[TMP5]]]
  // CHECK-NEXT: [[TMP7:%.+]] = dbg.struct {"clocks": [[TMP6]]}
  // CHECK-NEXT: dbg.variable "inD", [[TMP7]]

  // CHECK-NEXT: dbg.variable "outA", %outA

  // CHECK-NEXT: firrtl.strictconnect
  firrtl.strictconnect %outA, %inA : !firrtl.uint<42>
}

// CHECK-LABEL: firrtl.module @Decls
firrtl.module @Decls() {
  // CHECK-NEXT: firrtl.constant
  // CHECK-NEXT: firrtl.constant
  // CHECK-NEXT: firrtl.specialconstant
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c0_ui17 = firrtl.constant 0 : !firrtl.uint<17>
  %c0_clock = firrtl.specialconstant 0 : !firrtl.clock

  // CHECK-NEXT: firrtl.wire
  // CHECK-NEXT: dbg.variable "someWire", %someWire
  %someWire = firrtl.wire : !firrtl.uint<17>

  // CHECK-NEXT: firrtl.node
  // CHECK-NEXT: dbg.variable "someNode", %someNode
  %someNode = firrtl.node %c0_ui17 : !firrtl.uint<17>

  // CHECK-NEXT: firrtl.reg
  // CHECK-NEXT: dbg.variable "someReg1", %someReg1
  %someReg1 = firrtl.reg %c0_clock : !firrtl.clock, !firrtl.uint<17>

  // CHECK-NEXT: firrtl.regreset
  // CHECK-NEXT: dbg.variable "someReg2", %someReg2
  %someReg2 = firrtl.regreset %c0_clock, %c0_ui1, %c0_ui17 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<17>

  // CHECK-NEXT: firrtl.strictconnect
  firrtl.strictconnect %someWire, %c0_ui17 : !firrtl.uint<17>
}

}
