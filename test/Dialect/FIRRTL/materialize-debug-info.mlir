// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-materialize-debug-info)))' %s | FileCheck %s

firrtl.circuit "Ports" attributes {
  annotations = [
    {
      class = "chisel3.experimental.EnumAnnotations$EnumDefAnnotation", 
      definition = {IDLE = 0 : i64, A = 1 : i64, B = 2 : i64, C = 3 : i64, D = 4 : i64}, 
      typeName = "MyEnumMod$MyEnum"
    }]
  } {

// CHECK-LABEL: firrtl.module @Ports
firrtl.module @Ports(
  in %inA: !firrtl.uint<42>,
  in %inB: !firrtl.bundle<a: sint<19>, b: clock>,
  in %inC: !firrtl.vector<asyncreset, 2>,
  in %inD: !firrtl.bundle<clocks: vector<clock, 4>>,
  out %outA: !firrtl.uint<42>,

  //===----------------------------------------------------------------------===//
  // Type annotation for the ports
  //===----------------------------------------------------------------------===//
  in %inTypedA: !firrtl.uint<42> [{class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~Ports|Ports>inTypedA", typeName = "IO[UInt<42>]"}],
  in %inTypedB: !firrtl.bundle<a: sint<19>, b: clock> [{class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "IO[MyBundle]"},  // Target is not required anymore in this pass
                                                       {circt.fieldID = 1 : i32, class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "IO[AinMyBundle]"},
                                                       {circt.fieldID = 2 : i32, class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "IO[ClockInMyBundle]"}],
  in %inTypedC: !firrtl.vector<asyncreset, 2> [{class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "IO[Vec<AsyncReset>]"},
                                               {circt.fieldID = 1 : i32, class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "IO[AsyncReset]"}],
  in %inTypedD: !firrtl.bundle<clocks: vector<clock, 4>> [{class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~Ports|Ports>inTypedD", typeName = "IO[BundleVecClock]"},
                                                          {circt.fieldID = 1 : i32, class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~Ports|Ports>inTypedD.clocks", typeName = "IO[Clock[4]]"},
                                                          {circt.fieldID = 2 : i32, class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~Ports|Ports>inTypedD.clocks[0]", typeName = "IO[Clock]"}],
  in %inTypedEnum1: !firrtl.uint<3> [{class = "chisel3.experimental.EnumAnnotations$EnumComponentAnnotation", enumTypeName = "MyEnumMod$MyEnum", target = "~Ports|Ports>inTypedEnum"}],
  in %inTypedEnum2: !firrtl.uint<10> [{class = "chisel3.experimental.EnumAnnotations$EnumComponentAnnotation", enumTypeName = "MyEnumMod$MyEnum", target = "~Ports|Ports>inTypedEnum"}],
  
  out %outTypedA: !firrtl.uint<42> [{class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "Any Custom string"}]
) {
  // CHECK-NEXT: [[EDEF0:%.+]] = dbg.enumdef "MyEnumMod$MyEnum", id 0, {A = 1 : i64, B = 2 : i64, C = 3 : i64, D = 4 : i64, IDLE = 0 : i64}
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

  //===----------------------------------------------------------------------===//
  // Check the type annotations
  //===----------------------------------------------------------------------===//

  // CHECK-NEXT: dbg.variable "inTypedA", %inTypedA {typeName = "IO[UInt<42>]"}

  // CHECK-NEXT: [[TMP0:%.+]] = firrtl.subfield %inTypedB[a]
  // CHECK-NEXT: [[TMP1:%.+]] = dbg.subfield "inTypedB.a", [[TMP0]] {typeName = "IO[AinMyBundle]"}
  // CHECK-NEXT: [[TMP2:%.+]] = firrtl.subfield %inTypedB[b]
  // CHECK-NEXT: [[TMP3:%.+]] = dbg.subfield "inTypedB.b", [[TMP2]] {typeName = "IO[ClockInMyBundle]"}
  // CHECK-NEXT: [[TMP:%.+]] = dbg.struct {"a": [[TMP1]], "b": [[TMP3]]}
  // CHECK-NEXT: dbg.variable "inTypedB", [[TMP]] {typeName = "IO[MyBundle]"}

  // CHECK-NEXT: [[TMP0:%.+]] = firrtl.subindex %inTypedC[0]
  // CHECK-NEXT: [[TMP1:%.+]] = dbg.subfield "inTypedC[0]", [[TMP0]] {typeName = "IO[AsyncReset]"}
  // CHECK-NEXT: [[TMP2:%.+]] = firrtl.subindex %inTypedC[1]
  // CHECK-NEXT: [[TMP3:%.+]] = dbg.subfield "inTypedC[1]", [[TMP2]] {typeName = "IO[AsyncReset]"}
  // CHECK-NEXT: [[TMP:%.+]] = dbg.array [[[TMP1]], [[TMP3]]] 
  // CHECK-NEXT: dbg.variable "inTypedC", [[TMP]] {typeName = "IO[Vec<AsyncReset>]"}
  
  // CHECK-NEXT: [[TMP0:%.+]] = firrtl.subfield %inTypedD[clocks]
  // CHECK-NEXT: [[TMP1:%.+]] = firrtl.subindex [[TMP0]][0]
  // CHECK-NEXT: [[TMP2:%.+]] = dbg.subfield "inTypedD.clocks[0]", [[TMP1]] {typeName = "IO[Clock]"}
  // CHECK-NEXT: [[TMP3:%.+]] = firrtl.subindex [[TMP0]][1]
  // CHECK-NEXT: [[TMP4:%.+]] = dbg.subfield "inTypedD.clocks[1]", [[TMP3]] {typeName = "IO[Clock]"}
  // CHECK-NEXT: [[TMP5:%.+]] = firrtl.subindex [[TMP0]][2]
  // CHECK-NEXT: [[TMP6:%.+]] = dbg.subfield "inTypedD.clocks[2]", [[TMP5]] {typeName = "IO[Clock]"}
  // CHECK-NEXT: [[TMP7:%.+]] = firrtl.subindex [[TMP0]][3]
  // CHECK-NEXT: [[TMP8:%.+]] = dbg.subfield "inTypedD.clocks[3]", [[TMP7]] {typeName = "IO[Clock]"}
  // CHECK-NEXT: [[TMP9:%.+]] = dbg.array [[[TMP2]], [[TMP4]], [[TMP6]], [[TMP8]]] 
  // CHECK-NEXT: [[TMP10:%.+]] = dbg.subfield "inTypedD.clocks", [[TMP9]] {typeName = "IO[Clock[4]]"}
  // CHECK-NEXT: [[TMP:%.+]] = dbg.struct {"clocks": [[TMP10]]}
  // CHECK-NEXT: dbg.variable "inTypedD", [[TMP]] {typeName = "IO[BundleVecClock]"} 
  
  // CHECK-NEXT: [[EDEF0:%.+]] = dbg.enumdef "MyEnumMod$MyEnum", id 0, {A = 1 : i64, B = 2 : i64, C = 3 : i64, D = 4 : i64, IDLE = 0 : i64}
  // CHECK-NEXT: dbg.variable "inTypedEnum1", %inTypedEnum1 enumDef [[EDEF0]] 
  // CHECK-NEXT: [[EDEF0:%.+]] = dbg.enumdef "MyEnumMod$MyEnum", id 0, {A = 1 : i64, B = 2 : i64, C = 3 : i64, D = 4 : i64, IDLE = 0 : i64}
  // CHECK-NEXT: dbg.variable "inTypedEnum2", %inTypedEnum2 enumDef [[EDEF0]] 

  // CHECK-NEXT: dbg.variable "outTypedA", %outTypedA {typeName = "Any Custom string"}

  // CHECK-NEXT: firrtl.strictconnect
  firrtl.strictconnect %outA, %inA : !firrtl.uint<42>

  // The type annotations do not compromise the rest of the code
  // CHECK-NEXT: firrtl.strictconnect
  firrtl.strictconnect %outTypedA, %inTypedA : !firrtl.uint<42>
}

// CHECK-LABEL: firrtl.module @Decls
firrtl.module @Decls() {
  // CHECK-NEXT: [[EDEF0:%.+]] = dbg.enumdef "MyEnumMod$MyEnum", id 0, {A = 1 : i64, B = 2 : i64, C = 3 : i64, D = 4 : i64, IDLE = 0 : i64}
  // CHECK-NEXT: firrtl.constant
  // CHECK-NEXT: firrtl.constant
  // CHECK-NEXT: firrtl.specialconstant
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c0_ui17 = firrtl.constant 0 : !firrtl.uint<17>
  %c0_clock = firrtl.specialconstant 0 : !firrtl.clock

  // CHECK-NEXT: firrtl.wire
  // CHECK-NEXT: dbg.variable "someWire", %someWire
  // CHECK-NEXT: firrtl.wire
  // CHECK-NEXT: dbg.variable "someTypedWire", %someTypedWire {typeName = "Wire[SInt<17>]"}
  %someWire = firrtl.wire : !firrtl.uint<17>
  %someTypedWire = firrtl.wire {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "Wire[SInt<17>]"}]} : !firrtl.uint<17>

  // CHECK-NEXT: firrtl.node
  // CHECK-NEXT: dbg.variable "someNode", %someNode
  // CHECK-NEXT: firrtl.node
  // CHECK-NEXT: dbg.variable "someTypedNode", %someTypedNode {typeName = "UInt<17>"}
  %someNode = firrtl.node %c0_ui17 : !firrtl.uint<17>
  %someTypedNode = firrtl.node %c0_ui17 {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "UInt<17>"}]} : !firrtl.uint<17>

  // CHECK-NEXT: firrtl.reg
  // CHECK-NEXT: dbg.variable "someReg1", %someReg1
  // CHECK-NEXT: firrtl.reg
  // CHECK-NEXT: dbg.variable "someTypedReg1", %someTypedReg1 {typeName = "Reg[SInt<17>]"}
  %someReg1 = firrtl.reg %c0_clock : !firrtl.clock, !firrtl.uint<17>
  %someTypedReg1 = firrtl.reg %c0_clock {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "Reg[SInt<17>]"}]} : !firrtl.clock, !firrtl.uint<17>

  // CHECK-NEXT: firrtl.regreset
  // CHECK-NEXT: dbg.variable "someReg2", %someReg2
  // CHECK-NEXT: firrtl.regreset
  // CHECK-NEXT: dbg.variable "someTypedReg2", %someTypedReg2 {typeName = "Reg[SInt<17>]"}
  %someReg2 = firrtl.regreset %c0_clock, %c0_ui1, %c0_ui17 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<17>
  %someTypedReg2 = firrtl.regreset %c0_clock, %c0_ui1, %c0_ui17 {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "Reg[SInt<17>]"}]} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<17>

  // CHECK-NEXT: firrtl.strictconnect
  firrtl.strictconnect %someWire, %c0_ui17 : !firrtl.uint<17>  

  // CHECK-NEXT: firrtl.strictconnect
  firrtl.strictconnect %someTypedWire, %someWire : !firrtl.uint<17>
}

// CHECK-LABEL: firrtl.module @ConstructorParams
firrtl.module @ConstructorParams() {
  // CHECK-NEXT: [[EDEF0:%.+]] = dbg.enumdef "MyEnumMod$MyEnum", id 0, {A = 1 : i64, B = 2 : i64, C = 3 : i64, D = 4 : i64, IDLE = 0 : i64}
  // CHECK-NEXT: firrtl.constant
  %c0_ui17 = firrtl.constant 0 : !firrtl.uint<17>

  // CHECK-NEXT: firrtl.wire
  // CHECK-NEXT: dbg.variable "someTypedWire", %someTypedWire {params = [{name = "size", typeName = "int", value = "17"}], typeName = "Wire[SInt<17>]"}
  %someTypedWire = firrtl.wire {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "Wire[SInt<17>]", params = [{name="size", typeName="int", value="17"}]}]} : !firrtl.uint<17>
  
  // CHECK-NEXT: firrtl.wire
  // CHECK-NEXT: dbg.variable "anotherTypedWire", %anotherTypedWire {params = [{name = "p", typeName = "char"}], typeName = "Wire[SInt<17>]"}
  %anotherTypedWire = firrtl.wire {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", typeName = "Wire[SInt<17>]", params = [{name="p", typeName="char"}]}]} : !firrtl.uint<17>
  
  // CHECK-NEXT: firrtl.strictconnect
  firrtl.strictconnect %someTypedWire, %c0_ui17 : !firrtl.uint<17>  

}
}
firrtl.circuit "TopCircuitMultiModule" {
    // CHECK-LABEL: firrtl.module private @MyModule()
    // CHECK-NEXT: dbg.moduleinfo {typeName = "MyModule"}
    firrtl.module private @MyModule() attributes {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~TopCircuitMultiModule|MyModule", typeName = "MyModule"}]} {
      firrtl.skip
    }
    // CHECK-LABEL: firrtl.module private @MyModule_1()
    // CHECK-NEXT: dbg.moduleinfo {typeName = "MyModule"}
    firrtl.module private @MyModule_1() attributes {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~TopCircuitMultiModule|MyModule_1", typeName = "MyModule"}]} {
      firrtl.skip
    }
    // CHECK-LABEL: firrtl.module private @MyModule_2()
    // CHECK-NEXT: dbg.moduleinfo {typeName = "MyModule"}
    firrtl.module private @MyModule_2() attributes {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~TopCircuitMultiModule|MyModule_2", typeName = "MyModule"}]} {
      firrtl.skip
    }
    // CHECK-LABEL: firrtl.module private @MyModule_3()
    // CHECK-NEXT: dbg.moduleinfo {typeName = "MyModule"}
    firrtl.module private @MyModule_3() attributes {annotations = [{class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~TopCircuitMultiModule|MyModule_3", typeName = "MyModule"}]} {
      firrtl.skip
    }
    firrtl.module @TopCircuitMultiModule() attributes {annotations = [{class = "firrtl.transforms.DedupGroupAnnotation", group = "TopCircuitMultiModule"}, {class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~TopCircuitMultiModule|TopCircuitMultiModule", typeName = "TopCircuitMultiModule"}], convention = #firrtl<convention scalarized>} {
      firrtl.instance mod    interesting_name @MyModule()
      firrtl.instance mod1   interesting_name @MyModule()
      firrtl.instance mod2   interesting_name @MyModule_1()
      firrtl.instance mods_0 interesting_name @MyModule_2()
      firrtl.instance mods_1 interesting_name @MyModule_3()
    }
}
