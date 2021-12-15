// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s


module {
// CHECK-LABEL: module {
  hw.globalRef @glbl_B_M1 [#hw.innerNameRef<@A::@inst_1>, #hw.innerNameRef<@B::@memInst>]
  hw.globalRef @glbl_D_M1 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@memInst>]
    hw.globalRef @glbl_D_M2 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@SF>, #hw.innerNameRef<@F::@symA>]
    hw.globalRef @glbl_D_M3 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>,   #hw.innerNameRef<@D::@SF>, #hw.innerNameRef<@F::@symB>]

  // CHECK:  hw.globalRef @glbl_B_M1 [#hw.innerNameRef<@A::@inst_1>, #hw.innerNameRef<@B::@memInst>]
  // CHECK:  hw.globalRef @glbl_D_M1 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@memInst>]
  // CHECK:  hw.globalRef @glbl_D_M2 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@SF>, #hw.innerNameRef<@F::@symA>]
  // CHECK:  hw.globalRef @glbl_D_M3 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@SF>, #hw.innerNameRef<@F::@symB>]

  hw.module.extern @F(%in: i1 {hw.exportPort = @symA}) -> (out: i1 {hw.exportPort = @symB}) attributes {circt.globalRef = [[#hw.globalNameRef<@glbl_D_M2>], [#hw.globalNameRef<@glbl_D_M3>]]}
  hw.module @FIRRTLMem() -> () {
  }
  hw.module @D() -> () {
    hw.instance "M1" sym @memInst @FIRRTLMem() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>]}
    %c0 = hw.constant 0 : i1
    %2 = hw.instance "ab" sym @SF  @F (in: %c0: i1) -> (out : i1) {circt.globalRef = [#hw.globalNameRef<@glbl_D_M2>, #hw.globalNameRef<@glbl_D_M3>]}
  }
  hw.module @B() -> () {
     hw.instance "M1" sym @memInst @FIRRTLMem() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_B_M1>]}
  }
  hw.module @C() -> () {
    hw.instance "m" sym @inst @D() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>, #hw.globalNameRef<@glbl_D_M2>, #hw.globalNameRef<@glbl_D_M3>]}
  }
  hw.module @A() -> () {
    hw.instance "h1" sym @inst_1 @B() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_B_M1>]}
    hw.instance "h2" sym @inst_0 @C() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>, #hw.globalNameRef<@glbl_D_M2>, #hw.globalNameRef<@glbl_D_M3>]}
  }
}
