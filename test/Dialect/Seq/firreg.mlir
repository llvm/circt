// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | FileCheck %s

// CHECK-LABEL: hw.module @lowering
hw.module @lowering(%clk: i1, %rst: i1, %in: i32) -> (a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) {
  %cst0 = hw.constant 0 : i32

  // CHECK: %rA = sv.reg sym @regA : !hw.inout<i32>
  // CHECK: [[VAL_A:%.+]] = sv.read_inout %rA : !hw.inout<i32>
  %rA = seq.firreg %in clock %clk sym @regA : i32

  // CHECK: %rB = sv.reg sym @regB : !hw.inout<i32>
  // CHECK: [[VAL_B:%.+]] = sv.read_inout %rB : !hw.inout<i32>
  %rB = seq.firreg %in clock %clk sym @regB reset sync %rst, %cst0 : i32

  // CHECK: %rC = sv.reg sym @regC : !hw.inout<i32>
  // CHECK: [[VAL_C:%.+]] = sv.read_inout %rC : !hw.inout<i32>
  %rC = seq.firreg %in clock %clk sym @regC reset async %rst, %cst0 : i32

  // CHECK: %rD = sv.reg sym @regD : !hw.inout<i32>
  // CHECK: [[VAL_D:%.+]] = sv.read_inout %rD : !hw.inout<i32>
  %rD = seq.firreg %in clock %clk sym @regD : i32

  // CHECK: %rE = sv.reg sym @regE : !hw.inout<i32>
  // CHECK: [[VAL_E:%.+]] = sv.read_inout %rE : !hw.inout<i32>
  %rE = seq.firreg %in clock %clk sym @regE reset sync %rst, %cst0 : i32

  // CHECK: %rF = sv.reg sym @regF : !hw.inout<i32>
  // CHECK: [[VAL_F:%.+]] = sv.read_inout %rF : !hw.inout<i32>
  %rF = seq.firreg %in clock %clk sym @regF reset async %rst, %cst0 : i32

  // CHECK: %rAnamed = sv.reg sym @regA : !hw.inout<i32>
  %r = seq.firreg %in clock %clk sym @regA { "name" = "rAnamed" }: i32

  // CHECK: %rNoSym = sv.reg sym @__rNoSym__ : !hw.inout<i32>
  %rNoSym = seq.firreg %in clock %clk : i32

  // CHECK:      sv.always posedge %clk {
  // CHECK-NEXT:   sv.passign %rA, %in : i32
  // CHECK-NEXT:   sv.passign %rD, %in : i32
  // CHECK-NEXT:   sv.passign %rAnamed, %in : i32
  // CHECK-NEXT:   sv.passign %rNoSym, %in : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: sv.always posedge %clk {
  // CHECK-NEXT:   sv.if %rst {
  // CHECK-NEXT:     sv.passign %rB, %c0_i32 : i32
  // CHECK-NEXT:     sv.passign %rE, %c0_i32 : i32
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.passign %rB, %in : i32
  // CHECK-NEXT:     sv.passign %rE, %in : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: sv.always posedge %clk, posedge %rst {
  // CHECK-NEXT:   sv.if %rst {
  // CHECK-NEXT:     sv.passign %rC, %c0_i32 : i32
  // CHECK-NEXT:     sv.passign %rF, %c0_i32 : i32
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.passign %rC, %in : i32
  // CHECK-NEXT:     sv.passign %rF, %in : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // CHECK:      sv.ifdef  "SYNTHESIS" {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef  "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:       %_RANDOM = sv.reg sym @_RANDOM  : !hw.inout<i32>
  // CHECK-NEXT:       %_RANDOM_0 = sv.reg sym @_RANDOM_0  {name = "_RANDOM"} : !hw.inout<i32>
  // CHECK-NEXT:       %_RANDOM_1 = sv.reg sym @_RANDOM_1  {name = "_RANDOM"} : !hw.inout<i32>
  // CHECK-NEXT:       %_RANDOM_2 = sv.reg sym @_RANDOM_2  {name = "_RANDOM"} : !hw.inout<i32>
  // CHECK-NEXT:       %_RANDOM_3 = sv.reg sym @_RANDOM_3  {name = "_RANDOM"} : !hw.inout<i32>
  // CHECK-NEXT:       %_RANDOM_4 = sv.reg sym @_RANDOM_4  {name = "_RANDOM"} : !hw.inout<i32>
  // CHECK-NEXT:       %_RANDOM_5 = sv.reg sym @_RANDOM_5  {name = "_RANDOM"} : !hw.inout<i32>
  // CHECK-NEXT:       %_RANDOM_6 = sv.reg sym @_RANDOM_6  {name = "_RANDOM"} : !hw.inout<i32>
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_" {symbols = []}
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural  "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@lowering::@_RANDOM>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@lowering::@regA>, #hw.innerNameRef<@lowering::@_RANDOM>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@lowering::@_RANDOM_0>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@lowering::@regB>, #hw.innerNameRef<@lowering::@_RANDOM_0>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@lowering::@_RANDOM_1>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@lowering::@regC>, #hw.innerNameRef<@lowering::@_RANDOM_1>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@lowering::@_RANDOM_2>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@lowering::@regD>, #hw.innerNameRef<@lowering::@_RANDOM_2>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@lowering::@_RANDOM_3>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@lowering::@regE>, #hw.innerNameRef<@lowering::@_RANDOM_3>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@lowering::@_RANDOM_4>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@lowering::@regF>, #hw.innerNameRef<@lowering::@_RANDOM_4>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@lowering::@_RANDOM_5>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@lowering::@regA>, #hw.innerNameRef<@lowering::@_RANDOM_5>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@lowering::@_RANDOM_6>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@lowering::@__rNoSym__>, #hw.innerNameRef<@lowering::@_RANDOM_6>]}
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural  "RANDOMIZE" {
  // CHECK-NEXT:         sv.if %rst {
  // CHECK-NEXT:           sv.bpassign %rC, %c0_i32 : i32
  // CHECK-NEXT:           sv.bpassign %rF, %c0_i32 : i32
  // CHECK-NEXT:         }
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // CHECK: hw.output [[VAL_A]], [[VAL_B]], [[VAL_C]], [[VAL_D]], [[VAL_E]], [[VAL_F]] : i32, i32, i32, i32, i32, i32
  hw.output %rA, %rB, %rC, %rD, %rE, %rF : i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL: hw.module private @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
hw.module private @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
  // CHECK: %c0_i2 = hw.constant 0 : i2
  %c0_i2 = hw.constant 0 : i2
  // CHECK-NEXT: %count = sv.reg sym @count : !hw.inout<i2>
  // CHECK-NEXT: %0 = sv.read_inout %count : !hw.inout<i2>
  // CHECK-NEXT: %1 = comb.mux %cond, %value, %0 : i2
  // CHECK-NEXT: %2 = comb.mux %reset, %c0_i2, %1 : i2
  // CHECK-NEXT: sv.always posedge %clock {
  // CHECK-NEXT:   sv.passign %count, %2 : i2
  // CHECK-NEXT: }

  %count = seq.firreg %2 clock %clock sym @count : i2
  %1 = comb.mux %cond, %value, %count : i2
  %2 = comb.mux %reset, %c0_i2, %1 : i2

  // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:       %[[RANDOM:.+]] = sv.reg sym @[[RANDOM_SYM:[_A-Za-z0-9]+]] {{.+}}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_" {symbols = []}
  // CHECK-NEXT:       }
  // CHECK-NEXT:     sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK-NEXT:        sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@UninitReg1::@[[RANDOM_SYM]]>]}
  // CHECK-NEXT:        sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}}[1:0];" {symbols = [#hw.innerNameRef<@UninitReg1::@count>, #hw.innerNameRef<@UninitReg1::@[[RANDOM_SYM]]>]}
  // CHECK-NEXT:      }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // CHECK-NEXT: hw.output
  hw.output
}

// module InitReg1 :
//     input clock : Clock
//     input reset : UInt<1>
//     input io_d : UInt<32>
//     output io_q : UInt<32>
//     input io_en : UInt<1>
//
//     node _T = asAsyncReset(reset)
//     reg reg : UInt<32>, clock with :
//       reset => (_T, UInt<32>("h0"))
//     io_q <= reg
//     reg <= mux(io_en, io_d, reg)

// CHECK-LABEL: hw.module private @InitReg1(
hw.module private @InitReg1(%clock: i1, %reset: i1, %io_d: i32, %io_en: i1) -> (io_q: i32) {
  %false = hw.constant false
  %c1_i32 = hw.constant 1 : i32
  %c0_i32 = hw.constant 0 : i32
  %reg = seq.firreg %4 clock %clock sym @__reg__ reset async %reset, %c0_i32 : i32
  %reg2 = seq.firreg %reg2 clock %clock sym @__reg2__ reset sync %reset, %c0_i32 : i32
  %reg3 = seq.firreg %reg3 clock %clock sym @__reg3__ reset async %reset, %c1_i32 : i32
  %0 = comb.concat %false, %reg : i1, i32
  %1 = comb.concat %false, %reg2 : i1, i32
  %2 = comb.add %0, %1 : i33
  %3 = comb.extract %2 from 1 : (i33) -> i32
  %4 = comb.mux %io_en, %io_d, %3 : i32

  // CHECK:      %reg = sv.reg sym @[[reg_sym:.+]] : !hw.inout<i32>
  // CHECK-NEXT: %0 = sv.read_inout %reg : !hw.inout<i32>
  // CHECK-NEXT: %reg2 = sv.reg sym @[[reg2_sym:.+]] : !hw.inout<i32>
  // CHECK-NEXT: %1 = sv.read_inout %reg2 : !hw.inout<i32>
  // CHECK-NEXT: %reg3 = sv.reg sym @[[reg3_sym:.+]] : !hw.inout<i32
  // CHECK-NEXT: %2 = sv.read_inout %reg3 : !hw.inout<i32>
  // CHECK-NEXT: %3 = comb.concat %false, %0 : i1, i32
  // CHECK-NEXT: %4 = comb.concat %false, %1 : i1, i32
  // CHECK-NEXT: %5 = comb.add %3, %4 : i33
  // CHECK-NEXT: %6 = comb.extract %5 from 1 : (i33) -> i32
  // CHECK-NEXT: %7 = comb.mux %io_en, %io_d, %6 : i32
  // CHECK-NEXT: sv.always posedge %clock, posedge %reset  {
  // CHECK-NEXT:   sv.if %reset  {
  // CHECK-NEXT:     sv.passign %reg, %c0_i32 : i32
  // CHECK-NEXT:     sv.passign %reg3, %c1_i32 : i32
  // CHECK-NEXT:   } else  {
  // CHECK-NEXT:     sv.passign %reg, %7 : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: sv.always posedge %clock  {
  // CHECK-NEXT:   sv.if %reset  {
  // CHECK-NEXT:     sv.passign %reg2, %c0_i32 : i32
  // CHECK-NEXT:   } else  {
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:       %[[RANDOM:.+]] = sv.reg sym @[[RANDOM_SYM:[_A-Za-z0-9]+]] {{.+}}
  // CHECK-NEXT:       %[[RANDOM_2:.+]] = sv.reg sym @[[RANDOM_2_SYM:[_A-Za-z0-9]+]] {{.+}}
  // CHECK-NEXT:       %[[RANDOM_3:.+]] = sv.reg sym @[[RANDOM_3_SYM:[_A-Za-z0-9]+]] {{.+}}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_" {symbols = []}
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@InitReg1::@[[RANDOM_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@InitReg1::@[[reg_sym]]>, #hw.innerNameRef<@InitReg1::@[[RANDOM_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@InitReg1::@[[RANDOM_2_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@InitReg1::@[[reg2_sym]]>, #hw.innerNameRef<@InitReg1::@[[RANDOM_2_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@InitReg1::@[[RANDOM_3_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@InitReg1::@[[reg3_sym]]>, #hw.innerNameRef<@InitReg1::@[[RANDOM_3_SYM]]>]}
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE"  {
  // CHECK-NEXT:         sv.if %reset {
  // CHECK-NEXT:           sv.bpassign %reg, %c0_i32 : i32
  // CHECK-NEXT:           sv.bpassign %reg3, %c1_i32 : i32
  // CHECK-NEXT:         }
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output %0 : i32
  hw.output %reg : i32
}

// CHECK-LABEL: hw.module private @UninitReg42(%clock: i1, %reset: i1, %cond: i1, %value: i42) {
hw.module private @UninitReg42(%clock: i1, %reset: i1, %cond: i1, %value: i42) {
  %c0_i42 = hw.constant 0 : i42
  %count = seq.firreg %1 clock %clock sym @count : i42
  %0 = comb.mux %cond, %value, %count : i42
  %1 = comb.mux %reset, %c0_i42, %0 : i42

  // CHECK:      %count = sv.reg sym @count : !hw.inout<i42>
  // CHECK:      sv.ifdef "SYNTHESIS"  {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:       %[[RANDOM_0:.+]] = sv.reg sym @[[RANDOM_0_SYM:[_A-Za-z0-9]+]] {{.+}}
  // CHECK-NEXT:       %[[RANDOM_1:.+]] = sv.reg sym @[[RANDOM_1_SYM:[_A-Za-z0-9]+]] {{.+}}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_" {symbols = []}
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@UninitReg42::@[[RANDOM_0_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@UninitReg42::@[[RANDOM_1_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{][{]1[}][}]}}[9:0], {{[{][{]2[}][}][}]}};" {symbols = [#hw.innerNameRef<@UninitReg42::@count>, #hw.innerNameRef<@UninitReg42::@[[RANDOM_1_SYM]]>, #hw.innerNameRef<@UninitReg42::@[[RANDOM_0_SYM]]>]}
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  hw.output
}

// CHECK-LABEL: hw.module private @regInitRandomReuse
hw.module private @regInitRandomReuse(%clock: i1, %a: i1) -> (o1: i2, o2: i4, o3: i32, o4: i100) {
  %c0_i99 = hw.constant 0 : i99
  %c0_i31 = hw.constant 0 : i31
  %c0_i3 = hw.constant 0 : i3
  %false = hw.constant false
  %r1 = seq.firreg %0 clock %clock sym @__r1__ : i2
  %r2 = seq.firreg %1 clock %clock sym @__r2__ : i4
  %r3 = seq.firreg %2 clock %clock sym @__r3__ : i32
  %r4 = seq.firreg %3 clock %clock sym @__r4__ : i100
  %0 = comb.concat %false, %a : i1, i1
  %1 = comb.concat %c0_i3, %a : i3, i1
  %2 = comb.concat %c0_i31, %a : i31, i1
  %3 = comb.concat %c0_i99, %a : i99, i1
  // CHECK:      %r1 = sv.reg sym @[[r1_sym:[_A-Za-z0-9]+]]
  // CHECK:      %r2 = sv.reg sym @[[r2_sym:[_A-Za-z0-9]+]]
  // CHECK:      %r3 = sv.reg sym @[[r3_sym:[_A-Za-z0-9]+]]
  // CHECK:      %r4 = sv.reg sym @[[r4_sym:[_A-Za-z0-9]+]]
  // CHECK:      sv.ifdef "SYNTHESIS" {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:       %[[RANDOM_0:.+]] = sv.reg sym @[[RANDOM_0_SYM:[_A-Za-z0-9]+]]
  // CHECK-NEXT:       %[[RANDOM_1:.+]] = sv.reg sym @[[RANDOM_1_SYM:[_A-Za-z0-9]+]]
  // CHECK-NEXT:       %[[RANDOM_2:.+]] = sv.reg sym @[[RANDOM_2_SYM:[_A-Za-z0-9]+]]
  // CHECK-NEXT:       %[[RANDOM_3:.+]] = sv.reg sym @[[RANDOM_3_SYM:[_A-Za-z0-9]+]]
  // CHECK-NEXT:       %[[RANDOM_4:.+]] = sv.reg sym @[[RANDOM_4_SYM:[_A-Za-z0-9]+]]{{.+}}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_" {symbols = []}
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_0_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}}[1:0];" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[r1_sym]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_0_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}}[5:2];" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[r2_sym]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_0_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_1_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{]}}{{[{][{]1[}][}]}}[5:0], {{[{][{]2[}][}]}}[31:6]{{[}]}};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[r3_sym]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_1_SYM]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_0_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_2_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_3_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_4_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {{[{]}}{{[{][{]1[}][}]}}[9:0], {{[{][{]2[}][}]}}, {{[{][{]3[}][}]}}, {{[{][{]4[}][}]}}[31:6]{{[}]}};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[r4_sym]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_4_SYM]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_3_SYM]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_2_SYM]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_1_SYM]]>]}
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  hw.output %r1, %r2, %r3, %r4 : i2, i4, i32, i100
}

// CHECK-LABEL: hw.module private @init1DVector
hw.module private @init1DVector(%clock: i1, %a: !hw.array<2xi1>) -> (b: !hw.array<2xi1>) {
  %r = seq.firreg %a clock %clock sym @__r__ : !hw.array<2xi1>

  // CHECK:      %r = sv.reg sym @[[r_sym:[_A-Za-z0-9]+]]

  // CHECK:      sv.always posedge %clock  {
  // CHECK-NEXT:   sv.passign %r, %a : !hw.array<2xi1>
  // CHECK-NEXT: }

  // CHECK:      sv.ifdef "SYNTHESIS" {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:       %[[RANDOM:.+]] = sv.reg sym @[[RANDOM_SYM:[_A-Za-z0-9]+]]{{.+}}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_" {symbols = []}
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@init1DVector::@[[RANDOM_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}}[0] = {{[{][{]1[}][}]}}[0];" {symbols = [#hw.innerNameRef<@init1DVector::@[[r_sym]]>, #hw.innerNameRef<@init1DVector::@[[RANDOM_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}}[1] = {{[{][{]1[}][}]}}[1];" {symbols = [#hw.innerNameRef<@init1DVector::@[[r_sym]]>, #hw.innerNameRef<@init1DVector::@[[RANDOM_SYM]]>]}
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output %0 : !hw.array<2xi1>

  hw.output %r : !hw.array<2xi1>
}

// CHECK-LABEL: hw.module private @init2DVector
hw.module private @init2DVector(%clock: i1, %a: !hw.array<1xarray<1xi1>>) -> (b: !hw.array<1xarray<1xi1>>) {
  %r = seq.firreg %a clock %clock sym @__r__ : !hw.array<1xarray<1xi1>>

  // CHECK:      sv.always posedge %clock  {
  // CHECK-NEXT:   sv.passign %r, %a : !hw.array<1xarray<1xi1>>
  // CHECK-NEXT: }
  // CHECK-NEXT: sv.ifdef  "SYNTHESIS" {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef  "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:       %_RANDOM = sv.reg sym @_RANDOM  : !hw.inout<i32>
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_" {symbols = []}
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural  "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@init2DVector::@_RANDOM>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}}[0][0] = {{[{][{]1[}][}]}}[0];" {symbols = [#hw.innerNameRef<@init2DVector::@__r__>, #hw.innerNameRef<@init2DVector::@_RANDOM>]}
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  hw.output %r : !hw.array<1xarray<1xi1>>
  // CHECK: hw.output %0 : !hw.array<1xarray<1xi1>>
}

// CHECK-LABEL: hw.module private @initStruct
hw.module private @initStruct(%clock: i1) {
  %r = seq.firreg %r clock %clock sym @__r__ : !hw.struct<a: i1>

  // CHECK:      %r = sv.reg sym @[[r_sym:[_A-Za-z0-9]+]]
  // CHECK:      sv.ifdef "SYNTHESIS" {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:       %[[RANDOM:.+]] = sv.reg sym @[[RANDOM_SYM:[_A-Za-z0-9]+]]{{.+}}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_" {symbols = []}
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@initStruct::@[[RANDOM_SYM]]>]}
  // CHECK-NEXT:         sv.verbatim "{{[{][{]0[}][}]}}.a = {{[{][{]1[}][}]}}[0];" {symbols = [#hw.innerNameRef<@initStruct::@[[r_sym]]>, #hw.innerNameRef<@initStruct::@[[RANDOM_SYM]]>]}
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL" {symbols = []}
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  hw.output
}

// CHECK-LABEL: issue1594
// Make sure LowerToHW's merging of always blocks kicks in for this example.
hw.module @issue1594(%clock: i1, %reset: i1, %a: i1) -> (b: i1) {
  %true = hw.constant true
  %false = hw.constant false
  %reset_n = sv.wire sym @__issue1594__reset_n  : !hw.inout<i1>
  %0 = sv.read_inout %reset_n : !hw.inout<i1>
  %1 = comb.xor %reset, %true : i1
  sv.assign %reset_n, %1 : i1
  %r = seq.firreg %a clock %clock sym @__r__ reset sync %0, %false : i1
  // CHECK: sv.always posedge %clock
  // CHECK-NOT: sv.always
  // CHECK: hw.output
  hw.output %r : i1
}
