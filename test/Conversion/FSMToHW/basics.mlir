// RUN: circt-opt -convert-fsm-to-hw -split-input-file %s | FileCheck %s

// CHECK: module  {
// CHECK:   hw.module @foo(%in0: i1, %in1: i1, %in2: i32, %in3: i32, %clk: i1, %rst_n: i1) -> (%out0: i32) {
// CHECK:     %state = sv.reg  : !hw.inout<i1>
// CHECK:     %0 = sv.read_inout %state : !hw.inout<i1>
// CHECK:     %var1 = sv.reg  : !hw.inout<i32>
// CHECK:     %1 = sv.read_inout %var1 : !hw.inout<i32>
// CHECK:     %out = sv.reg  : !hw.inout<i32>
// CHECK:     %2 = sv.read_inout %out : !hw.inout<i32>
// CHECK:     %false = hw.constant false
// CHECK:     %true = hw.constant true
// CHECK:     sv.alwaysff(posedge %clk)  {
// CHECK:       sv.passign %state, %0 : i1
// CHECK:       sv.passign %var1, %1 : i32
// CHECK:       sv.passign %out, %2 : i32
// CHECK:       sv.casez %0 : i1
// CHECK:       case b0: {
// CHECK:         %3 = comb.and %in1, %in0 : i1
// CHECK:         sv.if %3  {
// CHECK:           sv.passign %state, %true : i1
// CHECK:           sv.passign %out, %1 : i32
// CHECK:           sv.passign %var1, %in3 : i32
// CHECK:         }
// CHECK:       }
// CHECK:       case b1: {
// CHECK:         sv.passign %state, %false : i1
// CHECK:         %c11_i32 = hw.constant 11 : i32
// CHECK:         sv.passign %out, %c11_i32 : i32
// CHECK:         sv.passign %var1, %in2 : i32
// CHECK:       }
// CHECK:     }(asyncreset : negedge %rst_n)  {
// CHECK:       sv.passign %state, %false : i1
// CHECK:       %c0_i32 = hw.constant 0 : i32
// CHECK:       sv.passign %var1, %c0_i32 : i32
// CHECK:       %c0_i32_0 = hw.constant 0 : i32
// CHECK:       sv.passign %out, %c0_i32_0 : i32
// CHECK:     }
// CHECK:     hw.output %2 : i32
// CHECK:   }
// CHECK: }

fsm.machine @foo (%run: i1, %stop: i1, %in1 : i32, %in2 : i32) -> (i32) attributes {stateType = i1} {
  %var1 = fsm.variable "var1" : i32
  %out = fsm.variable "out" : i32

  fsm.state "IDLE" entry  {
    fsm.update %var1, %in1 : i32, i32
  } exit  {
  } transitions  {
    fsm.transition @BUSY guard  {
      %cond = and %stop, %run : i1
      fsm.return %cond : i1
    } action  {
      fsm.update %out, %var1 : i32, i32
    }
  }

  fsm.state "BUSY" entry  {
    fsm.update %var1, %in2 : i32, i32
  } exit  {
  } transitions  {
    fsm.transition @IDLE guard  {
    } action  {
      %c11 = constant 11 : i32
      fsm.update %out, %c11 : i32, i32
    }
  }

  fsm.output %out : i32
}
