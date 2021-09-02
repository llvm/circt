// RUN: circt-opt %s | FileCheck %s

// CHECK: fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
// CHECK:   %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16
// CHECK:   fsm.state "IDLE" output  {
// CHECK:     %true = constant true
// CHECK:     fsm.output %true : i1
// CHECK:   } transitions  {
// CHECK:     fsm.transition @BUSY guard  {
// CHECK:       fsm.return %arg0
// CHECK:     } action  {
// CHECK:       %c256_i16 = constant 256 : i16
// CHECK:       fsm.update %cnt, %c256_i16 : i16
// CHECK:     }
// CHECK:   }
// CHECK:   fsm.state "BUSY" output  {
// CHECK:     %false = constant false
// CHECK:     fsm.output %false : i1
// CHECK:   } transitions  {
// CHECK:     fsm.transition @BUSY guard  {
// CHECK:       %c0_i16 = constant 0 : i16
// CHECK:       %0 = cmpi ne, %cnt, %c0_i16 : i16
// CHECK:       fsm.return %0
// CHECK:     } action  {
// CHECK:       %c1_i16 = constant 1 : i16
// CHECK:       %0 = subi %cnt, %c1_i16 : i16
// CHECK:       fsm.update %cnt, %0 : i16
// CHECK:     }
// CHECK:     fsm.transition @IDLE guard  {
// CHECK:       %c0_i16 = constant 0 : i16
// CHECK:       %0 = cmpi eq, %cnt, %c0_i16 : i16
// CHECK:       fsm.return %0
// CHECK:     } action  {
// CHECK:     }
// CHECK:   }
// CHECK: }
// CHECK: hw.module @bar(%clk: i1, %rst_n: i1) {
// CHECK:   %true = hw.constant true
// CHECK:   %0 = fsm.hw_instance "foo_inst" @foo(%true) : (i1) -> i1, clock %clk : i1, reset %rst_n : i1
// CHECK:   hw.output
// CHECK: }
// CHECK: func @qux() {
// CHECK:   %foo_inst = fsm.instance "foo_inst" @foo
// CHECK:   %true = constant true
// CHECK:   %0 = fsm.trigger %foo_inst(%true) : (i1) -> i1
// CHECK:   %false = constant false
// CHECK:   %1 = fsm.trigger %foo_inst(%false) : (i1) -> i1
// CHECK:   return
// CHECK: }

fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state "IDLE" output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    // Transit to BUSY when `arg0` is true.
    fsm.transition @BUSY guard  {
      fsm.return %arg0
    } action  {
      %c256_i16 = constant 256 : i16
      fsm.update %cnt, %c256_i16 : i16
    }
  }

  fsm.state "BUSY" output  {
    %false = constant false
    fsm.output %false : i1
  } transitions  {
    // Transit to BUSY itself when `cnt` is not equal to zero. Meanwhile,
    // decrease `cnt` by one.
    fsm.transition @BUSY guard  {
      %c0_i16 = constant 0 : i16
      %0 = cmpi ne, %cnt, %c0_i16 : i16
      fsm.return %0
    } action  {
      %c1_i16 = constant 1 : i16
      %0 = subi %cnt, %c1_i16 : i16
      fsm.update %cnt, %0 : i16
    }
    // Transit back to IDLE when `cnt` is equal to zero.
    fsm.transition @IDLE guard  {
      %c0_i16 = constant 0 : i16
      %0 = cmpi eq, %cnt, %c0_i16 : i16
      fsm.return %0
    } action  {
    }
  }
}

// Hardware-style instantiation.
hw.module @bar(%clk: i1, %rst_n: i1) {
  %in = hw.constant true
  %out = fsm.hw_instance "foo_inst" @foo(%in) : (i1) -> i1, clock %clk : i1, reset %rst_n : i1
}

// Software-style instantiation and triggering.
func @qux() {
  %foo_inst = fsm.instance "foo_inst" @foo
  %in0 = constant true
  %out0 = fsm.trigger %foo_inst(%in0) : (i1) -> i1
  %in1 = constant false
  %out1 = fsm.trigger %foo_inst(%in1) : (i1) -> i1
  return
}
