// RUN: circt-opt %s | FileCheck --check-prefix=BASIC %s
// RUN: circt-opt -fsm-print-state-graph %s -o %t 2>&1 | FileCheck --check-prefix=PRINT %s

// BASIC: fsm.machine @foo([[ARG0:%.+]]: i1, [[ARG1:%.+]]: i8) -> i1 attributes {stateType = i2} {
// BASIC:   %counter = fsm.variable "counter" {initValue = 0 : i8} : i8
// BASIC:   fsm.state "IDLE" output  {
// BASIC:     %true = constant true
// BASIC:     fsm.output %true : i1
// BASIC:   } transitions  {
// BASIC:     fsm.transition @BUSY guard  {
// BASIC:       fsm.return [[ARG0]] : i1
// BASIC:     } action  {
// BASIC:       fsm.update %counter, [[ARG1]] : i8
// BASIC:     }
// BASIC:     fsm.transition @IDLE guard  {
// BASIC:     } action  {
// BASIC:     }
// BASIC:   }
// BASIC:   fsm.state "BUSY" output  {
// BASIC:     %false = constant false
// BASIC:     fsm.output %false : i1
// BASIC:   } transitions  {
// BASIC:     fsm.transition @BUSY guard  {
// BASIC:     } action  {
// BASIC:     }
// BASIC:   }
// BASIC: }

// BASIC: func @bar() {
// BASIC:   %foo_inst = fsm.instance "foo_inst" @foo
// BASIC:   %true = constant true
// BASIC:   %c16_i8 = constant 16 : i8
// BASIC:   [[VAL0:%.+]] = fsm.trigger %foo_inst(%true, %c16_i8) : (i1, i8) -> i1
// BASIC:   %false = constant false
// BASIC:   %c0_i8 = constant 0 : i8
// BASIC:   [[VAL1:%.+]] = fsm.trigger %foo_inst(%false, %c0_i8) : (i1, i8) -> i1
// BASIC:   return
// BASIC: }

// BASIC: hw.module @qux(%clock: i1, %reset: i1) {
// BASIC:   %true = hw.constant true
// BASIC:   %c16_i8 = hw.constant 16 : i8
// BASIC:   [[VAL:%.+]] = fsm.hw_instance "foo_inst" @foo(%true, %c16_i8) : (i1, i8) -> i1, clock %clock : i1, reset %reset : i1
// BASIC:   hw.output
// BASIC: }

// PRINT: digraph "foo" {
// PRINT:   label="foo";
// PRINT:   [[IDLE:Node0x.+]] [shape=record,label="{IDLE}"];
// PRINT:   [[IDLE]] -> [[BUSY:Node0x.+]];
// PRINT:   [[IDLE]] -> [[IDLE]];
// PRINT:   [[BUSY]] [shape=record,label="{BUSY}"];
// PRINT:   [[BUSY]] -> [[BUSY]];
// PRINT: }

fsm.machine @foo(%i_valid: i1, %i_len: i8) -> i1 attributes {stateType = i2} {
  %counter = fsm.variable "counter" {initValue = 0 : i8} : i8

  fsm.state "IDLE" output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    fsm.transition @BUSY guard  {
      fsm.return %i_valid : i1
    } action  {
      fsm.update %counter, %i_len : i8
    }
    fsm.transition @IDLE guard  {
    } action  {
    }
  }

  fsm.state "BUSY" output  {
    %false = constant false
    fsm.output %false : i1
  } transitions  {
    fsm.transition @BUSY guard  {
    } action  {
    }
  }
}

func @bar() {
  %foo_inst = fsm.instance "foo_inst" @foo

  %i_valid0 = constant true
  %i_len0 = constant 16 : i8
  %o_ready0 = fsm.trigger %foo_inst(%i_valid0, %i_len0) : (i1, i8) -> i1

  %i_valid1 = constant false
  %i_len1 = constant 0 : i8
  %o_ready1 = fsm.trigger %foo_inst(%i_valid1, %i_len1) : (i1, i8) -> i1
  return
}

hw.module @qux(%clock: i1, %reset: i1) {
  %i_valid = hw.constant true
  %i_len = hw.constant 16 : i8
  %o_ready = fsm.hw_instance "foo_inst" @foo(%i_valid, %i_len) : (i1, i8) -> i1, clock %clock : i1, reset %reset : i1
}
