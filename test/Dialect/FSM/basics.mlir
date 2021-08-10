// RUN: circt-opt %s | FileCheck %s

// CHECK: fsm.machine @foo
fsm.machine @foo(%i_valid: i1, %i_len: i8) -> i1 attributes {stateType = i2} {

  // CHECK: %o_ready = fsm.variable "o_ready"
  // CHECK: %counter = fsm.variable "counter"
  %o_ready = fsm.variable "o_ready" {initValue = false} : i1
  %counter = fsm.variable "counter" {initValue = 0 : i8} : i8

  // CHECK: fsm.state "IDLE"
  fsm.state "IDLE" entry  {
    %true = constant true
    // CHECK: fsm.update %o_ready, %true : i1
    fsm.update %o_ready, %true : i1
  } exit  {
    %false = constant false
    // CHECK: fsm.update %o_ready, %false : i1
    fsm.update %o_ready, %false : i1
  } transitions  {
    // CHECK: fsm.transition @BUSY
    fsm.transition @BUSY guard  {
      fsm.return %i_valid : i1
    } action  {
      fsm.update %counter, %i_len : i8
    }
  }

  fsm.state "BUSY" entry  {
  } exit  {
  } transitions  {
    fsm.transition @BUSY guard  {
    } action  {
    }
  }

  // CHECK: fsm.output %o_ready : i1
  fsm.output %o_ready : i1
}

func @bar() {
  // CHECK: %foo_inst = fsm.instance "foo_inst" @foo
  %foo_inst = fsm.instance "foo_inst" @foo

  %i_valid0 = constant true
  %i_len0 = constant 16 : i8
  // CHECK: fsm.trigger %foo_inst
  %o_ready0 = fsm.trigger %foo_inst(%i_valid0, %i_len0) : (i1, i8) -> i1

  %i_valid1 = constant false
  %i_len1 = constant 0 : i8
  // CHECK: fsm.trigger %foo_inst
  %o_ready1 = fsm.trigger %foo_inst(%i_valid1, %i_len1) : (i1, i8) -> i1
  return
}

hw.module @qux() {
  %i_valid = hw.constant true
  %i_len = hw.constant 16 : i8
  // CHECK: fsm.hw_instance "foo_inst" @foo
  %o_ready = fsm.hw_instance "foo_inst" @foo(%i_valid, %i_len) : (i1, i8) -> i1
}
