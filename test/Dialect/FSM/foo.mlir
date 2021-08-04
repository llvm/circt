// RUN: circt-opt %s | FileCheck %s

// CHECK: fsm.machine
fsm.machine @foo(%i_valid: i1, %i_len: i8) -> (i1)
    attributes {stateType = i2} {

  %o_ready = fsm.variable "o_ready" : i1
  %counter = fsm.variable "counter" : i8

  fsm.state "IDLE" entry  {
    %true = constant true
    fsm.update %o_ready, %true : i1, i1
  } exit  {
    %false = constant false
    fsm.update %o_ready, %false : i1, i1
  } transitions  {
    fsm.transition @BUSY guard  {
      fsm.return %i_valid : i1
    } action  {
      fsm.update %counter, %i_len : i8, i8
    }
  }

  fsm.state "BUSY" entry  {
  } exit  {
  } transitions  {
    fsm.transition @BUSY guard  {
    } action  {
    }
  }

  fsm.output %o_ready : i1
}
