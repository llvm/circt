// RUN: circt-opt %s -split-input-file -verify-diagnostics

// Test missing initial state.

// expected-error @+1 {{'fsm.machine' op initial state 'IDLE' was not defined in the machine}}
fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {}
