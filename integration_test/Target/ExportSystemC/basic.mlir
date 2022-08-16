// REQUIRES: clang-tidy, systemc
// RUN: circt-translate %s --export-systemc > %t.cpp
// RUN: clang-tidy --extra-arg=-frtti %t.cpp

emitc.include <"systemc">

systemc.module @module (%port0: !systemc.in<i1>, %port1: !systemc.inout<i64>, %port2: !systemc.out<i512>, %port3: !systemc.out<i1024>) { }
