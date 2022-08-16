// REQUIRES: clang-tidy, systemc
// RUN: circt-translate %s --export-systemc > %t.cpp
// RUN: clang-tidy --extra-arg=-frtti %t.cpp

systemc.module @module () { }
