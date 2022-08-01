// REQUIRES: clang-tidy
// REQUIRES: systemc
// RUN: circt-translate %s --export-systemc > %t.cpp
// RUN: clang-tidy --extra-arg=-frtti --config-file=%S.clang-tidy %t.cpp

systemc.module @module () { }
