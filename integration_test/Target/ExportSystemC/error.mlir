// REQUIRES: clang-tidy
// RUN: cat %s > %t.cpp
// RUN: (clang-tidy %t.cpp 2>&1 || exit 0) | FileCheck %s

// CHECK: expected ';' after class
// CHECK: Found compiler error(s).
class TestClass { }
