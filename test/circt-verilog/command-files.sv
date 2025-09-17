// RUN: circt-verilog -C %S/include/filelist.f %s | FileCheck %s
// REQUIRES: slang
// Internal issue in Slang about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK: hw.module @library_module
