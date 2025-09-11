// RUN: circt-verilog -C %S/include/filelist.f %s | FileCheck %s
// REQUIRES: slang

// CHECK: hw.module @library_module
