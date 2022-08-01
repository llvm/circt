// RUN: circt-translate %s --export-systemc | FileCheck %s

// CHECK-LABEL: SC_MODULE(basic) {
// CHECK-NEXT: };
systemc.module @basic () { }
