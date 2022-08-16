// RUN: circt-translate %s --export-systemc | FileCheck %s

// CHECK-LABEL: // stdout.h
// CHECK-NEXT: #ifndef STDOUT_H
// CHECK-NEXT: #define STDOUT_H

// CHECK: #include <systemc>

// CHECK-LABEL: SC_MODULE(basic) {
// CHECK-NEXT: };
systemc.module @basic () { }

// CHECK: #endif // STDOUT_H
