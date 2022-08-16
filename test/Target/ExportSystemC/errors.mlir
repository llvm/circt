// RUN: circt-translate %s --export-systemc --verify-diagnostics | FileCheck %s

// CHECK: <<UNSUPPORTED OPERATION (hw.module)>>
// expected-error @+1 {{no emission pattern found for 'hw.module'}}
hw.module @notSupported () -> () { }
