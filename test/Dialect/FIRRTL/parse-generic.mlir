// RUN: circt-opt -mlir-print-op-generic %s | FileCheck %s

// This test checks nitty storage details that may be glossed over when using
// custom MLIR printer-parsers.

// Both domain information and port annotations should, if absent, parse into
// empty array attributes.
//
// CHECK-LABEL: "firrtl.circuit"{{.*}}"PortAnnotationsAndDomains"
firrtl.circuit "PortAnnotationsAndDomains" {
  // CHECK:      "firrtl.module"
  // CHECK-SAME:   domainInfo = []
  // CHECK-SAME:   portAnnotations = []
  firrtl.module @PortAnnotationsAndDomains(
    in %a: !firrtl.uint<1>
  ) {}
}
