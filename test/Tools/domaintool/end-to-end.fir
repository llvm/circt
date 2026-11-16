; RUN: firtool %s --domain-mode=check --strip-domain=false --output-final-mlir=%t.mlir -o /dev/null
; RUN: domaintool %t.mlir --module Foo --domain ClockDomain,A,A,synchronous --domain ClockDomain,B,A,synchronous --assign 0 --assign 1 | FileCheck %s
; RUN: not domaintool %t.mlir --module Foo --domain ClockDomain,A,A,synchronous --domain ClockDomain,B,A,rational --assign 0 --assign 1 2>&1 | FileCheck %s --check-prefix=ERROR

FIRRTL version 7.0.0
circuit Foo:
  domain ClockDomain:
    name: String
    source: String
    relationship: String

  public module Foo:
    input A: Domain of ClockDomain
    input B: Domain of ClockDomain
    input a: UInt<1> domains [A]
    input b: UInt<1> domains [B]

    wire w: Domain of ClockDomain
    propassert prop_eq(w.source, A.name), "B must refer to A"
    propassert prop_eq(w.relationship, String("synchronous")), "B must be synchronous with A"
    domain_define w = B

; CHECK:      {
; CHECK-NEXT:   "clocks": [
; CHECK-NEXT:     {
; CHECK-NEXT:       "name_pattern": "A",
; CHECK-NEXT:       "define_period": "A_PERIOD",
; CHECK-NEXT:       "clock_relationships": []
; CHECK-NEXT:     },
; CHECK-NEXT:     {
; CHECK-NEXT:       "name_pattern": "B",
; CHECK-NEXT:       "define_period": "B_PERIOD",
; CHECK-NEXT:       "clock_relationships": [
; CHECK-NEXT:         {
; CHECK-NEXT:           "name_pattern": "A",
; CHECK-NEXT:           "relationship": "sync"
; CHECK-NEXT:         }
; CHECK-NEXT:       ]
; CHECK-NEXT:     }
; CHECK-NEXT:   ],
; CHECK-NEXT:   "static_ports": [],
; CHECK-NEXT:   "asynchronous_ports": [],
; CHECK-NEXT:   "synchronous_ports": [
; CHECK-NEXT:     {
; CHECK-NEXT:       "name_pattern": "A",
; CHECK-NEXT:       "port_patterns": [
; CHECK-NEXT:         "a"
; CHECK-NEXT:       ],
; CHECK-NEXT:       "comment": null
; CHECK-NEXT:     },
; CHECK-NEXT:     {
; CHECK-NEXT:       "name_pattern": "B",
; CHECK-NEXT:       "port_patterns": [
; CHECK-NEXT:         "b"
; CHECK-NEXT:       ],
; CHECK-NEXT:       "comment": null
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT: }
; CHECK-NOT: {{.}}

; ERROR: error: OM property assertion failed: B must be synchronous with A
