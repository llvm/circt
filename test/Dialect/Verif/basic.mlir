// RUN: circt-opt %s | circt-opt | FileCheck %s

%true = hw.constant true
%s = unrealized_conversion_cast to !ltl.sequence
%p = unrealized_conversion_cast to !ltl.property

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

// CHECK: verif.assert {{%.+}} : i1
// CHECK: verif.assert {{%.+}} label "foo1" : i1
// CHECK: verif.assert {{%.+}} : !ltl.sequence
// CHECK: verif.assert {{%.+}} : !ltl.property
verif.assert %true : i1
verif.assert %true label "foo1" : i1
verif.assert %s : !ltl.sequence
verif.assert %p : !ltl.property

// CHECK: verif.assume {{%.+}} : i1
// CHECK: verif.assume {{%.+}} label "foo2" : i1
// CHECK: verif.assume {{%.+}} : !ltl.sequence
// CHECK: verif.assume {{%.+}} : !ltl.property
verif.assume %true : i1
verif.assume %true label "foo2" : i1
verif.assume %s : !ltl.sequence
verif.assume %p : !ltl.property

// CHECK: verif.cover {{%.+}} : i1
// CHECK: verif.cover {{%.+}} label "foo3" : i1
// CHECK: verif.cover {{%.+}} : !ltl.sequence
// CHECK: verif.cover {{%.+}} : !ltl.property
verif.cover %true : i1
verif.cover %true label "foo3" : i1
verif.cover %s : !ltl.sequence
verif.cover %p : !ltl.property
