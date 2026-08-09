// RUN: circt-opt %s | circt-opt | FileCheck %s

%true = hw.constant true
%c0_i8 = hw.constant 0 : i8
%clk = hw.constant true

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: unrealized_conversion_cast to !ltl.sequence
// CHECK: unrealized_conversion_cast to !ltl.property
%s = unrealized_conversion_cast to !ltl.sequence
%p = unrealized_conversion_cast to !ltl.property

//===----------------------------------------------------------------------===//
// Generic
//===----------------------------------------------------------------------===//

// CHECK-NEXT: ltl.and {{%.+}}, {{%.+}} : i1, i1
// CHECK-NEXT: ltl.and {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence
// CHECK-NEXT: ltl.and {{%.+}}, {{%.+}} : !ltl.property, !ltl.property
ltl.and %true, %true : i1, i1
ltl.and %s, %s : !ltl.sequence, !ltl.sequence
ltl.and %p, %p : !ltl.property, !ltl.property

// CHECK-NEXT: ltl.or {{%.+}}, {{%.+}} : i1, i1
// CHECK-NEXT: ltl.or {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence
// CHECK-NEXT: ltl.or {{%.+}}, {{%.+}} : !ltl.property, !ltl.property
ltl.or %true, %true : i1, i1
ltl.or %s, %s : !ltl.sequence, !ltl.sequence
ltl.or %p, %p : !ltl.property, !ltl.property

// Type inference. `unrealized_conversion_cast` used to detect unexpected return
// types on `ltl.and`.
%s0 = ltl.and %true, %true : i1, i1
%s1 = ltl.and %true, %s : i1, !ltl.sequence
%s2 = ltl.and %s, %true : !ltl.sequence, i1
%p0 = ltl.and %true, %p : i1, !ltl.property
%p1 = ltl.and %p, %true : !ltl.property, i1
%p2 = ltl.and %s, %p : !ltl.sequence, !ltl.property
%p3 = ltl.and %p, %s : !ltl.property, !ltl.sequence
unrealized_conversion_cast %s0 : i1 to index
unrealized_conversion_cast %s1 : !ltl.sequence to index
unrealized_conversion_cast %s2 : !ltl.sequence to index
unrealized_conversion_cast %p0 : !ltl.property to index
unrealized_conversion_cast %p1 : !ltl.property to index
unrealized_conversion_cast %p2 : !ltl.property to index
unrealized_conversion_cast %p3 : !ltl.property to index

//===----------------------------------------------------------------------===//
// Sequences
//===----------------------------------------------------------------------===//

// CHECK: ltl.clocked_delay {{%.+}}, posedge {{%.+}}, 0 : !ltl.sequence
// CHECK: ltl.clocked_delay {{%.+}}, posedge {{%.+}}, 42, 1337 : !ltl.sequence
ltl.clocked_delay %s, posedge %clk, 0 : !ltl.sequence
ltl.clocked_delay %s, posedge %clk, 42, 1337 : !ltl.sequence

// CHECK: ltl.concat {{%.+}} : !ltl.sequence
// CHECK: ltl.concat {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence
// CHECK: ltl.concat {{%.+}}, {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence, !ltl.sequence
ltl.concat %s : !ltl.sequence
ltl.concat %s, %s : !ltl.sequence, !ltl.sequence
ltl.concat %s, %s, %s : !ltl.sequence, !ltl.sequence, !ltl.sequence

// CHECK: ltl.clocked_repeat {{%.+}}, posedge {{%.+}}, 2, 2 : !ltl.sequence
%cr = ltl.clocked_repeat %s, posedge %true, 2, 2 : !ltl.sequence

// CHECK: ltl.clocked_goto_repeat {{%.+}}, posedge {{%.+}}, 1, 2 : !ltl.sequence
%cg = ltl.clocked_goto_repeat %s, posedge %true, 1, 2 : !ltl.sequence

// CHECK: ltl.clocked_non_consecutive_repeat {{%.+}}, posedge {{%.+}}, 1, 2 : !ltl.sequence
%cn = ltl.clocked_non_consecutive_repeat %s, posedge %true, 1, 2 : !ltl.sequence

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

// CHECK: ltl.boolean_constant true
%bc = ltl.boolean_constant true
unrealized_conversion_cast %bc : !ltl.property to index

// CHECK: ltl.not {{%.+}} : i1
// CHECK: ltl.not {{%.+}} : !ltl.sequence
// CHECK: ltl.not {{%.+}} : !ltl.property
ltl.not %true : i1
ltl.not %s : !ltl.sequence
ltl.not %p : !ltl.property

// CHECK: ltl.implication {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.property
ltl.implication %s, %p : !ltl.sequence, !ltl.property

// CHECK: ltl.clocked_until {{%.+}}, posedge {{%.+}}, {{%.+}} : !ltl.property, !ltl.property
%cu = ltl.clocked_until %p, posedge %true, %p : !ltl.property, !ltl.property

// CHECK: ltl.clocked_eventually {{%.+}}, posedge {{%.+}} : !ltl.property
%ce = ltl.clocked_eventually %p, posedge %true : !ltl.property

// CHECK: ltl.clocked_past {{%.+}}, 5 clk {{%.+}} : i8
ltl.clocked_past %c0_i8, 5 clk %true : i8
