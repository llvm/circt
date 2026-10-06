// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Top-level ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pir.cell @test
pir.cell @test {} {

  // CHECK: {{%.+}}, {{%.+}}, {{%.+}} = pir.input : i1, i2, i3
  %i_0, %i_1, %i_2 = pir.input : i1, i2, i3

  // CHECK: %b, %s, %p = pir.input : i1, !pir.sequence, !pir.property
  %b, %s, %p = pir.input : i1, !pir.sequence, !pir.property
  // CHECK: %clk_s, %clk_p = pir.input : !pir.clocked_sequence, !pir.clocked_property
  %clk_s, %clk_p = pir.input : !pir.clocked_sequence, !pir.clocked_property

  //===----------------------------------------------------------------------===//
  // Type Conversions
  //===----------------------------------------------------------------------===//

  // CHECK: {{%.+}} = pir.bool_to_clocked_seq {{%.+}}
  // CHECK: {{%.+}} = pir.seq_to_clocked_seq {{%.+}}
  // CHECK: {{%.+}} = pir.bool_to_clocked_prop {{%.+}}
  // CHECK: {{%.+}} = pir.clocked_seq_to_clocked_prop {{%.+}}
  %csb = pir.bool_to_clocked_seq %b 
  %css = pir.seq_to_clocked_seq %s
  %cpb = pir.bool_to_clocked_prop %b
  %cpcs = pir.clocked_seq_to_clocked_prop %clk_s

  //===---------------------------------------------------------------------===//
  // Assertions
  //===----------------------------------------------------------------------===//

  // CHECK: pir.assert_property {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} always : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} initial : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} always disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} always on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} always disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  pir.assert_property %clk_p : !pir.clocked_property
  pir.assert_property %clk_p always : !pir.clocked_property
  pir.assert_property %clk_p initial : !pir.clocked_property
  pir.assert_property %clk_p disable iff %b : !pir.clocked_property
  pir.assert_property %clk_p always disable iff %b : !pir.clocked_property
  pir.assert_property %clk_p on %b : !pir.clocked_property
  pir.assert_property %clk_p always on %b : !pir.clocked_property
  pir.assert_property %clk_p disable iff %b on %b : !pir.clocked_property
  pir.assert_property %clk_p always disable iff %b on %b : !pir.clocked_property

  // CHECK: pir.assume_property {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} always : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} initial : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} always disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} always on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} always disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  pir.assume_property %clk_p : !pir.clocked_property
  pir.assume_property %clk_p always : !pir.clocked_property
  pir.assume_property %clk_p initial : !pir.clocked_property
  pir.assume_property %clk_p disable iff %b : !pir.clocked_property
  pir.assume_property %clk_p always disable iff %b : !pir.clocked_property
  pir.assume_property %clk_p on %b : !pir.clocked_property
  pir.assume_property %clk_p always on %b : !pir.clocked_property
  pir.assume_property %clk_p disable iff %b on %b : !pir.clocked_property
  pir.assume_property %clk_p always disable iff %b on %b : !pir.clocked_property

  // CHECK: pir.cover_property {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} always : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} initial : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} always disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} always on {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} always disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  pir.cover_property %clk_p : !pir.clocked_property
  pir.cover_property %clk_p always : !pir.clocked_property
  pir.cover_property %clk_p initial : !pir.clocked_property
  pir.cover_property %clk_p disable iff %b : !pir.clocked_property
  pir.cover_property %clk_p always disable iff %b : !pir.clocked_property
  pir.cover_property %clk_p on %b : !pir.clocked_property
  pir.cover_property %clk_p always on %b : !pir.clocked_property
  pir.cover_property %clk_p disable iff %b on %b : !pir.clocked_property
  pir.cover_property %clk_p always disable iff %b on %b : !pir.clocked_property

  // CHECK: pir.restrict {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} always : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} initial : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} always disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} always on {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} always disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  pir.restrict %clk_p : !pir.clocked_property
  pir.restrict %clk_p always : !pir.clocked_property
  pir.restrict %clk_p initial : !pir.clocked_property
  pir.restrict %clk_p disable iff %b : !pir.clocked_property
  pir.restrict %clk_p always disable iff %b : !pir.clocked_property
  pir.restrict %clk_p on %b : !pir.clocked_property
  pir.restrict %clk_p always on %b : !pir.clocked_property
  pir.restrict %clk_p disable iff %b on %b : !pir.clocked_property
  pir.restrict %clk_p always disable iff %b on %b : !pir.clocked_property

}
