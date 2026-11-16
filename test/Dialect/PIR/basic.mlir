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

  // CHECK: %clock, %clk_val, %clk_def = pir.input : !seq.clock, i1, i1
  %clock, %clk_val, %clk_def = pir.input : !seq.clock, i1, i1

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
  // CHECK: pir.assert_property {{%.+}} initial : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} initial disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} initial on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} initial disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assert_property {{%.+}} strong initial disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  pir.assert_property %clk_p : !pir.clocked_property
  pir.assert_property %clk_p initial : !pir.clocked_property
  pir.assert_property %clk_p disable iff %b : !pir.clocked_property
  pir.assert_property %clk_p initial disable iff %b : !pir.clocked_property
  pir.assert_property %clk_p on %b : !pir.clocked_property
  pir.assert_property %clk_p initial on %b : !pir.clocked_property
  pir.assert_property %clk_p disable iff %b on %b : !pir.clocked_property
  pir.assert_property %clk_p initial disable iff %b on %b : !pir.clocked_property
  pir.assert_property %clk_p strong initial disable iff %b on %b : !pir.clocked_property


  // CHECK: pir.assume_property {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} initial : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} initial disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} initial on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} initial disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.assume_property {{%.+}} strong initial disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  pir.assume_property %clk_p : !pir.clocked_property
  pir.assume_property %clk_p initial : !pir.clocked_property
  pir.assume_property %clk_p disable iff %b : !pir.clocked_property
  pir.assume_property %clk_p initial disable iff %b : !pir.clocked_property
  pir.assume_property %clk_p on %b : !pir.clocked_property
  pir.assume_property %clk_p initial on %b : !pir.clocked_property
  pir.assume_property %clk_p disable iff %b on %b : !pir.clocked_property
  pir.assume_property %clk_p initial disable iff %b on %b : !pir.clocked_property
  pir.assume_property %clk_p strong initial disable iff %b on %b : !pir.clocked_property

  // CHECK: pir.cover_property {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} initial : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} initial disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} initial on {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} initial disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.cover_property {{%.+}} weak initial disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  pir.cover_property %clk_p : !pir.clocked_property
  pir.cover_property %clk_p initial : !pir.clocked_property
  pir.cover_property %clk_p disable iff %b : !pir.clocked_property
  pir.cover_property %clk_p initial disable iff %b : !pir.clocked_property
  pir.cover_property %clk_p on %b : !pir.clocked_property
  pir.cover_property %clk_p initial on %b : !pir.clocked_property
  pir.cover_property %clk_p disable iff %b on %b : !pir.clocked_property
  pir.cover_property %clk_p initial disable iff %b on %b : !pir.clocked_property
  pir.cover_property %clk_p weak initial disable iff %b on %b : !pir.clocked_property

  // CHECK: pir.restrict {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} initial : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} initial disable iff {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} initial on {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} initial disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  // CHECK: pir.restrict {{%.+}} weak initial disable iff {{%.+}} on {{%.+}} : !pir.clocked_property
  pir.restrict %clk_p : !pir.clocked_property
  pir.restrict %clk_p initial : !pir.clocked_property
  pir.restrict %clk_p disable iff %b : !pir.clocked_property
  pir.restrict %clk_p initial disable iff %b : !pir.clocked_property
  pir.restrict %clk_p on %b : !pir.clocked_property
  pir.restrict %clk_p initial on %b : !pir.clocked_property
  pir.restrict %clk_p disable iff %b on %b : !pir.clocked_property
  pir.restrict %clk_p initial disable iff %b on %b : !pir.clocked_property
  pir.restrict %clk_p weak initial disable iff %b on %b : !pir.clocked_property

  //===---------------------------------------------------------------------===//
  // Clocking Ops
  //===----------------------------------------------------------------------===//

  // CHECK: {{%.+}} = pir.gclk_sample posedge %clk_val, %clk_def 
  // CHECK: {{%.+}} = pir.gclk_sample negedge %clk_val, %clk_def 
  // CHECK: {{%.+}} = pir.gclk_sample edge %clk_val, %clk_def 
  %gclk_clk_r = pir.gclk_sample posedge %clk_val, %clk_def 
  %gclk_clk_f = pir.gclk_sample negedge %clk_val, %clk_def 
  %gclk_clk_c = pir.gclk_sample edge %clk_val, %clk_def 

  // CHECK: {{%.+}} = pir.gclk_sample posedge %clk_val 
  // CHECK: {{%.+}} = pir.gclk_sample negedge %clk_val 
  // CHECK: {{%.+}} = pir.gclk_sample edge %clk_val 
  %gclk_clk_r_ = pir.gclk_sample posedge %clk_val 
  %gclk_clk_f_ = pir.gclk_sample negedge %clk_val 
  %gclk_clk_c_ = pir.gclk_sample edge %clk_val 

  // CHECK: {{%.+}} = pir.gclk_future %clk_val
  %gclk_future = pir.gclk_future %clk_val

  // CHECK: {{%.+}} = pir.clocked %clock, %clk_p : !seq.clock, !pir.clocked_property
  // CHECK: {{%.+}} = pir.clocked %clock, %clk_s : !seq.clock, !pir.clocked_sequence
  %clocked_clk_p = pir.clocked %clock, %clk_p : !seq.clock, !pir.clocked_property
  %clocked_clk_s = pir.clocked %clock, %clk_s : !seq.clock, !pir.clocked_sequence

  // CHECK: {{%.+}} = pir.clocked %clk_val, %clk_p : i1, !pir.clocked_property
  // CHECK: {{%.+}} = pir.clocked %clk_val, %clk_s : i1, !pir.clocked_sequence
  %clocked_i1_clk_p = pir.clocked %clk_val, %clk_p : i1, !pir.clocked_property
  %clocked_i1_clk_s = pir.clocked %clk_val, %clk_s : i1, !pir.clocked_sequence

  //===----------------------------------------------------------------------===//
  // Clocked & Simple Sequences
  //===----------------------------------------------------------------------===//
  
  // CHECK: {{%.+}} = pir.initial
  %init = pir.initial

  // CHECK: {{%.+}} = pir.repeat[42 : + 27] %clk_s : !pir.clocked_sequence
  // CHECK: {{%.+}} = pir.repeat[42 : + 27] %s : !pir.sequence
  %r0 = pir.repeat [42 :+ 27] %clk_s : !pir.clocked_sequence
  %r1 = pir.repeat [42 :+ 27] %s : !pir.sequence

  // CHECK: {{%.+}} = pir.repeat[69] %clk_s : !pir.clocked_sequence
  // CHECK: {{%.+}} = pir.repeat[69] %s : !pir.sequence
  %r2 = pir.repeat [69] %clk_s : !pir.clocked_sequence
  %r3 = pir.repeat [69] %s : !pir.sequence

  // CHECK: {{%.+}} = pir.goto_repeat[42 : + 27] %i_0
  // CHECK: {{%.+}} = pir.goto_repeat[69] %i_0
  %r4 = pir.goto_repeat [42 :+ 27] %i_0
  %r5 = pir.goto_repeat [69] %i_0 

  // CHECK: {{%.+}} = pir.non_consecutive_repeat[42 : + 27] %i_0
  // CHECK: {{%.+}} = pir.non_consecutive_repeat[69] %i_0
  %r6 = pir.non_consecutive_repeat [42 :+ 27] %i_0
  %r7 = pir.non_consecutive_repeat [69] %i_0 

  // CHECK: {{%.+}} = pir.delay[42 : + 27] %clk_s
  // CHECK: {{%.+}} = pir.delay[69] %clk_s
  %d0 = pir.delay [42 :+ 27] %clk_s
  %d1 = pir.delay [69] %clk_s

  // CHECK: %s1, %s2, %s3 = pir.input : !pir.sequence, !pir.sequence, !pir.sequence
  // CHECK: %cs1, %cs2, %cs3 = pir.input : !pir.clocked_sequence, !pir.clocked_sequence, !pir.clocked_sequence
  // CHECK: %p1, %p2, %p3 = pir.input : !pir.property, !pir.property, !pir.property
  // CHECK: %cp1, %cp2, %cp3 = pir.input : !pir.clocked_property, !pir.clocked_property, !pir.clocked_property
  %s1, %s2, %s3 = pir.input : !pir.sequence, !pir.sequence, !pir.sequence
  %cs1, %cs2, %cs3 = pir.input : !pir.clocked_sequence, !pir.clocked_sequence, !pir.clocked_sequence
  %p1, %p2, %p3 = pir.input : !pir.property, !pir.property, !pir.property
  %cp1, %cp2, %cp3 = pir.input : !pir.clocked_property, !pir.clocked_property, !pir.clocked_property

  // CHECK: {{%.+}} = pir.concat %s1, %s2, %s3 : !pir.sequence
  // CHECK: {{%.+}} = pir.concat %cs1, %cs2, %cs3 : !pir.clocked_sequence
  %concat_s = pir.concat %s1, %s2, %s3 : !pir.sequence
  %concat_cs = pir.concat %cs1, %cs2, %cs3 : !pir.clocked_sequence

  // CHECK: {{%.+}} = pir.fusion %s1, %s2, %s3 : !pir.sequence
  // CHECK: {{%.+}} = pir.fusion %cs1, %cs2, %cs3 : !pir.clocked_sequence
  %fuse_s = pir.fusion %s1, %s2, %s3 : !pir.sequence
  %fuse_cs = pir.fusion %cs1, %cs2, %cs3 : !pir.clocked_sequence

  // CHECK: {{%.+}} = pir.or %s1, %s2, %s3 : !pir.sequence
  // CHECK: {{%.+}} = pir.or %cs1, %cs2, %cs3 : !pir.clocked_sequence
  // CHECK: {{%.+}} = pir.or %p1, %p2, %p3 : !pir.property
  // CHECK: {{%.+}} = pir.or %cp1, %cp2, %cp3 : !pir.clocked_property
  %or_s = pir.or %s1, %s2, %s3 : !pir.sequence
  %or_cs = pir.or %cs1, %cs2, %cs3 : !pir.clocked_sequence
  %or_p = pir.or %p1, %p2, %p3 : !pir.property
  %or_cp = pir.or %cp1, %cp2, %cp3 : !pir.clocked_property

  // CHECK: {{%.+}} = pir.intersect %s1, %s2, %s3 : !pir.sequence
  // CHECK: {{%.+}} = pir.intersect %cs1, %cs2, %cs3 : !pir.clocked_sequence
  %intersect_s = pir.intersect %s1, %s2, %s3 : !pir.sequence
  %intersect_cs = pir.intersect %cs1, %cs2, %cs3 : !pir.clocked_sequence

  // CHECK: {{%.+}} = pir.first_match %s : !pir.sequence
  // CHECK: {{%.+}} = pir.first_match %clk_s : !pir.clocked_sequence
  %fm_s = pir.first_match %s : !pir.sequence
  %fm_cs = pir.first_match %clk_s : !pir.clocked_sequence  

  // CHECK: {{%.+}} = pir.and %cs1, %cs2, %cs3 : !pir.clocked_sequence
  // CHECK: {{%.+}} = pir.and %cp1, %cp2, %cp3 : !pir.clocked_property
  // CHECK: {{%.+}} = pir.and %p1, %p2, %p3 : !pir.property
  %and_cs = pir.and %cs1, %cs2, %cs3 : !pir.clocked_sequence
  %and_cp = pir.and %cp1, %cp2, %cp3 : !pir.clocked_property
  %and_p = pir.and %p1, %p2, %p3 : !pir.property

  // CHECK: {{%.+}} = pir.throughout %i_0 in %cs1
  %cs_throughout_i0 = pir.throughout %i_0 in %cs1

  // CHECK: {{%.+}} = pir.within %cs1 in %cs2
  %cs1_within_cs2 = pir.within %cs1 in %cs2

  // CHECK: {{%.+}} = pir.not %cp1 : !pir.clocked_property
  // CHECK: {{%.+}} = pir.not %p1 : !pir.property
  %n_cp = pir.not %cp1 : !pir.clocked_property
  %n_p = pir.not %p1 : !pir.property

  // CHECK: {{%.+}} = pir.nexttime[42] %cp1 : !pir.clocked_property
  // CHECK: {{%.+}} = pir.nexttime[42] %p1 : !pir.property
  // CHECK: {{%.+}} = pir.nexttime strong[42] %cp1 : !pir.clocked_property
  // CHECK: {{%.+}} = pir.nexttime strong[42] %p1 : !pir.property
  %next42cp1 = pir.nexttime [42] %cp1 : !pir.clocked_property
  %next42p1 = pir.nexttime [42] %p1 : !pir.property
  %nexts42cp1 = pir.nexttime strong [42] %cp1 : !pir.clocked_property
  %nexts42p1 = pir.nexttime strong [42] %p1 : !pir.property

  // CHECK: {{%.+}} = pir.clk_overlapped_implication %cs1, %cp1
  // CHECK: {{%.+}} = pir.overlapped_implication %s1, %p1
  %cs_clkoi_cp = pir.clk_overlapped_implication %cs1, %cp1
  %s_oi_p = pir.overlapped_implication %s1, %p1

  // CHECK: {{%.+}} = pir.until %cp1, %cp2 : !pir.clocked_property
  // CHECK: {{%.+}} = pir.until %p1, %p2 : !pir.property
  %cp1_until_cp2 = pir.until %cp1, %cp2 : !pir.clocked_property
  %p1_until_p2 = pir.until %p1, %p2 : !pir.property

  // CHECK: {{%.+}} = pir.s_until %cp1, %cp2
  %cp1_suntil_cp2 = pir.s_until %cp1, %cp2 
}
