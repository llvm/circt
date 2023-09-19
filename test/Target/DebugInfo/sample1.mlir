// RUN: circt-translate %s --emit-hgldd

#loc2 = loc("sample1.scala":22:7)
#loc3 = loc("sample1.scala":23:7)
#loc6 = loc("sample1.scala":4:7)
#loc7 = loc("sample1.scala":5:7)
module {
  hw.module private @Bob(in %in: i16 loc(#loc2), out out: i16 loc(#loc3)) {
    %c-1_i16 = hw.constant -1 : i16 loc(#loc4)
    %0 = comb.xor bin %in, %c-1_i16 : i16 loc(#loc4)
    %x = hw.wire %0 sym @__Bob__x  : i16 loc(#loc4)
    hw.output %x : i16 loc(#loc1)
  } loc(#loc1)
  hw.module private @Bob_1(in %in: i16 loc(#loc2), out out: i16 loc(#loc3)) {
    %c-1_i16 = hw.constant -1 : i16 loc(#loc4)
    %0 = comb.xor bin %in, %c-1_i16 : i16 loc(#loc4)
    %x = hw.wire %0 sym @__Bob_1__x  : i16 loc(#loc4)
    hw.output %x : i16 loc(#loc1)
  } loc(#loc1)
  hw.module @Top(in %in: i16 loc(#loc6), out out: i16 loc(#loc7)) {
    %b0.out = hw.instance "b0" @Bob(in: %in: i16) -> (out: i16) loc(#loc8)
    %b1.out = hw.instance "b1" @Bob_1(in: %b0.out: i16) -> (out: i16) loc(#loc8)
    hw.output %b1.out : i16 loc(#loc5)
  } loc(#loc5)
} loc(#loc)
#loc = loc("test/Target/DebugInfo/sample1.fir":0:0)
#loc1 = loc("sample1.scala":21:7)
#loc4 = loc("sample1.scala":26:11)
#loc5 = loc("sample1.scala":3:7)
#loc8 = loc("sample1.scala":19:17)
