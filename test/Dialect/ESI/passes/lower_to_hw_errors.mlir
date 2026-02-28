// RUN: circt-opt --lower-esi-to-hw %s --verify-diagnostics --split-input-file

// Issue #8219
hw.module @CoerceBundleTransform(in %b_out_result : !esi.channel<i16>, out b_in_resp : !esi.channel<i8>) {
  // expected-error@+1 {{failed to legalize operation 'esi.unwrap.vr' that was explicitly marked illegal}}
  %rawOutput, %valid = esi.unwrap.vr %b_out_result, %ready : i16
  %0 = comb.extract %rawOutput from 0 : (i16) -> i8
  %chanOutput, %ready = esi.wrap.vr %0, %valid : i8
  hw.output %chanOutput : !esi.channel<i8>
}
