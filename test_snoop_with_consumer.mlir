// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --hw-flatten-io --lower-esi-to-hw | FileCheck %s

// Test that snoops work correctly when the channel has consumers that aren't unwrap operations.
// This tests the case where a snoop needs to observe a channel that's passed to a module instance.
// The snoop lowering should create an unwrap/wrap pair to allow the snoop to observe the signals
// while still providing the channel to the other consumer.

// Test snoop on a channel input that's then passed to output
// CHECK-LABEL: hw.module @TestSnoopPassthrough
// CHECK-SAME: in %chan_in : i32, in %chan_in_valid : i1, in %chan_out_ready : i1
// CHECK-SAME: out chan_in_ready : i1, out chan_out : i32, out chan_out_valid : i1
// CHECK: hw.output %chan_out_ready, %chan_in, %chan_in_valid
hw.module @TestSnoopPassthrough(in %chan_in: !esi.channel<i32>, out chan_out: !esi.channel<i32>) {
  // Snoop on the input channel
  %valid, %ready, %data = esi.snoop.vr %chan_in : !esi.channel<i32>
  
  // Pass it through to output
  hw.output %chan_in : !esi.channel<i32>
}

// Test snoop on a locally-created channel with a non-unwrap consumer
// CHECK-LABEL: hw.module @TestSnoopWithInstanceConsumer
// CHECK-NOT: esi.
hw.module @TestSnoopWithInstanceConsumer(in %in_data: i32, in %in_valid: i1, out out: !esi.channel<i32>) {
  // Create a channel
  %chan, %ready_out = esi.wrap.vr %in_data, %in_valid : i32
  
  // Snoop on it
  %valid, %ready_snoop, %data = esi.snoop.vr %chan : !esi.channel<i32>
  
  // Pass it through to another module (consumer that's not an unwrap)
  %out = hw.instance "passthrough" @TestSnoopPassthrough(chan_in: %chan: !esi.channel<i32>) -> (chan_out: !esi.channel<i32>)
  
  hw.output %out : !esi.channel<i32>
}
