// RUN: circt-opt --sroa %s | FileCheck %s

// CHECK-LABEL: @integers
hw.module @integers(in %in : i4, in %v2 : i2, out out : i4) {
  %time = llhd.constant_time <0ns, 0d, 1e>

  // CHECK-NOT: llhd.sig{{.*}} : i4
  // CHECK: %[[SIG0:.*]] = llhd.sig {{.*}} : i1
  // CHECK: %[[SIG1:.*]] = llhd.sig {{.*}} : i2
  // CHECK: %[[SIG3:.*]] = llhd.sig {{.*}} : i1
  %sig = llhd.sig name "test_sig" %in : i4

  // CHECK-NOT: llhd.prb %{{.*}} : i4
  // CHECK: %[[PRB0:.*]] = llhd.prb %[[SIG0]]
  // CHECK: %[[PRB1:.*]] = llhd.prb %[[SIG1]]
  // CHECK: %[[PRB3:.*]] = llhd.prb %[[SIG3]]
  // CHECK: %[[CONCAT:.*]] = comb.concat %[[PRB3]], %[[PRB1]], %[[PRB0]]
  %prb = llhd.prb %sig : i4

  // CHECK: %[[EXT0:.*]] = comb.extract %in from 0
  // CHECK: llhd.drv %[[SIG0]], %[[EXT0]]
  // CHECK: %[[EXT1:.*]] = comb.extract %in from 1
  // CHECK: llhd.drv %[[SIG1]], %[[EXT1]]
  // CHECK: %[[EXT3:.*]] = comb.extract %in from 3
  // CHECK: llhd.drv %[[SIG3]], %[[EXT3]]
  llhd.drv %sig, %in after %time : i4

  // Slice extract (middle 2 bits: 1 and 2)
  // CHECK-NOT: llhd.sig.extract
  %c1 = hw.constant 1 : i2
  %slice = llhd.sig.extract %sig from %c1 : !llhd.ref<i4> -> !llhd.ref<i2>

  // Writing to slice
  // CHECK: %[[V2_0:.*]] = comb.extract %v2 from 0
  // CHECK: llhd.drv %[[SIG1]], %[[V2_0]]
  llhd.drv %slice, %v2 after %time : i2

  // Reading from slice
  // CHECK: %[[SLICE_PRB0:.*]] = llhd.prb %[[SIG1]]
  %slice_prb = llhd.prb %slice : i2

  hw.output %prb : i4
}

// Checks that SROA does not destructure a simple multi-bit signal.
// CHECK-LABEL: @simple_add
// CHECK-NOT: i1
hw.module @simple_add(in %in : i4, out out : i4) {
  %sig = llhd.sig name "test_sig" %in : i4

  %prb = llhd.prb %sig : i4

  %add = comb.add %prb, %prb : i4

  hw.output %add : i4
}
