// RUN: circt-opt %s --llhd-mem2reg --canonicalize | FileCheck %s --check-prefix=MEM
// RUN: circt-opt %s --sroa | FileCheck %s --check-prefix=KEEP
// RUN: circt-opt %s --llhd-combine-drives | FileCheck %s --check-prefix=KEEP
// RUN: circt-opt %s --llhd-sig2reg --canonicalize | FileCheck %s --check-prefix=REG

// Drive and probe a slice spanning two fields. Mem2Reg must flatten before
// injecting the bits and reconstruct the aggregate afterwards. SROA and
// CombineDrives must not interpret bit offsets as field indices.
// MEM-LABEL: @struct_slice
// KEEP-LABEL: @struct_slice
// REG-LABEL: @struct_slice
hw.module @struct_slice(in %init: !hw.struct<hi: i8, lo: i8>, in %index: i4, in %value: i6, out result: !hw.struct<hi: i8, lo: i8>, out slice: i6) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %init : !hw.struct<hi: i8, lo: i8>
  // KEEP: llhd.combinational
  %result, %read = llhd.combinational -> !hw.struct<hi: i8, lo: i8>, i6 {
    // KEEP: [[SLICE:%.+]] = llhd.sig.extract %sig from %index : <!hw.struct<hi: i8, lo: i8>> -> <i6>
    %slice = llhd.sig.extract %sig from %index : <!hw.struct<hi: i8, lo: i8>> -> <i6>
    // KEEP: llhd.drv [[SLICE]], %value
    llhd.drv %slice, %value after %time : i6
    // MEM: hw.bitcast {{.*}} : (!hw.struct<hi: i8, lo: i8>) -> i16
    // MEM: comb.shl
    // MEM: [[STRUCT:%.+]] = hw.bitcast {{.*}} : (i16) -> !hw.struct<hi: i8, lo: i8>
    // MEM: comb.shru
    // MEM: llhd.yield [[STRUCT]], {{%.+}} : !hw.struct<hi: i8, lo: i8>, i6
    %read = llhd.prb %slice : i6
    %result = llhd.prb %sig : !hw.struct<hi: i8, lo: i8>
    llhd.yield %result, %read : !hw.struct<hi: i8, lo: i8>, i6
  }
  hw.output %result, %read : !hw.struct<hi: i8, lo: i8>, i6
}

// The constant slice deliberately starts at bit 4, beyond the field count.
// A neighboring named-field drive prevents treating the whole signal as a
// collection of bit-indexed slices in CombineDrives.
// MEM-LABEL: @mixed_projections
// KEEP-LABEL: @mixed_projections
// REG-LABEL: @mixed_projections
hw.module @mixed_projections(in %init: !hw.struct<hi: i8, lo: i8>, in %value: i4, in %high: i8, out result: !hw.struct<hi: i8, lo: i8>) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  %four = hw.constant 4 : i4
  // KEEP: %sig = llhd.sig
  %sig = llhd.sig %init : !hw.struct<hi: i8, lo: i8>
  // KEEP: [[BITS:%.+]] = llhd.sig.extract %sig from {{%.+}} : <!hw.struct<hi: i8, lo: i8>> -> <i4>
  %bits = llhd.sig.extract %sig from %four : <!hw.struct<hi: i8, lo: i8>> -> <i4>
  %hi = llhd.sig.struct_extract %sig["hi"] : <!hw.struct<hi: i8, lo: i8>>
  // KEEP: llhd.drv [[BITS]], %value
  llhd.drv %bits, %value after %time : i4
  // KEEP: llhd.drv {{%.+}}, %high
  llhd.drv %hi, %high after %time : i8
  %result = llhd.prb %sig : !hw.struct<hi: i8, lo: i8>
  hw.output %result : !hw.struct<hi: i8, lo: i8>
}

// Sig2Reg already reasons in bit intervals. A constant aggregate projection
// must preserve its bit offset and return the original aggregate type.
// MEM-LABEL: @constant_slice
// KEEP-LABEL: @constant_slice
// REG-LABEL: @constant_slice
hw.module @constant_slice(in %init: !hw.struct<hi: i8, lo: i8>, in %value: i6, out result: !hw.struct<hi: i8, lo: i8>) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  %six = hw.constant 6 : i4
  // KEEP: %sig = llhd.sig
  %sig = llhd.sig %init : !hw.struct<hi: i8, lo: i8>
  // KEEP: [[BITS:%.+]] = llhd.sig.extract %sig from {{%.+}} : <!hw.struct<hi: i8, lo: i8>> -> <i6>
  %bits = llhd.sig.extract %sig from %six : <!hw.struct<hi: i8, lo: i8>> -> <i6>
  // KEEP: llhd.drv [[BITS]], %value
  // REG: hw.bitcast %init : (!hw.struct<hi: i8, lo: i8>) -> i16
  // REG: [[RESULT:%.+]] = hw.bitcast {{.*}} : (i16) -> !hw.struct<hi: i8, lo: i8>
  // REG-NOT: llhd.
  // REG: hw.output [[RESULT]]
  llhd.drv %bits, %value after %time : i6
  %result = llhd.prb %sig : !hw.struct<hi: i8, lo: i8>
  hw.output %result : !hw.struct<hi: i8, lo: i8>
}

// Bit zero is the low bit of the last field, not the entire first field. In
// particular, it must not be rewired to subslot zero during SROA.
// MEM-LABEL: @low_bit
// KEEP-LABEL: @low_bit
// REG-LABEL: @low_bit
hw.module @low_bit(in %init: !hw.struct<hi: i8, lo: i8>, in %value: i1, out result: !hw.struct<hi: i8, lo: i8>) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  %zero = hw.constant 0 : i4
  // KEEP: %sig = llhd.sig
  %sig = llhd.sig %init : !hw.struct<hi: i8, lo: i8>
  // KEEP: [[BIT:%.+]] = llhd.sig.extract %sig from {{%.+}} : <!hw.struct<hi: i8, lo: i8>> -> <i1>
  %bit = llhd.sig.extract %sig from %zero : <!hw.struct<hi: i8, lo: i8>> -> <i1>
  // KEEP: llhd.drv [[BIT]], %value
  llhd.drv %bit, %value after %time : i1
  %result = llhd.prb %sig : !hw.struct<hi: i8, lo: i8>
  hw.output %result : !hw.struct<hi: i8, lo: i8>
}
