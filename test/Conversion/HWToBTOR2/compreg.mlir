// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s  

module {
    //CHECK:    [[NID0:[0-9]+]] sort bitvec 1
    //CHECK:    [[NID1:[0-9]+]] input [[NID0]] reset
    //CHECK:    [[NID2:[0-9]+]] input [[NID0]] en
    hw.module @Counter(in %clock : !seq.clock, in %reset : i1, in %en : i1) {
    %0 = seq.from_clock %clock
    // Registers are all emitted before any other operation
    //CHECK:    [[NID6:[0-9]+]] sort bitvec 32
    //CHECK:    [[INITCONST:[0-9]+]] constd [[NID6]] 0
    //CHECK:    [[NID12:[0-9]+]] state [[NID6]] count
    //CHECK:    [[INIT:[0-9]+]] init [[NID6]] [[NID12]] [[INITCONST]]
    //CHECK:    [[REG2NID:[0-9]+]] state [[NID6]] count2
    //CHECK-NOT: [[INITCONST]] constd [[NID6]] 0
    //CHECK:    [[INIT:[0-9]+]] init [[NID6]] [[REG2NID]] [[INITCONST]]

    //CHECK:    [[NID3:[0-9]+]] sort bitvec 28
    //CHECK:    [[NID4:[0-9]+]] constd [[NID3]] 0 
    %c0_i28 = hw.constant 0 : i28

    //CHECK:    [[NID5:[0-9]+]] constd [[NID0]] 0
    %false = hw.constant false
    
    //CHECK:    [[NID7:[0-9]+]] constd [[NID6]] 22
    %c22_i32 = hw.constant 22 : i32

    //CHECK:    [[NID8:[0-9]+]] constd [[NID0]] -1
    %true = hw.constant true

    //CHECK:    [[NID9:[0-9]+]] sort bitvec 4
    //CHECK:    [[NID10:[0-9]+]] constd [[NID9]] -6
    %c-6_i4 = hw.constant -6 : i4

    //CHECK:    [[NID11:[0-9]+]] constd [[NID6]] 0
    %c0_i32 = hw.constant 0 : i32
    
    %init = seq.initial () {
        %c0_i8 = hw.constant 0 : i32
        seq.yield %c0_i8 : i32
    } : () -> !seq.immutable<i32>

    %count = seq.compreg %9, %clock reset %reset, %c0_i32 initial %init : i32
    %count2 = seq.compreg %9, %clock reset %reset, %c0_i32 initial %init : i32

    //CHECK:    [[NID13:[0-9]+]] eq [[NID0]] [[NID12]] [[NID7]]
    %1 = comb.icmp eq %count, %c22_i32 : i32

    //CHECK:    [[NID14:[0-9]+]] and [[NID0]] [[NID13]] [[NID2]]
    %2 = comb.and %1, %en : i1

    //CHECK:    [[NID15:[0-9]+]] ite [[NID6]] [[NID14]] [[NID11]] [[NID12]]
    %3 = comb.mux %2, %c0_i32, %count : i32

    //CHECK:    [[NID16:[0-9]+]] neq [[NID0]] [[NID12]] [[NID7]]
    %4 = comb.icmp ne %count, %c22_i32 : i32

    //CHECK:    [[NID17:[0-9]+]] and [[NID0]] [[NID16]] [[NID2]]
    %5 = comb.and %4, %en : i1

    //CHECK:    [[NID18:[0-9]+]] sort bitvec 33
    //CHECK:    [[NID19:[0-9]+]] concat [[NID18]] [[NID5]] [[NID12]]
    %6 = comb.concat %false, %count : i1, i32

    //CHECK:    [[NID20:[0-9]+]] constd [[NID18]] 1
    %c1_i33 = hw.constant 1 : i33

    //CHECK:    [[NID21:[0-9]+]] add [[NID18]] [[NID19]] [[NID20]]
    %7 = comb.add %6, %c1_i33 : i33

    //CHECK:    [[NID22:[0-9]+]] slice [[NID6]] [[NID21]] 31 0
    %8 = comb.extract %7 from 0 : (i33) -> i32

    //CHECK:    [[NID23:[0-9]+]] ite [[NID6]] [[NID17]] [[NID22]] [[NID15]]
    %9 = comb.mux %5, %8, %3 : i32

    //CHECK:    [[NID24:[0-9]+]] constd [[NID6]] 10
    %c10_i32 = hw.constant 10 : i32

    //CHECK:    [[NID25:[0-9]+]] neq [[NID0]] [[NID12]] [[NID24]]
    %10 = comb.icmp ne %count, %c10_i32 : i32
    sv.always posedge %0 {
      sv.if %en {

        //CHECK:    [[NID26:[0-9]+]] implies [[NID0]] [[NID2]] [[NID25]]
        //CHECK:    [[NID27:[0-9]+]] not [[NID0]] [[NID26]]
        //CHECK:    [[NID28:[0-9]+]] bad [[NID27]]
        sv.assert %10, immediate message "Counter reached 10!"
      }
    }
    hw.output

    //CHECK:    [[NID30:[0-9]+]] ite [[NID6]] [[NID1]] [[NID11]] [[NID23]]
    //CHECK:    [[NID31:[0-9]+]] next [[NID6]] [[NID12]] [[NID30]]
  }
}
