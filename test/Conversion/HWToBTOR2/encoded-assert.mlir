// RUN: circt-opt %s --convert-hw-to-btor2 -o tmp.mlir | FileCheck %s  

module {
  sv.macro.decl @SYNTHESIS
  sv.macro.decl @ASSERT_VERBOSE_COND
  sv.macro.decl @ASSERT_VERBOSE_COND_
  emit.fragment @ASSERT_VERBOSE_COND_FRAGMENT {
    sv.verbatim "\0A// Users can define 'ASSERT_VERBOSE_COND' to add an extra gate to assert error printing."
    sv.ifdef  @ASSERT_VERBOSE_COND_ {
    } else {
      sv.ifdef  @ASSERT_VERBOSE_COND {
        sv.macro.def @ASSERT_VERBOSE_COND_ "(`ASSERT_VERBOSE_COND)"
      } else {
        sv.macro.def @ASSERT_VERBOSE_COND_ "1"
      }
    }
  }
  sv.macro.decl @STOP_COND
  sv.macro.decl @STOP_COND_
  emit.fragment @STOP_COND_FRAGMENT {
    sv.verbatim "\0A// Users can define 'STOP_COND' to add an extra gate to stop conditions."
    sv.ifdef  @STOP_COND_ {
    } else {
      sv.ifdef  @STOP_COND {
        sv.macro.def @STOP_COND_ "(`STOP_COND)"
      } else {
        sv.macro.def @STOP_COND_ "1"
      }
    }
  }
  //CHECK:    [[NID0:[0-9]+]] sort bitvec 1
  //CHECK:    [[NID1:[0-9]+]] input [[NID0]] reset
  hw.module @Counter(in %clock : !seq.clock, in %reset : i1) {
    //CHECK:    [[NID2:[0-9]+]] sort bitvec 32
    //CHECK:    [[NID3:[0-9]+]] state [[NID2]] count
    //CHECK:    [[NID4:[0-9]+]] constd [[NID2]] 1
    %c1_i32 = hw.constant 1 : i32
    //CHECK:    [[NID5:[0-9]+]] constd [[NID2]] 42
    %c42_i32 = hw.constant 42 : i32
    //CHECK:    [[NID6:[0-9]+]] constd [[NID0]] -1
    %true = hw.constant true
    //CHECK:    [[NID7:[0-9]+]] constd [[NID2]] 0
    %c0_i32 = hw.constant 0 : i32
    %0 = seq.from_clock %clock
    %count = seq.firreg %3 clock %clock reset sync %reset, %c0_i32 {firrtl.random_init_start = 0 : ui64} : i32
    //CHECK:    [[NID8:[0-9]+]] eq [[NID0]] [[NID3]] [[NID5]]
    %1 = comb.icmp bin eq %count, %c42_i32 : i32
    //CHECK:    [[NID9:[0-9]+]] add [[NID2]] [[NID3]] [[NID4]]
    %2 = comb.add %count, %c1_i32 {sv.namehint = "_count_T"} : i32
    //CHECK:    [[NID10:[0-9]+]] ite [[NID2]] [[NID8]] [[NID7]] [[NID9]]
    %3 = comb.mux bin %1, %c0_i32, %2 : i32
    //CHECK:    [[NID11:[0-9]+]] xor [[NID0]] [[NID1]] [[NID6]]
    %4 = comb.xor bin %reset, %true : i1
    //CHECK:    [[NID12:[0-9]+]] ugt [[NID0]] [[NID3]] [[NID5]]
    %5 = comb.icmp bin ugt %count, %c42_i32 : i32
    //CHECK:    [[NID13:[0-9]+]] and [[NID0]] [[NID11]] [[NID12]]
    %6 = comb.and bin %4, %5 : i1

    //CHECK:    [[NID14:[0-9]+]] not [[NID0]] [[NID13]]
    //CHECK:    [[NID15:[0-9]+]] bad [[NID14]] 
    sv.ifdef  @SYNTHESIS {
    } else {
      sv.always posedge %0 {
        sv.if %6 {
          %ASSERT_VERBOSE_COND_ = sv.macro.ref @ASSERT_VERBOSE_COND_() : () -> i1
          sv.if %ASSERT_VERBOSE_COND_ {
            sv.error "Assertion failed: counter overflowed!\0A    at LTLSpec.scala:85 chisel3.assert(count <= 42.U, \22counter overflowed!\22)\0A"
          }
          %STOP_COND_ = sv.macro.ref @STOP_COND_() : () -> i1
          sv.if %STOP_COND_ {
            sv.fatal 1
          }
        }
      }
    }
    //CHECK:    [[NID16:[0-9]+]] ite [[NID2]] [[NID1]] [[NID7]] [[NID10]]
    //CHECK:    [[NID17:[0-9]+]] next [[NID2]] [[NID3]] [[NID16]]
    hw.output
  }
  om.class @Counter_Class(%basepath: !om.basepath) {
  }
}

