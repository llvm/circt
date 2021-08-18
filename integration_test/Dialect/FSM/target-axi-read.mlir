// REQUIRES: ieee-sim
// RUN: circt-opt %s -convert-fsm-to-hw -cse -canonicalize -prettify-verilog | \
// RUN: circt-translate -export-verilog > %target-axi-read.sv
// RUN: circt-rtl-sim.py %target-axi-read.sv %S/driver.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK-NOT: Error: Assertion violation
// CHECK: Success

fsm.machine @axi_read_target(%arvalid: i1, %arlen: i8, %rready: i1) -> (i1, i1, i1) attributes {stateType = i2} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i8} : i8

  %true = constant true
  %false = constant false

  %c0_i8 = constant 0 : i8
  %c1_i8 = constant 1 : i8

  %arlen_eq0 = cmpi eq, %arlen, %c0_i8 : i8
  %arlen_ne0 = cmpi ne, %arlen, %c0_i8 : i8

  fsm.state "IDLE" output  {
    fsm.output %true, %false, %false : i1, i1, i1
  } transitions  {
    fsm.transition @MID guard  {
      %cond = and %arvalid, %arlen_ne0 : i1
      fsm.return %cond : i1
    } action  {
      %init_cnt = subi %arlen, %c1_i8 : i8
      fsm.update %cnt, %init_cnt : i8
    }
    fsm.transition @END guard  {
      %cond = and %arvalid, %arlen_eq0 : i1
      fsm.return %cond : i1
    } action  {
    }
  }

  fsm.state "MID" output  {
    fsm.output %false, %true, %false : i1, i1, i1
  } transitions  {
    fsm.transition @MID guard  {
      %cnt_ne0 = cmpi ne, %cnt, %c0_i8 : i8
      %cond = and %rready, %cnt_ne0 : i1
      fsm.return %cond : i1
    } action  {
      %next_cnt = subi %cnt, %c1_i8 : i8
      fsm.update %cnt, %next_cnt : i8
    }
    fsm.transition @END guard  {
      %cnt_eq0 = cmpi eq, %cnt, %c0_i8 : i8
      %cond = and %rready, %cnt_eq0 : i1
      fsm.return %cond : i1
    } action  {
    }
  }

  fsm.state "END" output  {
    fsm.output %true, %true, %true : i1, i1, i1
  } transitions  {
    fsm.transition @IDLE guard  {
      %arvalid_n = xor %arvalid, %true : i1
      %cond = and %arvalid_n, %rready : i1
      fsm.return %cond : i1
    } action  {
    }
    fsm.transition @MID guard  {
      %cond_tmp = and %arvalid, %rready : i1
      %cond = and %cond_tmp, %arlen_ne0 : i1
      fsm.return %cond : i1
    } action  {
      %init_cnt = subi %arlen, %c1_i8 : i8
      fsm.update %cnt, %init_cnt : i8
    }
    fsm.transition @END guard  {
      %cond_tmp = and %arvalid, %rready : i1
      %cond = and %cond_tmp, %arlen_eq0 : i1
      fsm.return %cond : i1
    } action  {
    }
    fsm.transition @HOLD guard  {
      %rready_n = xor %rready, %true : i1
      %cond = and %arvalid, %rready_n : i1
      fsm.return %cond : i1
    } action  {
    }
  }

  fsm.state "HOLD" output  {
    fsm.output %false, %true, %true : i1, i1, i1
  } transitions  {
    fsm.transition @MID guard  {
      %cond = and %rready, %arlen_ne0 : i1
      fsm.return %cond : i1
    } action  {
      %init_cnt = subi %arlen, %c1_i8 : i8
      fsm.update %cnt, %init_cnt : i8
    }
    fsm.transition @END guard  {
      %cond = and %rready, %arlen_eq0 : i1
      fsm.return %cond : i1
    } action  {
    }
  }
}
