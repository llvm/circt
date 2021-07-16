// RUN: circt-opt -convert-fsm-to-hw -split-input-file %s | FileCheck %s

// CHECK: module
fsm.machine @axi_read_target(%arvalid: i1, %arlen: i8, %rready: i1) -> (i1, i1, i1) attributes {stateType = i2} {
  %arready = fsm.variable "arready" : i1
  %rvalid = fsm.variable "rvalid" : i1
  %rlast = fsm.variable "rlast" : i1

  %cnt = fsm.variable "cnt" : i8

  %false = constant false
  %true = constant true

  %arvalid_n = xor %arvalid, %true : i1
  %rready_n = xor %rready, %true : i1

  %c0_i8 = constant 0 : i8
  %c1_i8 = constant 1 : i8

  %arlen_eq0 = cmpi eq, %arlen, %c0_i8 : i8
  %cnt_eq0 = cmpi eq, %cnt, %c0_i8 : i8

  %arlen_eq0_n = xor %arlen_eq0, %true : i1
  %cnt_eq0_n = xor %cnt_eq0, %true : i1

  fsm.state "IDLE" entry  {
    fsm.update %arready, %true : i1, i1
  } exit  {
    fsm.update %arready, %false : i1, i1
  } transitions  {
    fsm.transition @MID guard  {
      %cond = and %arvalid, %arlen_eq0_n : i1
      fsm.return %cond : i1
    } action  {
      fsm.update %cnt, %arlen : i8, i8
    }
    fsm.transition @END guard  {
      %cond = and %arvalid, %arlen_eq0 : i1
      fsm.return %cond : i1
    } action  {
    }
  }

  fsm.state "MID" entry  {
    fsm.update %rvalid, %true : i1, i1
  } exit  {
    fsm.update %rvalid, %false : i1, i1
  } transitions  {
    fsm.transition @MID guard  {
      %cond = and %rready, %cnt_eq0_n : i1
      fsm.return %cond : i1
    } action  {
      %next_cnt = subi %cnt, %c1_i8 : i8
      fsm.update %cnt, %next_cnt : i8, i8
    }
    fsm.transition @END guard  {
      %cond = and %rready, %cnt_eq0 : i1
      fsm.return %cond : i1
    } action  {
    }
  }

  fsm.state "HOLD" entry  {
    fsm.update %rvalid, %true : i1, i1
    fsm.update %rlast, %true : i1 , i1
  } exit  {
    fsm.update %rvalid, %false : i1, i1
    fsm.update %rlast, %false : i1 , i1
  } transitions  {
    fsm.transition @MID guard  {
      %cond = and %rready, %arlen_eq0_n : i1
      fsm.return %cond : i1
    } action  {
    }
    fsm.transition @END guard  {
      %cond = and %rready, %arlen_eq0 : i1
      fsm.return %cond : i1
    } action  {
    }
  }

  fsm.state "END" entry  {
    fsm.update %arready, %true : i1, i1
    fsm.update %rvalid, %true : i1, i1
    fsm.update %rlast, %true : i1 , i1
  } exit  {
    fsm.update %arready, %false : i1, i1
    fsm.update %rvalid, %false : i1, i1
    fsm.update %rlast, %false : i1 , i1
  } transitions  {
    fsm.transition @IDLE guard  {
      %cond = and %arvalid_n, %rready : i1
      fsm.return %cond : i1
    } action  {
    }
    fsm.transition @MID guard  {
      %cond_tmp = and %arvalid, %rready : i1
      %cond = and %cond_tmp, %arlen_eq0_n : i1
      fsm.return %cond : i1
    } action  {
      fsm.update %cnt, %arlen : i8, i8
    }
    fsm.transition @HOLD guard  {
      %cond = and %arvalid, %rready_n : i1
      fsm.return %cond : i1
    } action  {
    }
    fsm.transition @END guard  {
      %cond_tmp = and %arvalid, %rready : i1
      %cond = and %cond_tmp, %arlen_eq0 : i1
      fsm.return %cond : i1
    } action  {
    }
  }

  fsm.output %arready, %rvalid, %rlast : i1, i1, i1
}
