// RUN: circt-opt -convert-fsm-to-standard %s | \
// RUN: mlir-opt -convert-scf-to-std -convert-memref-to-llvm -convert-std-to-llvm -cse -canonicalize | \
// RUN: mlir-cpu-runner -e driver -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

func @driver() {
  %dut = fsm.instance "dut" @axi_read_target
  %rready = constant true

  %arvalid_0 = constant true
  %arlen_0 = constant 3 : i8
  %arready_0, %rvalid_0, %rlast_0 = fsm.trigger %dut(%arvalid_0, %arlen_0, %rready) : (i1, i8, i1) -> (i1, i1, i1)
  // CHECK: 010
  call @printU64(%arready_0): (i1) -> ()
  call @printU64(%rvalid_0): (i1) -> ()
  call @printU64(%rlast_0): (i1) -> ()
  call @printNewline() : () -> ()

  %arvalid_1 = constant false
  %arlen_1 = constant 0 : i8
  %arready_1, %rvalid_1, %rlast_1 = fsm.trigger %dut(%arvalid_1, %arlen_1, %rready) : (i1, i8, i1) -> (i1, i1, i1)
  // CHECK: 010
  call @printU64(%arready_1): (i1) -> ()
  call @printU64(%rvalid_1): (i1) -> ()
  call @printU64(%rlast_1): (i1) -> ()
  call @printNewline() : () -> ()
  return
}

func private @printU64(i1) -> ()
func private @printComma() -> ()
func private @printNewline() -> ()

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
