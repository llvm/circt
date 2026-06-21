// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog %s | FileCheck %s --check-prefix=CORE
// REQUIRES: slang
// UNSUPPORTED: valgrind

module ForLoopNoCondition(output int out);
  initial begin
    int i;
    for (i = 0; ; i++) begin
      if (i == 4)
        break;
    end
    out = i;
  end
endmodule

// MOORE-LABEL: moore.module @ForLoopNoCondition
// MOORE:       [[C1:%.+]] = moore.constant 1 : i32
// MOORE:       [[C4:%.+]] = moore.constant 4 : i32
// MOORE:       [[C0:%.+]] = moore.constant 0 : i32
// MOORE:       cf.br ^[[LOOP:.+]]([[C0]] : !moore.i32)
// MOORE:     ^[[LOOP]]([[I:%.+]]: !moore.i32):
// MOORE:       [[EQ:%.+]] = moore.eq [[I]], [[C4]] : i32 -> i1
// MOORE:       cf.cond_br {{%.+}}, ^[[EXIT:.+]], ^[[STEP:.+]]
// MOORE:     ^[[STEP]]:
// MOORE:       [[NEXT:%.+]] = moore.add [[I]], [[C1]] : i32
// MOORE:       cf.br ^[[LOOP]]([[NEXT]] : !moore.i32)
// MOORE:     ^[[EXIT]]:

// CORE-LABEL: hw.module @ForLoopNoCondition
// CORE:       [[C1:%.+]] = hw.constant 1 : i32
// CORE:       [[C4:%.+]] = hw.constant 4 : i32
// CORE:       [[C0:%.+]] = hw.constant 0 : i32
// CORE:       cf.br ^[[LOOP:.+]]([[C0]] : i32)
// CORE:     ^[[LOOP]]([[I:%.+]]: i32):
// CORE:       [[EQ:%.+]] = comb.icmp eq [[I]], [[C4]] : i32
// CORE:       cf.cond_br [[EQ]], ^[[EXIT:.+]], ^[[STEP:.+]]
// CORE:     ^[[STEP]]:
// CORE:       [[NEXT:%.+]] = comb.add [[I]], [[C1]] : i32
// CORE:       cf.br ^[[LOOP]]([[NEXT]] : i32)
// CORE:     ^[[EXIT]]:

module WaitLevelToCore(input bit enable, output bit y);
  initial begin
    wait (enable)
      y = 1'b1;
  end
endmodule

// CORE-LABEL: hw.module @WaitLevelToCore
// CORE:       llhd.process
// CORE:       cf.br ^[[CHECK:.+]]
// CORE:     ^[[CHECK]]:
// CORE:       cf.cond_br %enable, ^[[RESUME:.+]], ^[[WAIT:.+]]
// CORE:     ^[[WAIT]]:
// CORE:       llhd.wait{{.*}}(%enable : i1), ^[[AFTER:.+]](%enable : i1)
// CORE:     ^[[AFTER]]([[BEFORE:%.+]]: i1):
// CORE:       [[CHANGED:%.+]] = comb.icmp bin ne [[BEFORE]], %enable : i1
// CORE:       cf.cond_br [[CHANGED]], ^[[CHECK]], ^[[WAIT]]
// CORE:     ^[[RESUME]]:

function automatic int ImmediateAssertFunction(input bit a);
  assert (a);
  assume (a);
  cover (a);
  return 1;
endfunction

task automatic ImmediateAssertTask(input bit a);
  assert (a);
  assume (a);
  cover (a);
endtask

module ImmediateAssertInSubroutine(input bit a, output int y);
  initial begin
    y = ImmediateAssertFunction(a);
    ImmediateAssertTask(a);
  end
endmodule

// CORE-LABEL: llhd.coroutine private @ImmediateAssertTask
// CORE-SAME:  (%arg0: i1)
// CORE:       verif.assert %arg0 : i1
// CORE:       verif.assume %arg0 : i1
// CORE:       verif.cover %arg0 : i1
// CORE:       llhd.return

// CORE-LABEL: hw.module @ImmediateAssertInSubroutine
// CORE:       llhd.process
// CORE:       verif.assert %a : i1
// CORE:       verif.assume %a : i1
// CORE:       verif.cover %a : i1
// CORE:       llhd.call_coroutine @ImmediateAssertTask(%a) : (i1) -> ()

module EmptyForkToCore(output int y);
  initial begin
    fork
    join
    y = 1;
  end
endmodule

// MOORE-LABEL: moore.module @EmptyForkToCore
// MOORE-NOT:   moore.fork
// MOORE:       moore.blocking_assign
// MOORE:       moore.return

// CORE-LABEL: hw.module @EmptyForkToCore
// CORE:       [[C1:%.+]] = hw.constant 1 : i32
// CORE:       hw.output [[C1]] : i32
