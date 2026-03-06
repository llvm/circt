# Support Basic LLHD Processes in Arcilator

Consider this simple SystemVerilog input:

```systemverilog
module Foo(output bit [41:0] x);
  initial begin
    x = 0;
    #1ns;
    x = 42;
    #1ns;
    x = 1337;
    #1ns;
    x = 9001;
    #1ns;
    x = 0;
  end
endmodule
```

The expected behavior of this is to see the output x have the following values at given points in time:

```
0ns  x = 0
1ns  x = 42
2ns  x = 1337
3ns  x = 9001
4ns  x = 0
```

Running this through `circt-verilog` yields roughly the following MLIR:

```mlir
hw.module @Foo(out x : i42) {
  // ...
  %1 = llhd.process -> i42 {
    llhd.wait yield (%c0_i42 : i42), delay %0, ^bb1
  ^bb1:
    llhd.wait yield (%c42_i42 : i42), delay %0, ^bb2
  ^bb2:
    llhd.wait yield (%c1337_i42 : i42), delay %0, ^bb3
  ^bb3:
    llhd.wait yield (%c9001_i42 : i42), delay %0, ^bb4
  ^bb4:
    llhd.halt %c0_i42 : i42
  }
  hw.output %1 : i42
}

```

We want to add support for executing this kind of process to Arcilator.
Note that there may be a lot more operations in these basic blocks to compute actual values to yield from the process.

## Process Lowering

LLHD processes essentially describe a coroutine that can yield and suspend for a certain period of time, or until a certain input signal changes.
To implement this in a simulator, we need to find the values live across waits and create and allocate state in arc.storage to store them alongside an integer index indicating at which point to resume the process.
The process then lowers into a bit of code that checks if the process' wait condition has been met or if it hasn't executed yet.
If that is the case, we need a cf.switch op to jump to the resumption point with the block operands loaded from storage.
When the code hits a wait, we simply store the integer index corresponding to the target block, alongside the values to be passed to that block on resume, and jump to after the process.
We'll want to skip processes with values live across waits -- they should all have been promoted to branch operands earlier.
A first implementation should just whole-sale reject any block operands on the llhd.wait.
We should also reject any LLHD ops besides llhd.process and the llhd.wait/llhd.halt terminators.
Broader LLHD support we can add later.
I expect the process to be lowered to something like the following in the Arcilator pipeline:

```mlir
arc.model @Foo {
^bb0(%arg0: !arc.storage):
  // ...
  %earliest_resume_time_across_all_procs = arc.state %arg0 : i64
  arc.state_write %earliest_resume_time_across_all_procs = %c-1_i64  // maximum time
  %current_time = arc.current_time
  %resume_time = arc.state %arg0 : i64
  %resume_block = arc.state %arg0 : i16
  %proc_result = arc.state %arg0 : i42
  %resume_time_read = arc.state_read %resume_time
  %resume = comb.icmp uge %current_time, %resume_time_read  // check if resume time reached
  scf.if %resume {
    scf.execute_region {
      %resume_block_read = arc.state_read %resume_block
      cf.switch %resume_block_read : i16, [
        0: ^origbb0,
        1: ^origbb1,
        2: ^origbb2,
        3: ^origbb3,
        4: ^origbb4,
        5: ^halted
      ]
      ^origbb0:
        // llhd.wait yield (%c0_i42 : i42), delay %0, ^bb1
        %delay = llhd.time_to_int %0
        %next_resume = comb.add %current_time, %delay
        arc.state_write %resume_time = %next_resume
        arc.state_write %resume_block = %c1_i16  // resume at orgibb1
        arc.state_write %proc_result = %c0_i42
        scf.yield
      ^origbb1:
        // llhd.wait yield (%c42_i42 : i42), delay %0, ^bb2
        %delay = llhd.time_to_int %0
        %next_resume = comb.add %current_time, %delay
        arc.state_write %resume_time = %next_resume
        arc.state_write %resume_block = %c2_i16  // resume at orgibb2
        arc.state_write %proc_result = %c42_i42
        scf.yield
      ^origbb2:
        // llhd.wait yield (%c1337_i42 : i42), delay %0, ^bb3
        %delay = llhd.time_to_int %0
        %next_resume = comb.add %current_time, %delay
        arc.state_write %resume_time = %next_resume
        arc.state_write %resume_block = %c3_i16  // resume at orgibb3
        arc.state_write %proc_result = %c1337_i42
        scf.yield
      ^origbb3:
        // llhd.wait yield (%c9001_i42 : i42), delay %0, ^bb4
        %delay = llhd.time_to_int %0
        %next_resume = comb.add %current_time, %delay
        arc.state_write %resume_time = %next_resume
        arc.state_write %resume_block = %c4_i16  // resume at orgibb4
        arc.state_write %proc_result = %c9001_i42
        scf.yield
      ^origbb4:
        // llhd.halt %c0_i42 : i42
        arc.state_write %resume_time = %c-1_i64  // resume time set to max
        arc.state_write %resume_block = %c5_i16  // immediately resume at ^halted
        arc.state_write %proc_result = %c0_i42
        scf.yield
      ^halted:
        scf.yield
    }
    %next_resume = arc.state_read %resume_time
    %current_earliest_resume = arc.state_read %earliest_resume_time_across_all_procs
    %updated_earliest_resume = arith.minui %current_earliest_resume, %next_resume
    arc.state_write %earliest_resume_time_across_all_procs = %updated_earliest_resume
  }
  // After evaluation of @Foo, the caller can consult the
  // `earliest_resume_time_across_all_procs` state to know when the contained
  // processes will want to be woken up the next time.
}
```

Based on this, we can generate a standalone main function for modules which does roughly the following:
```mlir
func Foo_standalone() {
  %foo = alloc_model(Foo);
  call Foo_initial(%foo);
  br ^check
^check:
  %next = read_state(%foo, "earliest_resume_time_across_all_procs")
  cf.cond_br (%next < max_i64), ^next, ^exit
^next:
  write_state(%foo, "current_time", %next)
  call Foo_eval(%foo);
  cf.br ^check
^exit:
  call Foo_final(%foo);
  return
}
```

When we generate such a function, Arcilator can automatically run simulations for top-level modules.
In the long run this should be the default behavior.
Until everything is fleshed out, we should add a `--run-module` option which accepts a module name and runs the corresponding `*_standalone` function in JIT.

## Implementation Strategy

Let's implement this in the following distinct steps, each of which should be its own standalone PR:

1. Process lowering in Arcilator, with a manual test using `arc.sim.*` ops to see that the process changes output at the right points in time
2. Tracking of earliest scheduled event time, with new `arc.sim.*` ops to read that information; this should be another i64 added after the current time at the beginning of the simulation state
3. Generate standalone runner function for a module
