// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

// Lit testing random values is iffy, but the runtime environment should ensure
// reproducible results across runs and platforms.

// CHECK-LABEL: - addr = 21
// CHECK-NEXT: rndA = 707ca895977cf11
// CHECK-NEXT: rndB = 28e9cfdfcf6b898
// CHECK-NEXT: fill = cafe
// CHECK-NEXT: rept = 8000400020001
// CHECK-NEXT: - addr = 0
// CHECK-NEXT: rndA = 5160879eac03cbb
// CHECK-NEXT: rndB = d78aeb0b84b4823
// CHECK-NEXT: fill = cafe
// CHECK-NEXT: rept = 8000400020001
// CHECK-NEXT: - addr = 1ff
// CHECK-NEXT: rndA = 198ecb046b4841d
// CHECK-NEXT: rndB = 357020a9a09635b
// CHECK-NEXT: fill = cafe
// CHECK-NEXT: rept = 8000400020001
// CHECK-NEXT: - addr = aa
// CHECK-NEXT: rndA = 16b4a44c8c8ce64
// CHECK-NEXT: rndB = 476fc6a9fd6fb83
// CHECK-NEXT: fill = cafe
// CHECK-NEXT: rept = 8000400020001

module {
  arc.define @mem_write(%arg0: i9, %arg1: i60, %arg2: i1) -> (i9, i60, i1) {
    arc.output %arg0, %arg1, %arg2 : i9, i60, i1
  }
  hw.module @SyncRAM(
     in %clk : i1, in %reset : i1, in %en : i1, in %addr : i9, in %din : i60, in %wen : i1,
     out dout0 : i60, out dout1 : i60, out dout2 : i60, out dout3 : i60, out addrOut : i9) {
    %clock = seq.to_clock %clk

    %cst33_i9 = hw.constant 33 : i9

    %randInit = arc.initmem.randomized
    // Check that identical memories get different initial values
    %mem0 = arc.memory <512 x i60, i9> initial %randInit : !arc.memory_initializer<* x *>
    %mem3 = arc.memory <512 x i60, i9> initial %randInit : !arc.memory_initializer<* x *>

    %fillInit = arc.initmem.filled 0xcafe : i16
    %mem1 = arc.memory <512 x i60, i9> initial %fillInit : !arc.memory_initializer<* x *>

    %repeatInit = arc.initmem.filled repeat 1 : i17
    %mem2 = arc.memory <512 x i60, i9> initial %repeatInit : !arc.memory_initializer<* x *>



    %addrReg = seq.compreg %0, %clock powerOn %cst33_i9 : i9
    %0 = comb.mux bin %en, %addr, %addrReg : i9

    %3 = seq.compreg %en, %clock : i1

    %rd0 = arc.memory_read_port %mem0[%addrReg] : <512 x i60, i9>
    %rd1 = arc.memory_read_port %mem1[%addrReg] : <512 x i60, i9>
    %rd2 = arc.memory_read_port %mem2[%addrReg] : <512 x i60, i9>
    %rd3 = arc.memory_read_port %mem3[%addrReg] : <512 x i60, i9>

    %c0_i60 = hw.constant 0 : i60

    arc.memory_write_port %mem0, @mem_write(%addrReg, %din, %wen) clock %clock enable latency 1 : <512 x i60, i9>, i9, i60, i1
    arc.memory_write_port %mem1, @mem_write(%addrReg, %din, %wen) clock %clock enable latency 1 : <512 x i60, i9>, i9, i60, i1
    arc.memory_write_port %mem2, @mem_write(%addrReg, %din, %wen) clock %clock enable latency 1 : <512 x i60, i9>, i9, i60, i1
    arc.memory_write_port %mem3, @mem_write(%addrReg, %din, %wen) clock %clock enable latency 1 : <512 x i60, i9>, i9, i60, i1

    hw.output %rd0, %rd1, %rd2, %rd3, %addrReg : i60, i60, i60, i60 , i9
  }

  func.func @main() {
    %cst0 = arith.constant 0 : i9
    %cst1ff = arith.constant 0x1FF : i9
    %cstaa = arith.constant 0xAA : i9

    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1

    arc.sim.instantiate @SyncRAM as %model {
      %addr0  = arc.sim.get_port %model, "addrOut" : i9, !arc.sim.instance<@SyncRAM>
      %res0_0 = arc.sim.get_port %model, "dout0" : i60, !arc.sim.instance<@SyncRAM>
      %res1_0 = arc.sim.get_port %model, "dout1" : i60, !arc.sim.instance<@SyncRAM>
      %res2_0 = arc.sim.get_port %model, "dout2" : i60, !arc.sim.instance<@SyncRAM>
      %res3_0 = arc.sim.get_port %model, "dout3" : i60, !arc.sim.instance<@SyncRAM>
      arc.sim.emit " - addr", %addr0 : i9
      arc.sim.emit "rndA", %res0_0 : i60
      arc.sim.emit "rndB", %res3_0 : i60
      arc.sim.emit "fill", %res1_0 : i60
      arc.sim.emit "rept", %res2_0 : i60

      arc.sim.set_input %model, "en" = %true : i1, !arc.sim.instance<@SyncRAM>
      arc.sim.set_input %model, "wen" = %false : i1, !arc.sim.instance<@SyncRAM>
      arc.sim.set_input %model, "reset" = %false : i1, !arc.sim.instance<@SyncRAM>

      arc.sim.set_input %model, "addr" = %cst0 : i9, !arc.sim.instance<@SyncRAM>

      arc.sim.set_input %model, "clk" = %false : i1, !arc.sim.instance<@SyncRAM>
      arc.sim.step %model : !arc.sim.instance<@SyncRAM>

      arc.sim.set_input %model, "clk" = %true : i1, !arc.sim.instance<@SyncRAM>
      arc.sim.step %model : !arc.sim.instance<@SyncRAM>
      arc.sim.set_input %model, "clk" = %false : i1, !arc.sim.instance<@SyncRAM>
      arc.sim.step %model : !arc.sim.instance<@SyncRAM>

      %addr1  = arc.sim.get_port %model, "addrOut" : i9, !arc.sim.instance<@SyncRAM>
      %res0_1 = arc.sim.get_port %model, "dout0" : i60, !arc.sim.instance<@SyncRAM>
      %res1_1 = arc.sim.get_port %model, "dout1" : i60, !arc.sim.instance<@SyncRAM>
      %res2_1 = arc.sim.get_port %model, "dout2" : i60, !arc.sim.instance<@SyncRAM>
      %res3_1 = arc.sim.get_port %model, "dout3" : i60, !arc.sim.instance<@SyncRAM>
      arc.sim.emit " - addr", %addr1 : i9
      arc.sim.emit "rndA", %res0_1 : i60
      arc.sim.emit "rndB", %res3_1 : i60
      arc.sim.emit "fill", %res1_1 : i60
      arc.sim.emit "rept", %res2_1 : i60

      arc.sim.set_input %model, "addr" = %cst1ff : i9, !arc.sim.instance<@SyncRAM>

      arc.sim.set_input %model, "clk" = %true : i1, !arc.sim.instance<@SyncRAM>
      arc.sim.step %model : !arc.sim.instance<@SyncRAM>
      arc.sim.set_input %model, "clk" = %false : i1, !arc.sim.instance<@SyncRAM>
      arc.sim.step %model : !arc.sim.instance<@SyncRAM>

      %addr2  = arc.sim.get_port %model, "addrOut" : i9, !arc.sim.instance<@SyncRAM>
      %res0_2 = arc.sim.get_port %model, "dout0" : i60, !arc.sim.instance<@SyncRAM>
      %res1_2 = arc.sim.get_port %model, "dout1" : i60, !arc.sim.instance<@SyncRAM>
      %res2_2 = arc.sim.get_port %model, "dout2" : i60, !arc.sim.instance<@SyncRAM>
      %res3_2 = arc.sim.get_port %model, "dout3" : i60, !arc.sim.instance<@SyncRAM>
      arc.sim.emit " - addr", %addr2 : i9
      arc.sim.emit "rndA", %res0_2 : i60
      arc.sim.emit "rndB", %res3_2 : i60
      arc.sim.emit "fill", %res1_2 : i60
      arc.sim.emit "rept", %res2_2 : i60

      arc.sim.set_input %model, "addr" = %cstaa : i9, !arc.sim.instance<@SyncRAM>

      arc.sim.set_input %model, "clk" = %true : i1, !arc.sim.instance<@SyncRAM>
      arc.sim.step %model : !arc.sim.instance<@SyncRAM>
      arc.sim.set_input %model, "clk" = %false : i1, !arc.sim.instance<@SyncRAM>
      arc.sim.step %model : !arc.sim.instance<@SyncRAM>

      %addr3  = arc.sim.get_port %model, "addrOut" : i9, !arc.sim.instance<@SyncRAM>
      %res0_3 = arc.sim.get_port %model, "dout0" : i60, !arc.sim.instance<@SyncRAM>
      %res1_3 = arc.sim.get_port %model, "dout1" : i60, !arc.sim.instance<@SyncRAM>
      %res2_3 = arc.sim.get_port %model, "dout2" : i60, !arc.sim.instance<@SyncRAM>
      %res3_3 = arc.sim.get_port %model, "dout3" : i60, !arc.sim.instance<@SyncRAM>
      arc.sim.emit " - addr", %addr3 : i9
      arc.sim.emit "rndA", %res0_3 : i60
      arc.sim.emit "rndB", %res3_3 : i60
      arc.sim.emit "fill", %res1_3 : i60
      arc.sim.emit "rept", %res2_3 : i60
    }
    return
  }
}
