// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | FileCheck %s

// module {
//   hw.module.extern @param_in_param<VLD_RST_VALUE: i1 = false, DATA_RST_VALUE: !hw.int<#hw.param.decl.ref<"DATA_WIDTH">>, RST_EN: i1 = true, DATA_WIDTH: i32 = 32>(in %input : !hw.int<#hw.param.decl.ref<"DATA_WIDTH">>, in %clk : !seq.clock, in %xStall : i1, in %xFlush : i1, in %xValid : i1, in %yStall : i1, in %yFlush : i1, in %rstVal : !hw.int<#hw.param.decl.ref<"DATA_WIDTH">>, in %rst : i1, in %vldRstVal : i1, out output : !hw.int<#hw.param.decl.ref<"DATA_WIDTH">>, out yValid : i1) attributes {verilogName = "param_in_param"}
//   hw.module @xx(in %input : i96, in %clk : !seq.clock, in %xStall : i1, in %xFlush : i1, in %xValid : i1, in %yStall : i1, in %yFlush : i1, in %rstVal : i96, in %rst : i1, in %vldRstVal : i1, out output : i96, out yValid : i1) {
//     %x:2 = seq.sf_dff sym @x name "x" input %input, clock %clk, xStall %xStall, xFlush %xFlush, xValid %xValid, yStall %yStall, yFlush %yFlush, reset %rst, dataResetValue %rstVal, validResetValue %vldRstVal : i96 
//     hw.output %x#0, %x#1 : i96, i1
//   }
// }

module {
  hw.module.extern @param_in_param < DATA_RST_VALUE: !hw.int<#hw.param.decl.ref<"DATA_WIDTH">>
                                   , DATA_WIDTH: i32 = 32>
    ( in %input : !hw.int<#hw.param.decl.ref<"DATA_WIDTH">>
    , in %clk : i1
    , in %rst : i1
    , out output : !hw.int<#hw.param.decl.ref<"DATA_WIDTH">>
    ) attributes {verilogName = "param_in_param"}
  
  hw.module @xx
    ( in %input : i96
    , in %clk : i1
    , in %rst : i1
    , out output : i96
    ) {
    
    %x.output = hw.instance "pipe_reg111" @param_in_param < DATA_RST_VALUE: i96 = 0
                                                          , DATA_WIDTH: i32 = 96
                                                          > 
        ( input: %input: i96
        , clk: %clk: i1
        , rst: %rst: i1
        ) -> 
        ( output: i96
        )
     hw.output %x.output  : i96
  }
}