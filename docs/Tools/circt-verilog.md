# circt-verilog

**circt-verilog** \[_options_] \[_filename_]

A (System) Verilog parser that generates CIRCT code using the core dialects. 
The `circ-verilog` tool depends on the [slang](https://github.com/MikePopoloski/slang) parser and as such requires a specific build configuration. The supported subset of System Verilog is actively growing and can be tracked via the [SV Tests dashboard](https://chipsalliance.github.io/sv-tests-results/).

### Build
Add to your circt cmake command:`-DCIRCT_SLANG_FRONTEND_ENABLED=ON`.

```
$ cd circt
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_LLD=ON  \
    -DCIRCT_SLANG_FRONTEND_ENABLED=ON
$ ninja circt-verilog
``` 

### Example
A simple counter module that uses a register with a synchronous reset is written
in System Verilog (`counter.sv`).

```sv
module counter (
    input  logic       clk,
    input  logic       reset,
    output logic [7:0] count
);
    always_ff @(posedge clk or posedge reset) begin
        if (reset)
            count <= 8'b0;
        else
            count <= count + 9'd1;
    end
endmodule
```

Writing `circt-verilog counter.sv` generates the following:
```mlir
module {
  hw.module @counter(in %clk : i1, in %reset : i1, out count : i8) {
    %c1_i8 = hw.constant 1 : i8
    %c0_i8 = hw.constant 0 : i8
    %0 = comb.add %count, %c1_i8 : i8
    %1 = seq.to_clock %clk
    %count = seq.firreg %0 clock %1 reset async %reset, %c0_i8 : i8
    hw.output %count : i8
  }
}
```
