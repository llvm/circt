# EDA Tool Workarounds

This documents various bugs found in EDA tools and their workarounds in circt.
Each but will have a brief description, example code, and the mitigation added
(with links to the commit when possible).  

# Automatic Variables Cause Latch Warnings

Verilator issues a latch warning for fully-initialized, automatic variables.  This precludes using locally scoped variables.
https://github.com/verilator/verilator/issues/4022

## Example
```
module ALU(
  input         clock,
  input  [4:0]  operation,
  input  [63:0] inputs_1,
                inputs_0,
                inputs_2,
  input  [16:0] immediate,
  output [63:0] output_0
);
  reg  [63:0]  casez_tmp_1;
  always_comb begin
    automatic logic [63:0] lowHigh;
    casez (operation)
      5'b00011:
        casez_tmp_1 = inputs_0 & inputs_1;
      5'b00100:
        casez_tmp_1 = inputs_0 | inputs_1;
      5'b00101:
        casez_tmp_1 = inputs_0 ^ inputs_1;
      5'b01001: begin
        automatic logic [16:0] _aluOutput_T_22 =
          immediate >> {14'h0, inputs_2, inputs_1[0], inputs_0[0]};
        casez_tmp_1 = {63'h0, _aluOutput_T_22[0]};
      end
      default:
        casez_tmp_1 = inputs_0;
    endcase
  end
endmodule
```
Gives:
```
$ verilator --version
Verilator 5.008 2023-03-04 rev v5.008
$ verilator --lint-only ALU.sv
%Warning-LATCH: ALU.sv:11:3: Latch inferred for signal 'ALU.unnamedblk1.unnamedblk2._aluOutput_T_22' (not all control paths of combinational always assign a value)
                           : ... Suggest use of always_latch for intentional latches
   11 |   always_comb begin
      |   ^~~~~~~~~~~
                ... For warning description see https://verilator.org/warn/LATCH?v=4.218
                ... Use "/* verilator lint_off LATCH */" and lint_on around source to disable this message.
%Error: Exiting due to 1 warning(s)
```

## Workaround

Flag added to promote all storage to the top level of a module.
https://github.com/llvm/circt/commit/3c8b4b47b600ea6bcc6da56fe9b81d6fe4022e4c

# Inline Array calculations can cause synthesis failures

Some tools have bugs (version dependent) in const prop in this case.

## Example
```
module Foo (input clock, input in, output [2:0] out);
  reg [2:0] state;
  wire [7:0][2:0] array = 24'h4 << 6;
  wire [2:0] a = array[state];
  wire [2:0] b = array[state + 3'h1 + 3'h1];
  // works:      array[state + (3'h1 + 3'h1)]
  // works:      array[state + 3'h2]
  always @(posedge clock) state <= in ? a : b;
  assign out = b;
endmodule
```

## Workaround

Flag added to export verilog to force array index calculations to not be inline.

https://github.com/llvm/circt/commit/15a1f95f2d59767f20b459a12ac42338de22bc97

# Memory semantics changed by synthesis

Read/Write forwarding behavior is dependent on memory size, since the synthesis 
tool changes it's mapping based on that.  The "optimized" mapping does not 
preserve the behavior of the verilog.  This is a known issue reported on various
forums by multiple people.  There are some version dependencies on when this
manifests.

## Example
```
Qux:
  module Qux:
    input clock: Clock
    input addr: UInt<1>
    input r: {en: UInt<1>, flip data: {a: UInt<32>, b: UInt<32>}, addr: UInt<1>}
    input w: {en: UInt<1>, data: {a: UInt<32>, b: UInt<32>}, addr: UInt<1>, mask: {a: UInt<1>, b: UInt<1>}}

    mem m :
      data-type => {a: UInt<32>, b: UInt<32>}
      depth => 1
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    m.r.clk <= clock
    m.r.en <= r.en
    m.r.addr <= r.addr
    r.data <= m.r.data

    m.w.clk <= clock
    m.w.en <= w.en
    m.w.addr <= w.addr
    m.w.data <= w.data
    m.w.mask <= w.mask
```
Compile with either firtool -repl-seq-mem -repl-seq-mem-file=mem.conf Foo.fir and firrtl -i Foo.fir.

## Workaround

FIRRTL memory lowering has a flag to generate attributes on memory 
implementations that preserve the behavior described in the verilog.  This is 
not a general solution, this bug could impact anyone making memory-looking 
things.  It was decided not to try to reverse engineer the conditions which
cause the bug to manifest (since they are version dependent), thus there isn't
a universal fix that can be applied in the generated verilog.

https://github.com/llvm/circt/commit/e9f443be475e0ef796c0c6af1ce09d6e783fcd5a

# Inline memory module wrappers

By default, firtool wraps each memory in its own module and instantiates it inside the parent module.

Vivado, for example, does not merge memories with shift registers at their outputs when there is a module boundary between them. This can make it harder to achieve good timing.

## Example

Input FIRRTL:

```
FIRRTL version 3.0.0
circuit Foo:
  module Foo:
    input r: {addr: UInt<3>, en: UInt<1>, clk: Clock, flip data: UInt<32>}
    input w: {addr: UInt<3>, en: UInt<1>, clk: Clock, data: UInt<32>, mask: UInt<1>}

    mem memory :
      data-type => UInt<32>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => r
      writer => w
      read-under-write => undefined


    connect memory.r, r
    connect memory.w, w
```

Default output Verilog. The memory is inside a newly created module: 

```Verilog
module memory_8x32(
  input  [2:0]  R0_addr,
  input         R0_en,
                R0_clk,
  output [31:0] R0_data,
  input  [2:0]  W0_addr,
  input         W0_en,
                W0_clk,
  input  [31:0] W0_data
);

  reg [31:0] Memory[0:7];
  reg        _R0_en_d0;
  reg [2:0]  _R0_addr_d0;
  always @(posedge R0_clk) begin
    _R0_en_d0 <= R0_en;
    _R0_addr_d0 <= R0_addr;
  end // always @(posedge)
  always @(posedge W0_clk) begin
    if (W0_en & 1'h1)
      Memory[W0_addr] <= W0_data;
  end // always @(posedge)
  assign R0_data = _R0_en_d0 ? Memory[_R0_addr_d0] : 32'bx;
endmodule

module Foo(
  input  [2:0]  r_addr,
  input         r_en,
                r_clk,
  output [31:0] r_data,
  input  [2:0]  w_addr,
  input         w_en,
                w_clk,
  input  [31:0] w_data,
  input         w_mask
);

  memory_8x32 memory_ext (
    .R0_addr (r_addr),
    .R0_en   (r_en),
    .R0_clk  (r_clk),
    .R0_data (r_data),
    .W0_addr (w_addr),
    .W0_en   (w_en & w_mask),
    .W0_clk  (w_clk),
    .W0_data (w_data)
  );
endmodule
```

## Workaround

Compile with `firtool --inline-mem` to inline the content of the memory wrapper module into the parent module.

With this option, the memory is emitted inside the parent module.

```Verilog
module Foo(
  input  [2:0]  r_addr,
  input         r_en,
                r_clk,
  output [31:0] r_data,
  input  [2:0]  w_addr,
  input         w_en,
                w_clk,
  input  [31:0] w_data,
  input         w_mask
);

  reg [31:0] memory_ext_Memory[0:7];
  reg        memory_ext__R0_en_d0;
  reg [2:0]  memory_ext__R0_addr_d0;
  always @(posedge r_clk) begin
    memory_ext__R0_en_d0 <= r_en;
    memory_ext__R0_addr_d0 <= r_addr;
  end // always @(posedge)
  always @(posedge w_clk) begin
    if (w_en & w_mask & 1'h1)
      memory_ext_Memory[w_addr] <= w_data;
  end // always @(posedge)
  assign r_data =
    memory_ext__R0_en_d0 ? memory_ext_Memory[memory_ext__R0_addr_d0] : 32'bx;
endmodule
```
