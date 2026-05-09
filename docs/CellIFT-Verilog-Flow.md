# Instrumenting a Verilog Design with CellIFT (CIRCT)

This guide explains how to instrument an arbitrary Verilog design with
cell-level dynamic information flow tracking (CellIFT) using the CIRCT
implementation.

## Prerequisites

For the direct CIRCT flow, build `circt-opt`, `circt-verilog`, and the Slang
frontend support:

```bash
cd /path/to/circt
cmake -G Ninja llvm/llvm -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=circt \
  -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD \
  -DCIRCT_SLANG_FRONTEND_ENABLED=ON
ninja -C build bin/circt-opt bin/circt-verilog
```

`lld` is optional. If you previously configured the tree with
`-DLLVM_ENABLE_LLD=ON` and CMake now fails because `lld` is unavailable or
unsupported on your toolchain, rerun the same configure command with
`-DLLVM_ENABLE_LLD=OFF` to clear the cached value.

For the fallback Yosys-based flow, build `circt-opt` and `firtool`:

```bash
cd /path/to/circt
ninja -C build bin/circt-opt bin/firtool
```

You also need [Yosys](https://github.com/YosysHQ/yosys) installed if you use
the fallback Verilog-to-FIRRTL conversion path.

## Recommended Direct Flow (no Yosys)

If `circt-verilog` is available, you can feed Verilog or SystemVerilog directly
to CIRCT and skip Yosys entirely.

### 1. Import Verilog directly to HW/Comb/Seq MLIR

```bash
build/bin/circt-verilog your_design.sv --top your_top_module -o design_hw.mlir
```

For designs that use only Verilog-2005, `.v` works as well:

```bash
build/bin/circt-verilog your_design.v --top your_top_module -o design_hw.mlir
```

`circt-verilog` lowers the source through CIRCT's Verilog frontend into
`hw.module`, `comb.*`, and `seq.*` operations, which is exactly the IR shape the
CellIFT pass expects.

### 2. Apply CellIFT instrumentation

```bash
build/bin/circt-opt --cellift-instrument design_hw.mlir -o design_instrumented.mlir
```

### 3. Export instrumented Verilog

```bash
build/bin/circt-opt --export-verilog design_instrumented.mlir -o /dev/null \
  > design_instrumented.sv
```

Keep `-o /dev/null` so stdout contains only Verilog rather than Verilog plus
the final MLIR dump.

Or combine the last two steps:

```bash
build/bin/circt-opt --cellift-instrument --export-verilog design_hw.mlir \
  -o /dev/null > design_instrumented.sv
```

### Direct-flow one-liner

```bash
build/bin/circt-verilog your_design.sv --top your_top_module \
  | build/bin/circt-opt --cellift-instrument \
  | build/bin/circt-opt --export-verilog -o /dev/null
```

## Fallback Flow (via Yosys)

Use this if `circt-verilog` is not built, or if you specifically want Yosys to
normalize the source first.

## Step-by-step Flow

### 1. Convert Verilog to FIRRTL (via Yosys)

Yosys can read Verilog and write FIRRTL. Run synthesis passes to normalize the
design before importing it back into CIRCT:

```bash
yosys -p "
  read_verilog your_design.v;
  hierarchy -top your_top_module;
  proc; opt;
  write_firrtl design.fir
"
```

### 2. Add FIRRTL version header

CIRCT's firtool expects a version header:

```bash
echo 'FIRRTL version 2.0.0' | cat - design.fir > design_v2.fir
```

### 3. Convert FIRRTL to HW-level MLIR

Use `firtool` to lower FIRRTL to the HW/Comb/Seq dialects:

```bash
build/bin/firtool design_v2.fir --ir-hw -o design_hw.mlir
```

This produces MLIR with `hw.module`, `comb.*`, and `seq.*` operations.

### 4. Apply CellIFT instrumentation

Run the CellIFT pass on the HW-level MLIR:

```bash
build/bin/circt-opt --cellift-instrument design_hw.mlir -o design_instrumented.mlir
```

This:
- Adds taint ports (`_t` suffix) to every integer input/output of every module
- Instruments every combinational operation with precise taint propagation logic
- Instruments registers to track taint through sequential elements
- Updates all instances to pass taint signals through the hierarchy

#### Options

- `--cellift-instrument='taint-suffix=_taint'` — change the taint port suffix
  (default: `_t`)
- `--cellift-instrument='taint-constants=true'` — mark hardware constants as
  tainted (useful for tracking constant influences)

### 5. Export instrumented Verilog

Use `circt-opt` to emit Verilog from the instrumented MLIR:

```bash
build/bin/circt-opt --export-verilog design_instrumented.mlir -o /dev/null \
  > design_instrumented.sv
```

Keep `-o /dev/null` so stdout contains only Verilog rather than Verilog plus
the final MLIR dump.

Or combine steps 4 and 5 in a single command:

```bash
build/bin/circt-opt --cellift-instrument --export-verilog design_hw.mlir \
  -o /dev/null > design_instrumented.sv
```

### Full one-liner

```bash
build/bin/firtool design_v2.fir --ir-hw 2>/dev/null \
  | build/bin/circt-opt --cellift-instrument 2>/dev/null \
  | build/bin/circt-opt --export-verilog -o /dev/null
```

## Which Flow to Use

- Use the direct CIRCT flow when `circt-verilog` is available. It is shorter and
  removes the Yosys/FIRRTL detour.
- Use the Yosys flow when the Slang frontend is not enabled in your build.
- Use the Yosys flow when you explicitly want Yosys to canonicalize the design
  before CellIFT instrumentation.

## Example

Given `adder.v`:
```verilog
module adder (
  input  [7:0] a, b,
  output [7:0] y
);
  assign y = a + b;
endmodule
```

The instrumented output is:
```verilog
module adder(
  input  [7:0] a, a_t, b, b_t,
  output [7:0] y, y_t
);
  assign y = a + b;
  assign y_t = (a & ~a_t) + (b & ~b_t) ^ (a | a_t) + (b | b_t) | a_t | b_t;
endmodule
```

The taint formula computes the **precise** bit-level taint: for each output
bit, the taint is 1 if and only if changing any tainted input bit could change
that output bit.

## Taint Rules

The implementation matches the [Yosys CellIFT](https://github.com/comsec-group/cellift-yosys) rules:

| Operation | Rule | Precision |
|-----------|------|-----------|
| AND       | `(a & b_t) \| (b & a_t) \| (a_t & b_t)` | Precise |
| OR        | `(~a & b_t) \| (~b & a_t) \| (a_t & b_t)` | Precise |
| XOR       | `a_t \| b_t` | Precise |
| ADD       | `((a&~a_t)+(b&~b_t)) ^ ((a\|a_t)+(b\|b_t)) \| a_t \| b_t` | Precise |
| SUB       | `((a\|a_t)-(b&~b_t)) ^ ((a&~a_t)-(b\|b_t)) \| a_t \| b_t` | Precise |
| MUL       | `replicate(reduce_or(a_t) \| reduce_or(b_t))` | Conservative |
| MUX       | `mux(s, t_t, f_t) \| replicate(s_t) & (t^f \| t_t \| f_t)` | Precise |
| EQ/NE     | `has_taint & (masked_a == masked_b)` | Precise |
| GE/GT/LE/LT | `cmp(min_a, max_b) ^ cmp(max_a, min_b)` | Precise |
| SHL/SHR   | Taint broadcase if offset tainted, else shift taints | Conservative |
| MOD       | Conservative (full taint broadcast) | Conservative |
| DIV       | Conservative (full taint broadcast) | Conservative |

## Using Taint in Simulation

In your testbench, drive the `_t` input ports to mark which input bits are
tainted:

```verilog
// Mark input 'a' as fully tainted, 'b' as untainted
adder dut (.a(a), .a_t(8'hFF), .b(b), .b_t(8'h00), .y(y), .y_t(y_t));

// Check taint propagation
always @(posedge clk) begin
  if (y_t != 0)
    $display("Output is tainted: y_t = %h", y_t);
end
```

## Writing MLIR Directly

You can also write HW/Comb/Seq MLIR directly and skip the Yosys/FIRRTL steps:

```mlir
hw.module @my_design(in %a : i8, in %b : i8, out y : i8) {
  %sum = comb.add %a, %b : i8
  hw.output %sum : i8
}
```

Then instrument with:
```bash
build/bin/circt-opt --cellift-instrument my_design.mlir
```
