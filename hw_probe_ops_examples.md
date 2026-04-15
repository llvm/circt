# HW Probe Operations - Code Examples

This document provides concrete examples of how HW probe operations will work in practice.

## Example 1: Basic Probe Send and Resolve

### FIRRTL Input
```mlir
firrtl.circuit "Top" {
  firrtl.module @Child(out %ref_out: !firrtl.probe<uint<8>>) {
    %wire = firrtl.wire : !firrtl.uint<8>
    %const = firrtl.constant 42 : !firrtl.uint<8>
    firrtl.matchingconnect %wire, %const : !firrtl.uint<8>
    
    %ref = firrtl.ref.send %wire : !firrtl.uint<8>
    firrtl.ref.define %ref_out, %ref : !firrtl.probe<uint<8>>
  }
  
  firrtl.module @Top() {
    %child_ref = firrtl.instance child @Child(out ref_out: !firrtl.probe<uint<8>>)
    %value = firrtl.ref.resolve %child_ref : !firrtl.probe<uint<8>>
    // use %value...
  }
}
```

### After LowerToHW
```mlir
hw.module @Child(out ref_out: !hw.probe<i8>) {
  %c42_i8 = hw.constant 42 : i8
  %wire = hw.wire %c42_i8 : i8
  
  %ref = hw.probe.send %wire : i8
  hw.output %ref : !hw.probe<i8>
}

hw.module @Top() {
  %child_ref = hw.instance "child" @Child() -> (ref_out: !hw.probe<i8>)
  %value = hw.probe.resolve %child_ref : !hw.probe<i8>
  // use %value...
}
```

### After HW LowerXMR
```mlir
hw.hierpath private @xmr_path [@Top::@child, @Child::@wire_sym]

hw.module @Child() {
  %c42_i8 = hw.constant 42 : i8
  %wire = hw.wire sym @wire_sym %c42_i8 : i8
  // probe port removed
}

hw.module @Top() {
  hw.instance "child" sym @child @Child() -> ()
  %xmr = sv.xmr.ref @xmr_path : !hw.inout<i8>
  %value = sv.read_inout %xmr : !hw.inout<i8>
  // use %value...
}
```

### Final Verilog
```verilog
module Child();
  wire [7:0] wire_0 = 8'd42;
endmodule

module Top();
  Child child ();
  wire [7:0] value = Top.child.wire_0;  // XMR
endmodule
```

## Example 2: Aggregate Probe with Subindex

### FIRRTL Input
```mlir
firrtl.module @ArrayProbe(out %elem_ref: !firrtl.probe<uint<4>>) {
  %arr = firrtl.wire : !firrtl.vector<uint<4>, 4>
  // ... initialize array ...
  
  %arr_ref = firrtl.ref.send %arr : !firrtl.vector<uint<4>, 4>
  %elem2_ref = firrtl.ref.sub %arr_ref[2] : !firrtl.probe<vector<uint<4>, 4>>
  firrtl.ref.define %elem_ref, %elem2_ref : !firrtl.probe<uint<4>>
}
```

### After LowerToHW
```mlir
hw.module @ArrayProbe(out elem_ref: !hw.probe<i4>) {
  %arr = hw.wire : !hw.array<4xi4>
  // ... initialize array ...
  
  %arr_ref = hw.probe.send %arr : !hw.array<4xi4>
  %elem2_ref = hw.probe.sub %arr_ref[2] : !hw.probe<!hw.array<4xi4>>
  hw.output %elem2_ref : !hw.probe<i4>
}
```

### After HW LowerXMR
```mlir
hw.hierpath private @xmr_path [@Parent::@child, @ArrayProbe::@arr_sym]

hw.module @ArrayProbe() {
  %arr = hw.wire sym @arr_sym : !hw.array<4xi4>
  // probe port removed
}

hw.module @Parent() {
  hw.instance "child" sym @child @ArrayProbe() -> ()
  %xmr = sv.xmr.ref @xmr_path : !hw.inout<!hw.array<4xi4>>
  %elem = hw.array_get %xmr[%c2] : !hw.inout<!hw.array<4xi4>>
  %value = sv.read_inout %elem : !hw.inout<i4>
}
```

## Example 3: RWProbe with Force

### FIRRTL Input
```mlir
firrtl.module @DUT(out %probe_out: !firrtl.rwprobe<uint<8>>) {
  %reg, %reg_ref = firrtl.reg sym @myReg forceable %clock : !firrtl.clock, 
                   !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
  firrtl.ref.define %probe_out, %reg_ref : !firrtl.rwprobe<uint<8>>
}

firrtl.module @Top() {
  %dut_probe = firrtl.instance dut @DUT(out probe_out: !firrtl.rwprobe<uint<8>>)
  
  %force_val = firrtl.constant 123 : !firrtl.uint<8>
  %true = firrtl.constant 1 : !firrtl.uint<1>
  firrtl.ref.force_initial %true, %dut_probe, %force_val : 
    !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
}
```

### After LowerToHW
```mlir
hw.module @DUT(out probe_out: !hw.rwprobe<i8>) {
  %reg = seq.firreg %clock : i8
  %reg_ref = hw.probe.rwprobe @DUT::@myReg : !hw.rwprobe<i8>
  hw.output %reg_ref : !hw.rwprobe<i8>
}

hw.module @Top() {
  %dut_probe = hw.instance "dut" @DUT() -> (probe_out: !hw.rwprobe<i8>)
  
  %c123_i8 = hw.constant 123 : i8
  %true = hw.constant true
  hw.probe.force_initial %true, %dut_probe, %c123_i8 : 
    i1, !hw.rwprobe<i8>, i8
}
```

### After HW LowerXMR
```mlir
hw.hierpath private @force_path [@Top::@dut, @DUT::@myReg]

hw.module @DUT() {
  %reg = seq.firreg sym @myReg %clock : i8
  // probe port removed
}

hw.module @Top() {
  hw.instance "dut" sym @dut @DUT() -> ()
  
  %c123_i8 = hw.constant 123 : i8
  %true = hw.constant true
  %xmr = sv.xmr.ref @force_path : !hw.inout<i8>
  
  sv.initial {
    sv.if %true {
      sv.force %xmr, %c123_i8 : i8
    }
  }
}
```

### Final Verilog
```verilog
module DUT(input clock);
  reg [7:0] myReg;
  always @(posedge clock)
    myReg <= ...;
endmodule

module Top();
  DUT dut (.clock(clock));

  initial begin
    if (1'b1) begin
      force Top.dut.myReg = 8'd123;
    end
  end
endmodule
```

## Example 4: Probe Type Casting

### RWProbe to Probe Demotion
```mlir
// After LowerToHW
hw.module @Example(in %rwprobe: !hw.rwprobe<i32>, out probe: !hw.probe<i32>) {
  // Demote RWProbe to Probe (read-only)
  %ro_probe = hw.probe.cast %rwprobe : !hw.rwprobe<i32> -> !hw.probe<i32>
  hw.output %ro_probe : !hw.probe<i32>
}
```

### Width-Compatible Casting
```mlir
hw.module @WidthCast() {
  %wire = hw.wire : i8
  %ref = hw.probe.send %wire : i8

  // Cast to wider type (like upcasting)
  %wider = hw.probe.cast %ref : !hw.probe<i8> -> !hw.probe<i32>

  // When resolved, appropriate padding/extension happens
  %val = hw.probe.resolve %wider : !hw.probe<i32>
}
```

## Example 5: Multi-Level Hierarchy XMR

### Input Hierarchy
```mlir
hw.module @Leaf() {
  %data = hw.wire : i16
  %ref = hw.probe.send %data : i16
  hw.output %ref : !hw.probe<i16>
}

hw.module @Middle() {
  %leaf_ref = hw.instance "leaf" @Leaf() -> (!hw.probe<i16>)
  hw.output %leaf_ref : !hw.probe<i16>
}

hw.module @Top() {
  %middle_ref = hw.instance "mid" @Middle() -> (!hw.probe<i16>)
  %value = hw.probe.resolve %middle_ref : !hw.probe<i16>
}
```

### After HW LowerXMR
```mlir
hw.hierpath private @deep_path [@Top::@mid, @Middle::@leaf, @Leaf::@data_sym]

hw.module @Leaf() {
  %data = hw.wire sym @data_sym : i16
}

hw.module @Middle() {
  hw.instance "leaf" sym @leaf @Leaf() -> ()
}

hw.module @Top() {
  hw.instance "mid" sym @mid @Middle() -> ()
  %xmr = sv.xmr.ref @deep_path : !hw.inout<i16>
  %value = sv.read_inout %xmr : !hw.inout<i16>
}
```

### Final Verilog
```verilog
module Leaf();
  wire [15:0] data;
endmodule

module Middle();
  Leaf leaf();
endmodule

module Top();
  Middle mid();
  wire [15:0] value = Top.mid.leaf.data;  // Multi-level XMR
endmodule
```

## Example 6: Struct Probe with Field Access

### HW Dialect
```mlir
!MyStruct = !hw.struct<a: i8, b: i16>

hw.module @StructProbe(out field_ref: !hw.probe<i16>) {
  %struct = hw.wire : !MyStruct

  %struct_ref = hw.probe.send %struct : !MyStruct
  %field_b_ref = hw.probe.sub %struct_ref[1] : !hw.probe<!MyStruct>
  hw.output %field_b_ref : !hw.probe<i16>
}

hw.module @Parent() {
  %b_ref = hw.instance "sp" @StructProbe() -> (!hw.probe<i16>)
  %b_value = hw.probe.resolve %b_ref : !hw.probe<i16>
}
```

### After HW LowerXMR
```mlir
hw.hierpath private @struct_b_path [@Parent::@sp, @StructProbe::@struct_sym]

hw.module @StructProbe() {
  %struct = hw.wire sym @struct_sym : !MyStruct
}

hw.module @Parent() {
  hw.instance "sp" sym @sp @StructProbe() -> ()
  %xmr = sv.xmr.ref @struct_b_path ".b" : !hw.inout<i16>
  %b_value = sv.read_inout %xmr : !hw.inout<i16>
}
```

### Verilog
```verilog
module StructProbe();
  struct packed {
    logic [7:0] a;
    logic [15:0] b;
  } struct_0;
endmodule

module Parent();
  StructProbe sp();
  wire [15:0] b_value = Parent.sp.struct_0.b;
endmodule
```

## Key Takeaways

1. **Clean Separation**: Each transformation step is well-defined
2. **Type Preservation**: Types flow naturally through the pipeline
3. **Symbol Management**: Symbols are added during XMR resolution
4. **Hierarchical Paths**: Built incrementally as references cross modules
5. **Verilog Compatibility**: Final output matches expected XMR syntax

## Testing Strategy

Each example above should have:
- Unit test for HW probe operations
- Conversion test (FIRRTL → HW)
- Lowering test (HW → SV)
- End-to-end test (FIRRTL → Verilog)
- Verification that Verilog is semantically correct
