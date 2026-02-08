; RUN: firtool %s --verilog | FileCheck %s --check-prefix=VERILOG
; RUN: firtool %s --ir-hw | FileCheck %s --check-prefix=HW

FIRRTL version 4.0.0

; Test the ReadDesignConfigInt intrinsic end-to-end
; This intrinsic allows creating configurable parameters that can be
; overridden after Verilog generation

circuit Top:
  extmodule Buffer:
    input data_in : UInt<8>
    input mode : UInt<2>
    output data_out : UInt<8>
    defname = Buffer
  public module Top:
    input clock: Clock
    input reset: UInt<1>
    input data_in: UInt<8>
    output data_out: UInt<8>
    output buffer_out: UInt<8>

    ; Create parameters with different default values
    wire mode : UInt<2>
    wire enable_feature : UInt<1>

    connect mode, intrinsic(circt_read_design_config_int<name = "mode", defaultValue = 3, comment = "Mode selection"> : UInt<2>)
    connect enable_feature, intrinsic(circt_read_design_config_int<name = "enable_feature", defaultValue = 1, comment = "Enable feature flag"> : UInt<1>)

    ; Use the parameters in logic
    inst buffer of Buffer
    connect buffer.data_in, data_in
    connect buffer.mode, mode
    connect data_out, buffer.data_out
    node selected_data = mux(enable_feature, data_in, UInt<8>(0))
    connect data_out, selected_data
    connect buffer_out, buffer.data_out

; VERILOG-LABEL: module Top(
; VERILOG:   localparam [1:0] mode = DesignConfigPackage::mode;
; VERILOG:   localparam [0:0] enable_feature = DesignConfigPackage::enable_feature;
; VERILOG:   Buffer buffer (
; VERILOG:     .data_in     (data_in),
; VERILOG:     .mode (mode),
; VERILOG:     .data_out    (buffer_out)
; VERILOG:   );
; VERILOG:   assign data_out = enable_feature ? data_in : 8'h0;

; VERILOG-LABEL: package DesignConfigPackage;
; VERILOG-NEXT:    // Mode selection
; VERILOG-NEXT:    parameter mode = 3;
; VERILOG-NEXT:    // Enable feature flag
; VERILOG-NEXT:    parameter enable_feature = 1;
; VERILOG-NEXT:  endpackage

; HW-LABEL: hw.module @Top
; HW:         %mode = sv.localparam {value = #hw.param.verbatim<"DesignConfigPackage::mode"> : i2} : i2
; HW:         %enable_feature = sv.localparam {value = #hw.param.verbatim<"DesignConfigPackage::enable_feature"> : i1} : i1

