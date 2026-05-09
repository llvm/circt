; RUN: firtool %s --split-input-file --strip-debug-info | FileCheck %s

; Test for register under inline layer.

FIRRTL version 5.0.0

circuit RegUnderInlineLayer:
  layer A, inline:

  public module RegUnderInlineLayer:
    input reset: UInt<1>
    input clock: Clock
    input data: UInt<1>
    output probe: Probe<UInt<1>, A>

    ; CHECK: `ifdef layer$A
    ; CHECK:   reg  myreg;
    ; CHECK:   always @(posedge clock) begin
    ; CHECK:     if (reset)
    ; CHECK:       myreg <= 1'h0;
    ; CHECK:     else
    ; CHECK:       myreg <= data;
    ; CHECK:   end // always @(posedge)
    ; CHECK: `endif // layer$A

    ; CHECK: initial begin
    ; CHECK:   automatic logic [31:0] _RANDOM[0:0];
    ; CHECK:   _RANDOM[/*Zero width*/ 1'b0] = `RANDOM;
    ; CHECK:   `ifdef layer$A
    ; CHECK:     RegUnderInlineLayer.myreg = _RANDOM[/*Zero width*/ 1'b0][0];
    ; CHECK:   `endif // layer$A
    ; CHECK:    end // initial
    layerblock A:
      regreset myreg : UInt<1>, clock, reset, UInt<1>(0)
      connect myreg, data
      define probe = probe(myreg)

; // -----

; Test for register under nested inline layer.

FIRRTL version 5.0.0

circuit RegUnderInlineLayer:
  layer A, inline:
    layer B, inline:

  public module RegUnderInlineLayer:
    input reset: UInt<1>
    input clock: Clock
    input data: UInt<1>
    output probe: Probe<UInt<1>, A.B>

    ; CHECK: `ifdef layer$A
    ; CHECK:   `ifdef layer$A$B
    ; CHECK:     reg  myreg;
    ; CHECK:     always @(posedge clock) begin
    ; CHECK:       if (reset)
    ; CHECK:         myreg <= 1'h0;
    ; CHECK:       else
    ; CHECK:         myreg <= data;
    ; CHECK:     end // always @(posedge)
    ; CHECK:   `endif // layer$A$B
    ; CHECK: `endif // layer$A

    ; CHECK: initial begin
    ; CHECK:   automatic logic [31:0] _RANDOM[0:0];
    ; CHECK:   _RANDOM[/*Zero width*/ 1'b0] = `RANDOM;
    ; CHECK:   `ifdef layer$A
    ; CHECK:     `ifdef layer$A$B
    ; CHECK:       RegUnderInlineLayer.myreg = _RANDOM[/*Zero width*/ 1'b0][0];
    ; CHECK:     `endif // layer$A$B
    ; CHECK:   `endif // layer$A
    ; CHECK:    end // initial
    layerblock A:
      layerblock B:
        regreset myreg : UInt<1>, clock, reset, UInt<1>(0)
        connect myreg, data
        define probe = probe(myreg)

; // -----

; Test for register under inline layer, which is under a bind layer.

FIRRTL version 5.0.0

circuit RegUnderInlineLayer:
  layer A, bind:
    layer B, inline:

  ; Within the extracted bind-layer, there is an ifdef for layer b, guarding
  ; the register and its initialization.

  ; CHECK: module RegUnderInlineLayer_A();
  ; CHECK:   `ifdef layer$A$B
  ; CHECK:     reg  myreg;
  ; CHECK:     always @(posedge RegUnderInlineLayer.clock) begin
  ; CHECK:       if (RegUnderInlineLayer.reset)
  ; CHECK:         myreg <= 1'h0;
  ; CHECK:       else
  ; CHECK:         myreg <= RegUnderInlineLayer.data;
  ; CHECK:     end // always @(posedge)
  ; CHECK:   `endif // layer$A$B
  ; CHECK:     initial begin
  ; CHECK:       automatic logic [31:0] _RANDOM[0:0];
  ; CHECK:       `ifdef RANDOMIZE_REG_INIT
  ; CHECK:         _RANDOM[/*Zero width*/ 1'b0] = `RANDOM;
  ; CHECK:         `ifdef layer$A$B
  ; CHECK:           RegUnderInlineLayer_A.myreg = _RANDOM[/*Zero width*/ 1'b0][0];
  ; CHECK:         `endif // layer$A$B
  ; CHECK:       `endif // RANDOMIZE_REG_INIT
  ; CHECK:     end // initial
  ; CHECK: endmodule

  public module RegUnderInlineLayer:
    input reset: UInt<1>
    input clock: Clock
    input data: UInt<1>
    output probe: Probe<UInt<1>, A.B>

    layerblock A:
      layerblock B:
        regreset myreg : UInt<1>, clock, reset, UInt<1>(0)
        connect myreg, data
        define probe = probe(myreg)
