# Intrinsics

Intrinsics provide an implementation-specific way to extend the FIRRTL language
with new operations.

Intrinsics are currently implemented as annotated external modules.  We expect
that native FIRRTL support for intrinsics will be added to the language.

## Motivation

Intrinsics provide a way to add functionality to FIRRTL without having to extend
the FIRRTL language. This allows a fast path for prototyping new operations to 
rapidly respond to output requirements.  Intrinsics maintain strict definitions
and type checking.

## Supported Intrinsics

Annotations here are written in their JSON format. A "reference target"
indicates that the annotation could target any object in the hierarchy,
although there may be further restrictions in the annotation.

We also accept the `intmodule` version of any of these annotation strings used
as the intrinsic name.  To work with the requirement of the intrinsic name being
an identifier, replace any period is with an underscore in the intrinsic name.

### circt.sizeof

Returns the size of a type.  The input port is not read from and may be any 
type, including uninferred types.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| i          | input     | Any      | value whose type is to be returned  |
| size       | output    | UInt<32> | Size of type of i                   |

### circt.isX

Tests if the value is a literal `x`.  FIRRTL doesn't have a notion of 'x per-se, 
but x can come in to the system from external modules and from SV constructs.  
Verification constructs need to explicitly test for 'x.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| i          | input     | Any      | value test                          |
| found      | output    | UInt<1>  | i is `x`                            |

### circt.plusargs.value

Tests and extracts a value from simulator command line options with SystemVerilog
`$value$plusargs`.  This is described in SystemVerilog 2012 section 21.6.

We do not currently check that the format string substitution flag matches the
type of the result.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |
| FORMAT     | string | Format string per SV 21.6                         |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| found      | output    | UInt<1>  | found in args                       |
| result     | output    | AnyType  | found in args                       |

### circt.plusargs.test

Tests simulator command line options with SystemVerilog `$test$plusargs`.  This
is described in SystemVerilog 2012 section 21.6.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |
| FORMAT     | string | Format string per SV 21.6                         |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| found      | output    | UInt<1>  | found in args                       |

### circt.clock_gate

Enables and disables a clock safely, without glitches, based on a boolean enable value. If the enable input is 1, the output clock produced by the clock gate is identical to the input clock. If the enable input is 0, the output clock is a constant zero.

The enable input is sampled at the rising edge of the input clock; any changes on the enable before or after that edge are ignored and do not affect the output clock.

| Parameter | Type | Description |
| --------- | ---- | ----------- |

| Port | Direction | Type     | Description                 |
| ---- | --------- | -------- | --------------------------- |
| in   | input     | Clock    | input clock                 |
| en   | input     | UInt<1>  | enable for the output clock |
| out  | output    | Clock    | gated output clock          |

### circt.chisel_assert_assume

Generate a clocked SV assertion with companion assume statement.

Has legacy special behavior and should not be used by new code.

| Parameter | Type   | Description                                                                         |
| --------- | ------ | ----------------------------------------------------------------------------------- |
| format    | string | Format string per SV 20.10, 21.2.1.  Optional.                                      |
| label     | string | Label for assert/assume.  Optional.                                                 |
| guards    | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards.  Optional. |

| Port      | Direction | Type     | Description                |
| --------- | --------- | -------- | -------------------------- |
| clock     | input     | Clock    | input clock                |
| predicate | input     | UInt<1>  | predicate to assert/assume |
| enable    | input     | UInt<1>  | enable signal              |
| ...       | input     | Signals  | arguments to format string |

Example output:
```systemverilog
wire _GEN = ~enable | cond;
assert__label: assert property (@(posedge clock) _GEN) else $error("message");
`ifdef USE_PROPERTY_AS_CONSTRAINT
  assume__label: assume property (@(posedge clock) _GEN);
`endif // USE_PROPERTY_AS_CONSTRAINT
```

### circt.chisel_ifelsefatal

Generate a particular Verilog sequence that's similar to an assertion.

Has legacy special behavior and should not be used by new code.

| Parameter | Type   | Description                                    |
| --------- | ------ | ---------------------------------------------- |
| format    | string | Format string per SV 20.10, 21.2.1.  Optional. |

This intrinsic also accepts the `label` and `guard` parameters which
are recorded but not used in the normal emission.

| Port      | Direction | Type     | Description                 |
| --------- | --------- | -------- | --------------------------- |
| clock     | input     | Clock    | input clock                 |
| predicate | input     | UInt<1>  | predicate to check          |
| enable    | input     | UInt<1>  | enable signal               |
| ...       | input     | Signals  | arguments to format string  |

Example SV output:
```systemverilog
`ifndef SYNTHESIS
  always @(posedge clock) begin
    if (enable & ~cond) begin
      if (`ASSERT_VERBOSE_COND_)
        $error("message");
      if (`STOP_COND_)
        $fatal;
    end
  end // always @(posedge)
`endif // not def SYNTHESIS
```

### circt.chisel_assume

Generate a clocked SV assume statement, with optional formatted error message.


| Parameter | Type   | Description                                                                         |
| --------- | ------ | ----------------------------------------------------------------------------------- |
| format    | string | Format string per SV 20.10, 21.2.1.  Optional.                                      |
| label     | string | Label for assume statement.  Optional.                                              |
| guards    | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards.  Optional. |

| Port      | Direction | Type     | Description                |
| --------- | --------- | -------- | -------------------------- |
| clock     | input     | Clock    | input clock                |
| predicate | input     | UInt<1>  | predicate to assume        |
| enable    | input     | UInt<1>  | enable signal              |
| ...       | input     | Signals  | arguments to format string |

Example SV output:
```systemverilog
assume__label: assume property (@(posedge clock) ~enable | cond) else $error("message");	
```

### circt.chisel_cover

Generate a clocked SV cover statement.

| Parameter | Type   | Description                                                                         |
| --------- | ------ | ----------------------------------------------------------------------------------- |
| label     | string | Label for cover statement.  Optional.                                               |
| guards    | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards.  Optional. |

| Port      | Direction | Type     | Description        |
| --------- | --------- | -------- | ------------------ |
| clock     | input     | Clock    | input clock        |
| predicate | input     | UInt<1>  | predicate to cover |
| enable    | input     | UInt<1>  | enable signal      |

Example SV output:
```systemverilog
cover__label: cover property (@(posedge clock) enable & cond);
```
