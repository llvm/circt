# Intrinsics

Intrinsics provide an implementation-specific way to extend the FIRRTL language
with new operations.

Intrinsics are currently implemented as `intmodule`'s.

## Motivation

Intrinsics provide a way to add functionality to FIRRTL without having to extend
the FIRRTL language. This allows a fast path for prototyping new operations to 
rapidly respond to output requirements.  Intrinsics maintain strict definitions
and type checking.

## Supported Intrinsics


### circt_sizeof

Returns the size of a type.  The input port is not read from and may be any 
type, including uninferred types.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| i          | input     | Any      | value whose type is to be returned  |
| size       | output    | UInt<32> | Size of type of i                   |

### circt_isX

Tests if the value is a literal `x`.  FIRRTL doesn't have a notion of 'x per-se, 
but x can come in to the system from external modules and from SV constructs.  
Verification constructs need to explicitly test for 'x.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| i          | input     | Any      | value test                          |
| found      | output    | UInt<1>  | i is `x`                            |

### circt_plusargs_value

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

### circt_plusargs_test

Tests simulator command line options with SystemVerilog `$test$plusargs`.  This
is described in SystemVerilog 2012 section 21.6.

| Parameter  | Type   | Description                                       |
| ---------- | ------ | -------------                                     |
| FORMAT     | string | Format string per SV 21.6                         |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| found      | output    | UInt<1>  | found in args                       |

### circt_clock_gate

Enables and disables a clock safely, without glitches, based on a boolean enable value. If the enable input is 1, the output clock produced by the clock gate is identical to the input clock. If the enable input is 0, the output clock is a constant zero.

The enable input is sampled at the rising edge of the input clock; any changes on the enable before or after that edge are ignored and do not affect the output clock.

| Parameter | Type | Description |
| --------- | ---- | ----------- |

| Port | Direction | Type     | Description                 |
| ---- | --------- | -------- | --------------------------- |
| in   | input     | Clock    | input clock                 |
| en   | input     | UInt<1>  | enable for the output clock |
| out  | output    | Clock    | gated output clock          |

### circt_chisel_assert

Generate a clocked SV assert statement, with optional formatted error message.

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
```

### circt_chisel_ifelsefatal

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

### circt_chisel_assume

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

### circt_chisel_cover

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

### circt_unclocked_assume

Generate a SV assume statement whose predicate is used in a sensitivity list of the enclosing always block.


| Parameter | Type   | Description                                                                         |
| --------- | ------ | ----------------------------------------------------------------------------------- |
| format    | string | Format string per SV 20.10, 21.2.1.  Optional.                                      |
| label     | string | Label for assume statement.  Optional.                                              |
| guards    | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards.  Optional. |

| Port      | Direction | Type     | Description                |
| --------- | --------- | -------- | -------------------------- |
| predicate | input     | UInt<1>  | predicate to assume        |
| enable    | input     | UInt<1>  | enable signal              |
| ...       | input     | Signals  | arguments to format string |

Example SV output:
```systemverilog
ifdef USE_FORMAL_ONLY_CONSTRAINTS
 `ifdef USE_UNR_ONLY_CONSTRAINTS
   wire _GEN = ~enable | pred;
   always @(edge _GEN)
     assume_label: assume(_GEN) else $error("Conditional compilation example for UNR-only and formal-only assert");
 `endif // USE_UNR_ONLY_CONSTRAINTS
endif // USE_FORMAL_ONLY_CONSTRAINTS
```

### circt_dpi_call

Call a DPI function. `clock` is optional and if `clock` is not provided,
the callee is invoked when input values are changed.
If provided, the dpi function is called at clock's posedge. The result values behave
like registers and the DPI function is used as a state transfer function of them.

`enable` operand is used to conditionally call the DPI since DPI call could be quite
more expensive than native constructs. When `enable` is low, results of unclocked
calls are undefined and evaluated into `X`. Users are expected to gate result values
by another `enable` to model a default value of results.

For clocked calls, a low enable means that its register state transfer function is
not called. Hence their values will not be modify in that clock.

| Parameter     | Type   | Description                                         |
| ------------- | ------ | --------------------------------------------------- |
| isClocked     | int    | Set 1 if the dpi call is clocked.                   |
| functionName  | string | Specify the function name.                          |
| inputNames    | string | Semicolon-delimited list of input names.  Optional. |
| outputName    | string | Output name. Optional.                              |


| Port              | Direction | Type     | Description                     |
| ----------------- | --------- | -------- | ------------------------------- |
| clock (optional)  | input     | Clock    | Optional clock operand          |
| enable            | input     | UInt<1>  | Enable signal                   |
| ...               | input     | Signals  | Arguments to DPI function call  |
| result (optional) | output    | Signal   | Optional result of the dpi call |

##### DPI Intrinsic ABI
Function Declaration:
* Imported DPI function must be a void function that has input arguments which correspond to operand types, and an output argument which correspond to a result type.
* Output argument must be a last argument.

Types:
* Operand and result types must be passive.
* A vector is lowered to an unpacked open array type, e.g. `a: Vec<4, UInt<8>>` to `byte a []`.
* A bundle is lowered to a packed struct.
* Integer types are lowered into into 2-state types.
* Small integer types (< 64 bit) must be compatible to C-types and arguments are passed by values. Users are required to use specific integer types for small integers shown in the table below. Large integers are lowered to `bit` and passed by a reference.

| Width | Verilog Type | Argument Passing Modes |
| ----- | ------------ | ---------------------- |
| 1     | bit          | value                  |
| 8     | byte         | value                  |
| 16    | shortint     | value                  |
| 32    | int          | value                  |
| 64    | longint      | value                  |
| > 64  | bit [w-1:0]  | reference              |

Example SV output:
```firrtl
node result = intrinsic(circt_dpi_call<isClocked = 1, functionName="dpi_func"> : UInt<64>, clock, enable, uint_8_value, uint_32_value, uint_8_vector)
```
```verilog
import "DPI-C" function void dpi_func(
  input  byte    in_0,
         int     in_1,
         byte    in_2[],
  output longint out_0
);

...

logic [63:0] _dpi_func_0;
reg   [63:0] _GEN;
always @(posedge clock) begin
  if (enable) begin
    dpi_func(in1, in2, _dpi_func_0);
    _GEN <= _dpi_func_0;
  end
end
```

### circt_view

This will become a SystemVerilog Interface that is driven by its arguments.
This is _not_ a true SystemVerilog Interface, it is only lowered to one.


| Parameter     | Type   | Description                                         |
| ------------- | ------ | --------------------------------------------------- |
| name          | string | Instance name of the view.                          |
| view          | string | JSON encoding the view structure.                   |

| Port       | Direction | Type     | Description                         |
| ---------- | --------- | -------- | ----------------------------------- |
| ...        | input     | Ground   | Leaf ground values for the view     |

The structure of the view is encoded using JSON, with the top-level object
required to be an `AugmentedBundleType`.

The intrinsic operands correspond to the `AugmentedGroundType` leaves,
and must be of ground type.

This encoding is a trimmed version of what's used for the old GrandCentral View
annotation.

These views do not interact with any GrandCentral annotations or options governing
the behavior of the annotation-based views.

Output directory behavior is a work in progress presently.

#### AugmentedGroundType

| Property   | Type   | Description                                          |
| ---------- | ------ | -------------                                        |
| class      | string | `sifive.enterprise.grandcentral.AugmentedGroundType` |

Creates a SystemVerilog logic type.

Each ground type corresponds to an operand to the view intrinsic.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.AugmentedGroundType"
}
```

#### AugmentedVectorType

| Property   | Type   | Description                                          |
| ---------- | ------ | -------------                                        |
| class      | string | `sifive.enterprise.grandcentral.AugmentedVectorType` |
| elements   | array  | List of augmented types.

Creates a SystemVerilog unpacked array.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.AugmentedVectorType",
  "elements": [
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType"
    },
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType"
    }
  ]
}
```

#### AugmentedField

| Property    | Type   | Description                        |
| ----------  | ------ | -------------                      |
| name        | string | Name of the field                  |
| description | string | A textual description of this type |
| tpe         | string | A nested augmented type            |

A field in an augmented bundle type.  This can provide a small description of
what the field in the bundle is.

#### AugmentedBundleType

| Property   | Type   | Description                                               |
| ---------- | ------ | -------------                                             |
| class      | string | sifive.enterprise.grandcentral.AugmentedBundleType        |
| defName    | string | The name of the SystemVerilog interface.  May be renamed. |
| elements   | array  | List of AugmentedFields                                   |

Creates a SystemVerilog interface for each bundle type.
