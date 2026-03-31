//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the table of FIRRTL intrinsics supported by CIRCT. This is distinct
// from the MLIR ops representing some of these intrinsics in
// `include/circt/Dialect/FIRRTL/FIRRTLIntrinsics.td`.
//
//===----------------------------------------------------------------------===//

// A FIRRTL intrinsic. Defs of this class generate part of the intrinsic
// lowering code, as well as fragments for the dialect documentation.
class FIRRTLIntrinsic {
  // The mnemonic for the intrinsic, e.g. "circt.sizeof". This should use the
  // form with dots. A second mnemonic with only underscores will be generated
  // automatically.
  string mnemonic;

  // The C++ class name (potentially including template parameters) that
  // converts the generic intrinsic to a concrete op during lowering. Defaults
  // to `<DefName>Converter`.
  code converter = ?;

  // A description of the intrinsic for documentation purposes. Markdown.
  string description;
}

//===----------------------------------------------------------------------===//
// Intrinsics
//===----------------------------------------------------------------------===//

def CirctSizeof : FIRRTLIntrinsic {
  let mnemonic = "circt.sizeof";
  let description = [{
    Returns the size of a type. The input port is not read from and may be any
    type, including uninferred types.

    | Argument | Type | Description                               |
    | -------- | ---- | ----------------------------------------- |
    | i        | Any  | value whose type's size is to be returned |

    | Result | Type     | Description       |
    | ------ | -------- | ----------------- |
    | size   | UInt<32> | Size of type of i |
  }];
}

def CirctIsX : FIRRTLIntrinsic {
  let mnemonic = "circt.isX";
  let description = [{
    Tests if the value is a literal `x`. FIRRTL doesn't have a notion of X
    per-se, but x can come in to the system from external modules and from SV
    constructs. Verification constructs need to explicitly test for X.

    | Argument | Type | Description   |
    | -------- | ---- | ------------- |
    | i        | Any  | value to test |

    | Result | Type    | Description |
    | ------ | ------- | ----------- |
    | found  | UInt<1> | i is `x`    |
  }];
}

def CirctPlusArgTest : FIRRTLIntrinsic {
  let mnemonic = "circt.plusargs.test";
  let description = [{
    Tests simulator command line options with SystemVerilog `$test$plusargs`.
    This is described in SystemVerilog 2012 section 21.6.

    | Parameter | Type   | Description               |
    | --------- | ------ | ------------------------- |
    | FORMAT    | string | Format string per SV 21.6 |

    | Result | Type    | Description   |
    | ------ | ------- | ------------- |
    | found  | UInt<1> | found in args |
  }];
}

def CirctPlusArgValue : FIRRTLIntrinsic {
  let mnemonic = "circt.plusargs.value";
  let description = [{
    Tests and extracts a value from simulator command line options with
    SystemVerilog `$value$plusargs`. This is described in SystemVerilog 2012
    section 21.6.

    We do not currently check that the format string substitution flag matches
    the type of the result.

    | Parameter | Type   | Description               |
    | --------- | ------ | ------------------------- |
    | FORMAT    | string | Format string per SV 21.6 |

    | Result | Type    | Description           |
    | ------ | ------- | --------------------- |
    | found  | UInt<1> | found in args         |
    | result | AnyType | value of the argument |
  }];
}

def CirctClockGate : FIRRTLIntrinsic {
  let mnemonic = "circt.clock_gate";
  let description = [{
    Enables and disables a clock safely, without glitches, based on a boolean
    enable value. If the enable input is 1, the output clock produced by the
    clock gate is identical to the input clock. If the enable input is 0, the
    output clock is a constant zero.

    The enable input is sampled at the rising edge of the input clock; any
    changes on the enable before or after that edge are ignored and do not
    affect the output clock.

    | Argument | Type    | Description                 |
    | -------- | ------- | --------------------------- |
    | in       | Clock   | input clock                 |
    | en       | UInt<1> | enable for the output clock |

    | Result | Type  | Description        |
    | ------ | ----- | ------------------ |
    | out    | Clock | gated output clock |
  }];
}

def CirctClockInverter : FIRRTLIntrinsic {
  let mnemonic = "circt.clock_inv";
  let description = [{
    Inverts a clock signal. The output clock is the logical inverse of the input
    clock: when the input is high the output is low, and vice versa.

    | Argument | Type  | Description |
    | -------- | ----- | ----------- |
    | in       | Clock | input clock |

    | Result | Type  | Description    |
    | ------ | ----- | -------------- |
    | out    | Clock | inverted clock |
  }];
}

def CirctClockDivider : FIRRTLIntrinsic {
  let mnemonic = "circt.clock_div";
  let description = [{
    Divides a clock frequency by a power of two. The division factor is
    specified by the `POW_2` parameter.

    | Parameter | Type | Description                       |
    | --------- | ---- | --------------------------------- |
    | POW_2     | int  | log2 of the clock division factor |

    | Argument | Type  | Description |
    | -------- | ----- | ----------- |
    | in       | Clock | input clock |

    | Result | Type  | Description   |
    | ------ | ----- | ------------- |
    | out    | Clock | divided clock |
  }];
}

def CirctLTLDelay : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.delay";
  let description = [{
    Delays the occurrence of a boolean or sequence by a specified number of
    cycles, with an optional bound on the length of the delay.

    | Parameter | Type | Description                               |
    | --------- | ---- | ----------------------------------------- |
    | delay     | int  | minimum number of cycles to delay         |
    | length    | int  | maximum additional delay cycles; optional |

    | Argument | Type    | Description               |
    | -------- | ------- | ------------------------- |
    | in       | UInt<1> | input boolean or sequence |

    | Result | Type    | Description      |
    | ------ | ------- | ---------------- |
    | out    | UInt<1> | delayed sequence |
  }];
}

def CirctLTLRepeat : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.repeat";
  let description = [{
    Repeats a boolean or sequence a specified number of times consecutively,
    with an optional bound on the number of additional repetitions.

    | Parameter | Type | Description                               |
    | --------- | ---- | ----------------------------------------- |
    | base      | int  | minimum number of consecutive repetitions |
    | more      | int  | maximum additional repetitions; optional  |

    | Argument | Type    | Description               |
    | -------- | ------- | ------------------------- |
    | in       | UInt<1> | input boolean or sequence |

    | Result | Type    | Description       |
    | ------ | ------- | ----------------- |
    | out    | UInt<1> | repeated sequence |
  }];
}

def CirctLTLGoToRepeat : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.goto_repeat";
  let description = [{
    Repeats a boolean or sequence a specified number of times, not necessarily
    consecutively. The repetitions may be interspersed with cycles where the
    property does not hold.

    | Parameter | Type | Description                              |
    | --------- | ---- | ---------------------------------------- |
    | base      | int  | minimum number of repetitions            |
    | more      | int  | maximum additional repetitions; optional |

    | Argument | Type    | Description               |
    | -------- | ------- | ------------------------- |
    | in       | UInt<1> | input boolean or sequence |

    | Result | Type    | Description           |
    | ------ | ------- | --------------------- |
    | out    | UInt<1> | go-to repeat sequence |
  }];
}

def CirctLTLNonConsecutiveRepeat : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.non_consecutive_repeat";
  let description = [{
    Like `circt.ltl.goto_repeat`, but allows additional cycles after the last
    repetition before the overall sequence completes.

    | Parameter | Type | Description                              |
    | --------- | ---- | ---------------------------------------- |
    | base      | int  | minimum number of repetitions            |
    | more      | int  | maximum additional repetitions; optional |

    | Argument | Type    | Description               |
    | -------- | ------- | ------------------------- |
    | in       | UInt<1> | input boolean or sequence |

    | Result | Type    | Description                     |
    | ------ | ------- | ------------------------------- |
    | out    | UInt<1> | non-consecutive repeat sequence |
  }];
}

def CirctLTLPast : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.past";
  let description = [{
    Observes a boolean value from a specified number of cycles in the past.
    Corresponds to the SystemVerilog `$past(a, n)` function.

    | Parameter | Type | Description                   |
    | --------- | ---- | ----------------------------- |
    | delay     | int  | number of cycles to look back |

    | Argument | Type    | Description               |
    | -------- | ------- | ------------------------- |
    | in       | UInt<1> | input boolean             |
    | clock    | Clock   | clock for delay; optional |

    | Result | Type    | Description            |
    | ------ | ------- | ---------------------- |
    | out    | UInt<1> | value from `delay` ago |
  }];
}

def CirctLTLClock : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.clock";
  let description = [{
    Associates a clock with a boolean or sequence. This is used to indicate the
    clock domain in which a property is evaluated.

    | Argument | Type    | Description               |
    | -------- | ------- | ------------------------- |
    | in       | UInt<1> | input boolean or sequence |
    | clock    | Clock   | clock for the property    |

    | Result | Type    | Description      |
    | ------ | ------- | ---------------- |
    | out    | UInt<1> | clocked sequence |
  }];
}

def CirctLTLAnd : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.and";
  let converter = "CirctLTLBinaryConverter<LTLAndIntrinsicOp>";
  let description = [{
    Computes the conjunction of two booleans, sequences, or properties.

    | Argument | Type    | Description   |
    | -------- | ------- | ------------- |
    | lhs      | UInt<1> | left operand  |
    | rhs      | UInt<1> | right operand |

    | Result | Type    | Description |
    | ------ | ------- | ----------- |
    | out    | UInt<1> | lhs AND rhs |
  }];
}

def CirctLTLOr : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.or";
  let converter = "CirctLTLBinaryConverter<LTLOrIntrinsicOp>";
  let description = [{
    Computes the disjunction of two booleans, sequences, or properties.

    | Argument | Type    | Description   |
    | -------- | ------- | ------------- |
    | lhs      | UInt<1> | left operand  |
    | rhs      | UInt<1> | right operand |

    | Result | Type    | Description |
    | ------ | ------- | ----------- |
    | out    | UInt<1> | lhs OR rhs  |
  }];
}

def CirctLTLIntersect : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.intersect";
  let converter = "CirctLTLBinaryConverter<LTLIntersectIntrinsicOp>";
  let description = [{
    Computes the intersection of two sequences. Both sequences must match at the
    same time for the result to match.

    | Argument | Type    | Description   |
    | -------- | ------- | ------------- |
    | lhs      | UInt<1> | left operand  |
    | rhs      | UInt<1> | right operand |

    | Result | Type    | Description          |
    | ------ | ------- | -------------------- |
    | out    | UInt<1> | intersection of both |
  }];
}

def CirctLTLConcat : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.concat";
  let converter = "CirctLTLBinaryConverter<LTLConcatIntrinsicOp>";
  let description = [{
    Concatenates two sequences. The second sequence starts immediately after the
    first sequence completes.

    | Argument | Type    | Description     |
    | -------- | ------- | --------------- |
    | lhs      | UInt<1> | first sequence  |
    | rhs      | UInt<1> | second sequence |

    | Result | Type    | Description            |
    | ------ | ------- | ---------------------- |
    | out    | UInt<1> | concatenated sequences |
  }];
}

def CirctLTLImplication : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.implication";
  let converter = "CirctLTLBinaryConverter<LTLImplicationIntrinsicOp>";
  let description = [{
    Computes a property implication. If the antecedent sequence matches, the
    consequent property must hold starting from the end of the antecedent.

    | Argument | Type    | Description         |
    | -------- | ------- | ------------------- |
    | lhs      | UInt<1> | antecedent sequence |
    | rhs      | UInt<1> | consequent property |

    | Result | Type    | Description          |
    | ------ | ------- | -------------------- |
    | out    | UInt<1> | implication property |
  }];
}

def CirctLTLUntil : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.until";
  let converter = "CirctLTLBinaryConverter<LTLUntilIntrinsicOp>";
  let description = [{
    The first property must hold until the second property holds.

    | Argument | Type    | Description                          |
    | -------- | ------- | ------------------------------------ |
    | lhs      | UInt<1> | property that must hold continuously |
    | rhs      | UInt<1> | property that releases lhs           |

    | Result | Type    | Description    |
    | ------ | ------- | -------------- |
    | out    | UInt<1> | until property |
  }];
}

def CirctLTLNot : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.not";
  let converter = "CirctLTLUnaryConverter<LTLNotIntrinsicOp>";
  let description = [{
    Negates a property. The result holds if the input property does not hold.

    | Argument | Type    | Description        |
    | -------- | ------- | ------------------ |
    | in       | UInt<1> | property to negate |

    | Result | Type    | Description      |
    | ------ | ------- | ---------------- |
    | out    | UInt<1> | negated property |
  }];
}

def CirctLTLEventually : FIRRTLIntrinsic {
  let mnemonic = "circt.ltl.eventually";
  let converter = "CirctLTLUnaryConverter<LTLEventuallyIntrinsicOp>";
  let description = [{
    Asserts that a property will eventually hold at some point in the future.

    | Argument | Type    | Description       |
    | -------- | ------- | ----------------- |
    | in       | UInt<1> | property to check |

    | Result | Type    | Description         |
    | ------ | ------- | ------------------- |
    | out    | UInt<1> | eventually property |
  }];
}

def CirctMux2Cell : FIRRTLIntrinsic {
  let mnemonic = "circt.mux2cell";
  let description = [{
    A 2-to-1 MUX cell intrinsic. This is lowered to a technology-specific MUX
    cell, and is primarily used to prevent optimizers from decomposing the MUX.

    | Argument | Type    | Description         |
    | -------- | ------- | ------------------- |
    | sel      | UInt<1> | select signal       |
    | high     | T       | value when sel is 1 |
    | low      | T       | value when sel is 0 |

    | Result | Type | Description    |
    | ------ | ---- | -------------- |
    | out    | T    | selected value |
  }];
}

def CirctMux4Cell : FIRRTLIntrinsic {
  let mnemonic = "circt.mux4cell";
  let description = [{
    A 4-to-1 MUX cell intrinsic. This is lowered to a technology-specific MUX
    cell, and is primarily used to prevent optimizers from decomposing the MUX.

    | Argument | Type    | Description         |
    | -------- | ------- | ------------------- |
    | sel      | UInt<2> | select signal       |
    | v3       | T       | value when sel is 3 |
    | v2       | T       | value when sel is 2 |
    | v1       | T       | value when sel is 1 |
    | v0       | T       | value when sel is 0 |

    | Result | Type | Description    |
    | ------ | ---- | -------------- |
    | out    | T    | selected value |
  }];
}

def CirctHasBeenReset : FIRRTLIntrinsic {
  let mnemonic = "circt.has_been_reset";
  let description = [{
    Checks whether a reset signal has been asserted at least once. Produces a
    `UInt<1>` that is low before the reset has ever been asserted, and high
    afterwards.

    | Argument | Type             | Description  |
    | -------- | ---------------- | ------------ |
    | clock    | Clock            | input clock  |
    | reset    | Reset/AsyncReset | reset signal |

    | Result | Type    | Description             |
    | ------ | ------- | ----------------------- |
    | out    | UInt<1> | whether reset has fired |
  }];
}

def CirctProbe : FIRRTLIntrinsic {
  let mnemonic = "circt.fpga_probe";
  let description = [{
    Marks a signal for FPGA probing. The signal is made available for
    observation via a vendor-specific debug infrastructure (e.g., Xilinx ILA).
    The probed signal is not otherwise affected.

    | Argument | Type  | Description        |
    | -------- | ----- | ------------------ |
    | clock    | Clock | probe clock domain |
    | signal   | Any   | signal to probe    |
  }];
}

def CirctVerifAssert : FIRRTLIntrinsic {
  let mnemonic = "circt.verif.assert";
  let converter = "CirctVerifConverter<VerifAssertIntrinsicOp>";
  let description = [{
    Asserts that a property holds.
    The property may be a boolean, sequence, or property.
    Booleans are represented as `UInt<1>` values.
    Sequences and properties are defined by the corresponding `circt.ltl.*`
    intrinsics and are also represented as `UInt<1>`, but are converted into
    dedicated sequence and property types later in the compiler.

    | Parameter | Type   | Description                  |
    | --------- | ------ | ---------------------------- |
    | label     | string | user-defined label; optional |

    | Argument | Type    | Description                                     |
    | -------- | ------- | ----------------------------------------------- |
    | property | UInt<1> | A property to be checked.                       |
    | enable   | UInt<1> | Enable condition. Optional.                     |
    |          |         | If 0, behaves as if the assert was not present. |
  }];
}

def CirctVerifAssume : FIRRTLIntrinsic {
  let mnemonic = "circt.verif.assume";
  let converter = "CirctVerifConverter<VerifAssumeIntrinsicOp>";
  let description = [{
    Assumes that a property holds.
    Otherwise behaves like [`circt.verif.assert`](#circtverifassert).
  }];
}

def CirctVerifCover : FIRRTLIntrinsic {
  let mnemonic = "circt.verif.cover";
  let converter = "CirctVerifConverter<VerifCoverIntrinsicOp>";
  let description = [{
    Checks that a property holds at least once, or can hold at all.
    Otherwise behaves like [`circt.verif.assert`](#circtverifassert).
  }];
}

def CirctVerifRequire : FIRRTLIntrinsic {
  let mnemonic = "circt.verif.require";
  let converter = "CirctVerifConverter<VerifRequireIntrinsicOp>";
  let description = [{
    Requires that a property holds as a pre-condition to a contract.
    Gets converted into an assert if used outside of a FIRRTL `contract`.
    Otherwise behaves like [`circt.verif.assert`](#circtverifassert).
  }];
}

def CirctVerifEnsure : FIRRTLIntrinsic {
  let mnemonic = "circt.verif.ensure";
  let converter = "CirctVerifConverter<VerifEnsureIntrinsicOp>";
  let description = [{
    Ensures that a property holds as a post-condition of a contract.
    Gets converted into an assert if used outside of a FIRRTL `contract`.
    Otherwise behaves like [`circt.verif.assert`](#circtverifassert).
  }];
}

def CirctChiselAssert : FIRRTLIntrinsic {
  let mnemonic = "circt.chisel_assert";
  let converter = "CirctAssertConverter<AssertOp>";
  let description = [{
    Generate a clocked SV assert statement, with optional formatted error
    message.

    | Parameter | Type   | Description                                                                        |
    | --------- | ------ | ---------------------------------------------------------------------------------- |
    | format    | string | FIRRTL format string. Optional.                                                    |
    | label     | string | Label for assert/assume. Optional.                                                 |
    | guards    | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards. Optional. |

    | Argument  | Type    | Description                |
    | --------- | ------- | -------------------------- |
    | clock     | Clock   | input clock                |
    | predicate | UInt<1> | predicate to assert/assume |
    | enable    | UInt<1> | enable signal              |
    | ...       | Signals | arguments to format string |

    Example output:
    ```systemverilog
    wire _GEN = ~enable | cond;
    assert__label: assert property (@(posedge clock) _GEN) else $error("message");
    ```
  }];
}

def CirctChiselIfElseFatal : FIRRTLIntrinsic {
  let mnemonic = "circt.chisel_ifelsefatal";
  let converter = "CirctAssertConverter<AssertOp, /*ifElseFatal=*/true>";
  let description = [{
    Generate a particular Verilog sequence that's similar to an assertion.

    Has legacy special behavior and should not be used by new code.

    | Parameter | Type   | Description                     |
    | --------- | ------ | ------------------------------- |
    | format    | string | FIRRTL format string. Optional. |

    This intrinsic also accepts the `label` and `guard` parameters which
    are recorded but not used in the normal emission.

    | Argument  | Type    | Description                |
    | --------- | ------- | -------------------------- |
    | clock     | Clock   | input clock                |
    | predicate | UInt<1> | predicate to check         |
    | enable    | UInt<1> | enable signal              |
    | ...       | Signals | arguments to format string |

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
  }];
}

def CirctChiselAssume : FIRRTLIntrinsic {
  let mnemonic = "circt.chisel_assume";
  let converter = "CirctAssertConverter<AssumeOp>";
  let description = [{
    Generate a clocked SV assume statement, with optional formatted error
    message.

    | Parameter | Type   | Description                                                                        |
    | --------- | ------ | ---------------------------------------------------------------------------------- |
    | format    | string | FIRRTL format string. Optional.                                                    |
    | label     | string | Label for assume statement. Optional.                                              |
    | guards    | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards. Optional. |

    | Argument  | Type    | Description                |
    | --------- | ------- | -------------------------- |
    | clock     | Clock   | input clock                |
    | predicate | UInt<1> | predicate to assume        |
    | enable    | UInt<1> | enable signal              |
    | ...       | Signals | arguments to format string |

    Example SV output:
    ```systemverilog
    assume__label: assume property (@(posedge clock) ~enable | cond) else $error("message");
    ```
  }];
}

def CirctCover : FIRRTLIntrinsic {
  let mnemonic = "circt.chisel_cover";
  let description = [{
    Generate a clocked SV cover statement.

    | Parameter | Type   | Description                                                                        |
    | --------- | ------ | ---------------------------------------------------------------------------------- |
    | label     | string | Label for cover statement. Optional.                                               |
    | guards    | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards. Optional. |

    | Argument  | Type    | Description        |
    | --------- | ------- | ------------------ |
    | clock     | Clock   | input clock        |
    | predicate | UInt<1> | predicate to cover |
    | enable    | UInt<1> | enable signal      |

    Example SV output:
    ```systemverilog
    cover__label: cover property (@(posedge clock) enable & cond);
    ```
  }];
}

def CirctUnclockedAssume : FIRRTLIntrinsic {
  let mnemonic = "circt.unclocked_assume";
  let description = [{
    Generate a SV assume statement whose predicate is used in a sensitivity
    list of the enclosing always block.

    | Parameter | Type   | Description                                                                        |
    | --------- | ------ | ---------------------------------------------------------------------------------- |
    | format    | string | FIRRTL format string. Optional.                                                    |
    | label     | string | Label for assume statement. Optional.                                              |
    | guards    | string | Semicolon-delimited list of pre-processor tokens to use as ifdef guards. Optional. |

    | Argument  | Type    | Description                |
    | --------- | ------- | -------------------------- |
    | predicate | UInt<1> | predicate to assume        |
    | enable    | UInt<1> | enable signal              |
    | ...       | Signals | arguments to format string |

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
  }];
}

def CirctDPICall : FIRRTLIntrinsic {
  let mnemonic = "circt.dpi_call";
  let description = [{
    Call a DPI function. `clock` is optional and if `clock` is not provided,
    the callee is invoked when input values are changed.
    If provided, the dpi function is called at clock's posedge. The result
    values behave like registers and the DPI function is used as a state
    transfer function of them.

    `enable` operand is used to conditionally call the DPI since DPI call could
    be quite more expensive than native constructs. When `enable` is low,
    results of unclocked calls are undefined and evaluated into `X`. Users are
    expected to gate result values by another `enable` to model a default value
    of results.

    For clocked calls, a low enable means that its register state transfer
    function is not called. Hence their values will not be modify in that clock.

    | Parameter    | Type   | Description                                        |
    | ------------ | ------ | -------------------------------------------------- |
    | isClocked    | int    | Set 1 if the dpi call is clocked.                  |
    | functionName | string | Specify the function name.                         |
    | inputNames   | string | Semicolon-delimited list of input names. Optional. |
    | outputName   | string | Output name. Optional.                             |

    | Argument | Type    | Description                    |
    | -------- | ------- | ------------------------------ |
    | clock    | Clock   | Clock operand; optional        |
    | enable   | UInt<1> | Enable signal                  |
    | ...      | Signals | Arguments to DPI function call |

    | Result | Type   | Description                      |
    | ------ | ------ | -------------------------------- |
    | result | Signal | Result of the DPI call; optional |

    #### DPI Intrinsic ABI

    Function Declaration:
    * Imported DPI function must be a void function that has input arguments
      which correspond to operand types, and an output argument which correspond
      to a result type.
    * Output argument must be a last argument.

    Types:
    * Operand and result types must be passive.
    * A vector is lowered to an unpacked open array type, e.g.
      `a: Vec<4, UInt<8>>` to `byte a []`.
    * A bundle is lowered to a packed struct.
    * Integer types are lowered into into 2-state types.
    * Small integer types (< 64 bit) must be compatible to C-types and
      arguments are passed by values. Users are required to use specific integer
      types for small integers shown in the table below. Large integers are
      lowered to `bit` and passed by a reference.

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
  }];
}

def View : FIRRTLIntrinsic {
  let mnemonic = "circt.view";
  let description = [{
    This will become a SystemVerilog Interface that is driven by its arguments.
    This is _not_ a true SystemVerilog Interface, it is only lowered to one.

    | Parameter | Type   | Description                              |
    | --------- | ------ | ---------------------------------------- |
    | name      | string | Instance name of the view.               |
    | info      | string | JSON encoding the view structure.        |
    | yaml      | string | Path to emit YAML description. Optional. |

    | Argument | Type   | Description                     |
    | -------- | ------ | ------------------------------- |
    | ...      | Ground | Leaf ground values for the view |

    The structure of the view is encoded using JSON, with the top-level object
    required to be an `AugmentedBundleType`.

    The intrinsic operands correspond to the `AugmentedGroundType` leaves,
    and must be of ground type.

    This encoding is a trimmed version of what's used for the old GrandCentral
    View annotation.

    Example usage (the `info` parameter contains a JSON-encoded
    `AugmentedBundleType`; shown abbreviated here):
    ```firrtl
    circuit ViewExample:
      public module ViewExample:
        input in : { x : UInt<2>, y : { z : UInt<3>[2] } }
        intrinsic(circt_view<name="view", info="...">, in.x, in.y.z[0], in.y.z[1])
    ```

    Example Output:
    ```systemverilog
    module ViewExample(
      input [1:0] in_x,
      input [2:0] in_y_z_0,
                  in_y_z_1
    );

      ViewName view();
      assign view.x = in_x;
      assign view.y.z[0] = in_y_z_0;
      assign view.y.z[1] = in_y_z_1;
    endmodule

    // VCS coverage exclude_file
    interface ViewName;
      // X marks the spot
      logic [1:0] x;
      // y bundle
      YView y();
    endinterface

    // VCS coverage exclude_file
    interface YView;
      logic [2:0] z[0:1];
    endinterface
    ```

    #### AugmentedGroundType

    | Property | Type   | Description                                          |
    | -------- | ------ | ---------------------------------------------------- |
    | class    | string | `sifive.enterprise.grandcentral.AugmentedGroundType` |

    Creates a SystemVerilog logic type.

    Each ground type corresponds to an operand to the view intrinsic.

    Example:
    ```json
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType"
    }
    ```

    #### AugmentedVectorType

    | Property | Type   | Description                                          |
    | -------- | ------ | ---------------------------------------------------- |
    | class    | string | `sifive.enterprise.grandcentral.AugmentedVectorType` |
    | elements | array  | List of augmented types.                             |

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
    | ----------- | ------ | ---------------------------------- |
    | name        | string | Name of the field                  |
    | description | string | A textual description of this type |
    | tpe         | string | A nested augmented type            |

    A field in an augmented bundle type. This can provide a small description
    of what the field in the bundle is.

    #### AugmentedBundleType

    | Property | Type   | Description                                              |
    | -------- | ------ | -------------------------------------------------------- |
    | class    | string | sifive.enterprise.grandcentral.AugmentedBundleType       |
    | defName  | string | The name of the SystemVerilog interface. May be renamed. |
    | elements | array  | List of AugmentedFields                                  |

    Creates a SystemVerilog interface for each bundle type.
  }];
}
