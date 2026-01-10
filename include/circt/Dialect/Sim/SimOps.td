//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMOPS_TD
#define CIRCT_DIALECT_SIM_SIMOPS_TD

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "circt/Dialect/Sim/SimDialect.td"
include "circt/Dialect/Sim/SimTypes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

class SimOp<string mnemonic, list<Trait> traits = []> :
    Op<SimDialect, mnemonic, traits>;

def PlusArgsTestOp : SimOp<"plusargs.test", [Pure]> {
  let summary = "SystemVerilog `$test$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found);
  let assemblyFormat = "$formatString attr-dict";
}

def PlusArgsValueOp : SimOp<"plusargs.value", [Pure]> {
  let summary = "SystemVerilog `$value$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found, AnyType:$result);
  let assemblyFormat = "$formatString attr-dict `:` type($result)";
}

def DPIFuncOp : SimOp<"func.dpi",
     [IsolatedFromAbove, Symbol, OpAsmOpInterface,
      FunctionOpInterface]> {
  let summary = "A System Verilog function";
  let description = [{
    `sim.func.dpi` models an external function in a core dialect.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<ModuleType>:$module_type,
    OptionalAttr<DictArrayAttr>:$per_argument_attrs,
    OptionalAttr<LocationArrayAttr>:$argument_locs,
    OptionalAttr<StrAttr>:$verilogName
  );
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    mlir::FunctionType getFunctionType() {
      return getModuleType().getFuncType();
    }

    void setFunctionTypeAttr(mlir::TypeAttr mlirType) {
      setModuleType(cast<hw::ModuleType>(mlirType.getValue()));
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    ::mlir::Region *getCallableRegion() { return nullptr; }

    /// Stub implementations for ArgumentAttributesMethods. If desired,
    /// implement these by defining arg_attrs and res_attrs as arguments to the
    /// operation as OptionalAttr<DictArrayAttr>.
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];
}

def DPICallOp : SimOp<"func.dpi.call",
     [CallOpInterface, AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "A call option for DPI function with an optional clock and enable";
  let description = [{
    `sim.func.dpi.call` represents SystemVerilog DPI function call. There are two
    optional operands `clock` and `enable`.

    If `clock` is not provided, the callee is invoked when input values are changed.
    If provided, the DPI function is called at clock's posedge. The result values behave
    like registers and the DPI function is used as a state transfer function of them.

    `enable` operand is used to conditionally call the DPI since DPI call could be quite
    more expensive than native constructs. When `enable` is low, results of unclocked
    calls are undefined and in SV results they are lowered into `X`. Users are expected
    to gate result values by another `enable` to model a default value of results.

    For clocked calls, a low enable means that its register state transfer function is
    not called. Hence their values will not be modify in that clock.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                      Optional<ClockType>:$clock,
                      Optional<I1>:$enable,
                      Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)?
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getInputs();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(), llvm::cast<mlir::SymbolRefAttr>(callee));
    }

    /// Stub implementations for ArgumentAttributesMethods. If desired,
    /// implement these by defining arg_attrs and res_attrs as arguments to the
    /// operation as OptionalAttr<DictArrayAttr>.
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];

}

//===----------------------------------------------------------------------===//
// String Formatting
//===----------------------------------------------------------------------===//

def FormatLiteralOp : SimOp<"fmt.literal", [Pure, ConstantLike]> {
  let summary = "Literal string fragment";
  let description = [{
    Creates a constant, raw ASCII string literal for formatted printing.
    The given string attribute will be outputted as is,
    including non-printable characters. The literal may be empty or contain
    null characters ('\0') which must not be interpreted as string
    terminators by backends.
  }];

  let assemblyFormat = "$literal attr-dict";

  let arguments = (ins StrAttr:$literal);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;
}

def FormatHexOp : SimOp<"fmt.hex", [Pure]> {
  let summary = "Hexadecimal format specifier";
  let description = [{
    Format the given integer value as lower-case hexadecimal string.

    The printed value will be left-padded with '0' up to the
    length required to print the maximum value of the argument's
    type. Zero width values will produce the empty string.
    No further prefix will be added.
  }];


  let arguments = (ins AnyInteger:$value, 
    BoolAttr:$isHexUppercase,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    DefaultValuedAttr<I8Attr,"48">:$paddingChar,
    OptionalAttr<I32Attr>:$specifierWidth);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value `,`
    `isUpper`  $isHexUppercase
    (`isLeftAligned` $isLeftAligned^)?
    (`paddingChar` $paddingChar^)?
    (`specifierWidth` $specifierWidth^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatOctOp : SimOp<"fmt.oct", [Pure]> {
  let summary = "Octal format specifier";
  let description = [{
    Format the given integer value as lower-case octal (base eight) string.

    The printed value will be left-padded with '0' up to the
    length required to print the maximum value of the argument's
    type. Zero width values will produce the empty string.
    No further prefix will be added.
  }];


  let arguments = (ins AnyInteger:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    DefaultValuedAttr<I8Attr,"48">:$paddingChar,
    OptionalAttr<I32Attr>:$specifierWidth);

  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`paddingChar` $paddingChar^)?
    (`specifierWidth` $specifierWidth^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatBinOp : SimOp<"fmt.bin", [Pure]> {
  let summary = "Binary format specifier";
  let description = [{
    Format the given integer value as binary (base two) string.

    The printed value will be left-padded with '0' up to the number
    of bits of the argument's type. Zero width values will produce
    the empty string. No further prefix will be added.
  }];

  let arguments = (ins AnyInteger:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    DefaultValuedAttr<I8Attr,"48">:$paddingChar,
    OptionalAttr<I32Attr>:$specifierWidth);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`paddingChar` $paddingChar^)?
    (`specifierWidth` $specifierWidth^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatScientificOp : SimOp<"fmt.exp", [Pure]> {
  let summary = "Real format specifier";
  let description = [{
    Format the given real value in scientific (exponential) notation
  }];

  let arguments = (ins AnyFloat:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    OptionalAttr<I32Attr>:$fieldWidth,
    DefaultValuedAttr<I32Attr,"6">:$fracDigits
  );
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatFloatOp : SimOp<"fmt.flt", [Pure]> {
  let summary = "Real format specifier";
  let description = [{
    Format the given real value in floating point (decimal) notation  
  }];

  let arguments = (ins AnyFloat:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    OptionalAttr<I32Attr>:$fieldWidth,
    DefaultValuedAttr<I32Attr,"6">:$fracDigits
  );
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatGeneralOp : SimOp<"fmt.gen", [Pure]> {
  let summary = "Real format specifier";
  let description = [{
    Format floating-point numbers using either decimal or scientific notation
    based on the exponent and precision parameter fracDigits (default: 6)

    For exp ≥ 0: 
    Uses decimal notation if exp + 1 ≤ max(1, fracDigits), otherwise switches
    to scientific notation with max(0, fracDigits - 1) fractional digits
    
    For exp < 0:
    If exp ≥ -4: Uses decimal notation with max(b - 1, 0) + |exp| fractional digits
    If exp < -4: Uses scientific notation with max(b - 1, 0) fractional digits
  }];

  let arguments = (ins AnyFloat:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    OptionalAttr<I32Attr>:$fieldWidth,
    DefaultValuedAttr<I32Attr,"6">:$fracDigits
  );
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatDecOp : SimOp<"fmt.dec", [Pure]> {
  let summary = "Decimal format specifier";
  let description = [{
    Format the given integer value as signed or unsigned decimal string.

    Leading zeros are omitted. Non-negative or unsigned values will
    _not_ be prefixed with a '+'.

    For unsigned formatting, the printed value will
    be left-padded with spaces up to _at least_ the length required to print
    the maximum unsigned value of the argument's type.
    For signed formatting, the printed value will be
    left-padded with spaces up to _at least_ the length required
    to print the minimum signed value of the argument's type
    including the '-' character.
    E.g., a zero value of type `i1` requires no padding for unsigned
    formatting and one leading space for signed formatting.
    Format specifiers of same argument type and signedness must be
    padded to the same width. Zero width values will produce
    a single '0'.

    Backends are recommended to not exceed the required amount of padding.
  }];

  let arguments = (ins AnyInteger:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    DefaultValuedAttr<I8Attr,"32">:$paddingChar,
    OptionalAttr<I32Attr>:$specifierWidth,
    UnitAttr:$isSigned);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`paddingChar` $paddingChar^)?
    (`specifierWidth` $specifierWidth^)?
    (`signed` $isSigned^)?
    attr-dict `:` qualified(type($value))
  }];

  let extraClassDeclaration = [{
      static inline unsigned getDecimalWidth(unsigned bits, bool isSigned) {
        if (bits == 0)
          return 1;
        if (bits == 1)
          return isSigned ? 2 : 1;

        if (isSigned)
          bits--;

        // Should be precise up until bits = 13301
        // log(2) / log(10) + epsilon
        const double baseConversionFactor = 0.30103;
        unsigned digits = std::ceil(bits * baseConversionFactor);

        return isSigned ? digits + 1 : digits;
      }
  }];
}

def FormatCharOp : SimOp<"fmt.char", [Pure]> {
  let summary = "Character format specifier";
  let description = [{
    Format the given integer value as a single character.

    For integer values up to 127, ASCII compatible encoding is assumed.
    For larger values, the encoding is unspecified.

    If the argument's type width is less than eight bits, the value is
    zero extended.
    If the width is greater than eight bits, the resulting formatted string
    is undefined.
  }];

  let arguments = (ins AnyInteger:$value);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = "$value attr-dict `:` qualified(type($value))";
}

def FormatStringConcatOp : SimOp<"fmt.concat", [Pure]> {
  let summary = "Concatenate format strings";
  let description = [{
    Concatenates an arbitrary number of format strings from
    left to right. If the argument list is empty, the empty string
    literal is produced.

    Concatenations must not be recursive. I.e., a concatenated string should
    not contain itself directly or indirectly.
  }];

  let arguments = (ins Variadic<FormatStringType>:$inputs);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;
  let hasCanonicalizeMethod = true;
  let hasVerifier = true;

  let assemblyFormat = "` ` `(` $inputs `)` attr-dict";

  let extraClassDeclaration = [{
    /// Returns true iff all of the input strings are primitive
    /// (i.e. non-concatenated) fragments or block arguments.
    bool isFlat() {
      return llvm::none_of(getInputs(), [](Value operand) {
        return !!operand.getDefiningOp<circt::sim::FormatStringConcatOp>();
      });
    };

    /// Attempts to flatten this operation's input strings as much as possible.
    ///
    /// The flattened values are pushed into the passed vector.
    /// If the concatenation is acyclic, the function will return 'success'
    /// and all the flattened values are guaranteed to _not_ be the result of
    /// a format string concatenation.
    /// If a cycle is encountered, the function will return 'failure'.
    /// On encountering a cycle, the result of the concat operation
    /// forming the cycle is pushed into the list of flattened values
    /// and flattening continues without recursing into the cycle.
    LogicalResult getFlattenedInputs(
      llvm::SmallVectorImpl<Value> &flatOperands);
  }];
}

def PrintFormattedOp : SimOp<"print"> {
  let summary = "Print a formatted string on a given clock and condition";
  let description = [{
    Evaluate a format string and print it to the simulation console on the
    rising edge of the given clock, if, and only if, the condition argument
    is 'true'.

    Multiple print operations in the same module and on the same clock edge
    are performed according to their order of occurence in the IR. The order
    of printing for operations in different modules, instances or on different
    clocks is undefined.
  }];
  let arguments = (ins
    FormatStringType:$input,
    ClockType:$clock,
    I1:$condition
  );
  let hasCanonicalizeMethod = true;
  let assemblyFormat = "$input `on` $clock `if` $condition attr-dict";
}

def PrintFormattedProcOp : SimOp<"proc.print"> {
  let summary = "Print a formatted string within a procedural region";
  let description = [{
    Evaluate a format string and print it to the simulation console.

    This operation must be within a procedural region.
  }];
  let arguments = (ins FormatStringType:$input);
  let hasVerifier = true;
  let hasCanonicalizeMethod = true;
  let assemblyFormat = "$input attr-dict";
}

//===----------------------------------------------------------------------===//
// Dynamic Strings
//===----------------------------------------------------------------------===//

def StringConstantOp : SimOp<"string.literal", [Pure, ConstantLike]> {
  let summary = "Literal string fragment";
  let description = [{
    Creates a constant, raw ASCII string literal
  }];

  let arguments = (ins StrAttr:$literal);
  let results = (outs DynamicStringType:$result);

  let hasFolder = true;

  let assemblyFormat = "$literal attr-dict";
}

def StringLengthOp : SimOp<"string.length", [Pure]> {
  let summary = "Length of a string";
  let description = [{
    Returns the length of a Dynamic String instance's value
  }];

  let arguments = (ins DynamicStringType:$input);
  let results = (outs I64:$result);

  let hasFolder = true;

  let assemblyFormat = "$input attr-dict";
}

def StringConcatOp : SimOp<"string.concat", [Pure]> {
  let summary = "Concatenates a variadic list of dynamic strings together";
  let description = [{
    Returns the length of a Dynamic String instance
  }];

  let assemblyFormat = "` ` `(` $inputs `)` attr-dict";

  let arguments = (ins Variadic<DynamicStringType>:$inputs);
  let results = (outs DynamicStringType:$result);

  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// Simulation Control
//===----------------------------------------------------------------------===//

def ClockedTerminateOp : SimOp<"clocked_terminate"> {
  let summary = "Terminate the simulation";
  let description = [{
    Implements the semantics of `sim.terminate` if the given condition is true
    on the rising edge of the clock operand.
  }];
  let arguments = (ins
    ClockType:$clock,
    I1:$condition,
    BoolAttr:$success,
    BoolAttr:$verbose
  );
  let assemblyFormat = [{
    $clock `,` $condition `,`
    custom<KeywordBool>($success, "\"success\"", "\"failure\"") `,`
    custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
  }];
}

def ClockedPauseOp : SimOp<"clocked_pause"> {
  let summary = "Pause the simulation";
  let description = [{
    Implements the semantics of `sim.pause` if the given condition is true on
    the rising edge of the clock operand.
  }];
  let arguments = (ins
    ClockType:$clock,
    I1:$condition,
    BoolAttr:$verbose
  );
  let assemblyFormat = [{
    $clock `,` $condition `,`
    custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
  }];
}

def TerminateOp : SimOp<"terminate"> {
  let summary = "Terminate the simulation";
  let description = [{
    Terminate the simulation with a success or failure exit code. Depending on
    the `verbose` operand, simulators may print additional information about the
    current simulation time and hierarchical location of the op.

    This op correpsonds to the following SystemVerilog constructs:

    | Operation                        | SystemVerilog             |
    |----------------------------------|---------------------------|
    | `sim.terminate success, quiet`   | `$finish(0)`              |
    | `sim.terminate success, verbose` | `$finish` or `$finish(1)` |
    | `sim.terminate failure, quiet`   | `$fatal(0)`               |
    | `sim.terminate failure, verbose` | `$fatal` or `$fatal(1)`   |

    Note that this op does not match the behavior of the `$exit` system task in
    SystemVerilog, which blocks execution of the calling process until all
    `program` instances have terminated, and then calls `$finish`.
  }];
  let arguments = (ins
    BoolAttr:$success,
    BoolAttr:$verbose
  );
  let assemblyFormat = [{
    custom<KeywordBool>($success, "\"success\"", "\"failure\"") `,`
    custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
  }];
}

def PauseOp : SimOp<"pause"> {
  let summary = "Pause the simulation";
  let description = [{
    Interrupt the simulation and give control back to the user in case of an
    interactive simulation. Non-interactive simulations may choose to terminate
    instead. Depending on the `verbose` operand, simulators may print additional
    information about the current simulation time and hierarchical location of
    the op.

    This op corresponds to the following SystemVerilog constructs:

    | Operation           | SystemVerilog          |
    |---------------------|------------------------|
    | `sim.pause quiet`   | `$stop(0)`             |
    | `sim.pause verbose` | `$stop` or `$stop(1)`  |
  }];
  let arguments = (ins BoolAttr:$verbose);
  let assemblyFormat = [{
    custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
  }];
}

#endif // CIRCT_DIALECT_SIM_SIMOPS_TD
