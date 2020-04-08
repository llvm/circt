//===- FIRRTL.td - FIRRTL dialect definition ---------------*- tablegen -*-===//
//
// This is the operation definition file for FIRRTL dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef FIRRTL_OPS
#define FIRRTL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"

def FIRRTLDialect : Dialect {
  let name = "firrtl";

  let summary = "Types and operations for firrtl dialect";
  let description = [{
    This dialect defines the `firrtl` dialect, which is used to lower from
    Chisel code to Verilog.  For more information, see the
    [FIRRTL GitHub page](https://github.com/freechipsproject/firrtl).
  }];

  let cppNamespace = "firrtl";
}

//===----------------------------------------------------------------------===//
// FIRRTL Types Definitions
//===----------------------------------------------------------------------===//

def ClockType : Type<CPred<"$_self.isa<ClockType>()">, "clock">,
                BuildableType<"ClockType::get($_builder.getContext())">;

def UInt1Type : Type<CPred<"$_self.isa<UIntType>() && "
                           "$_self.cast<UIntType>().getWidth() == 1">,
                           "UInt<1>">,
                BuildableType<"UIntType::get($_builder.getContext(), 1)">;

//===----------------------------------------------------------------------===//
// FIRRTL op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class FIRRTLOp<string mnemonic, list<OpTrait> traits = []> :
    Op<FIRRTLDialect, mnemonic, traits>;

def CircuitOp : FIRRTLOp<"circuit",
      [IsolatedFromAbove, SymbolTable, 
       SingleBlockImplicitTerminator<"DoneOp">]> {
  let summary = "FIRRTL Circuit";
  let description = [{
    The "firrtl.circuit" operation represents an overall Verilog circuit,
    containing a list of modules.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringAttr name">
  ];

  let extraClassDeclaration = [{
    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder(&bodyBlock, std::prev(bodyBlock.end()));
    }
  }];
  
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def FModuleOp : FIRRTLOp<"module",
      [IsolatedFromAbove, FunctionLike, Symbol,
       SingleBlockImplicitTerminator<"DoneOp">]> {
  let summary = "FIRRTL Module";
  let description = [{
    The "firrtl.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringAttr name, "
              "ArrayRef<std::pair<StringAttr, FIRRTLType>> ports">
  ];

  let extraClassDeclaration = [{
    // Decode information about the input and output ports on this module.
    void getPortInfo(SmallVectorImpl<ModulePortInfo> &results) {
      getModulePortInfo(*this, results);
    }

    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder(&bodyBlock, std::prev(bodyBlock.end()));
    }
  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FModuleOp>;

    /// Returns the number of arguments, implementing OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }
    /// Returns the number of results, implementing OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    ///  safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  public:
  }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def FExtModuleOp : FIRRTLOp<"extmodule",
      [IsolatedFromAbove, FunctionLike, Symbol]> {
  let summary = "FIRRTL extmodule";
  let description = [{
    The "firrtl.extmodule" operation represents an external reference to a
    Verilog module, including a given name, a list of ports, and a body that
    represents the connections within the module.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringAttr name, "
              "ArrayRef<std::pair<StringAttr, FIRRTLType>> ports">
  ];

  let extraClassDeclaration = [{
     // Decode information about the input and output ports on this module.
     void getPortInfo(SmallVectorImpl<ModulePortInfo> &results) {
       getModulePortInfo(*this, results);
     }

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FExtModuleOp>;

    /// Returns the number of arguments, implementing OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }
    /// Returns the number of results, implementing OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    ///  safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  public:
  }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def DoneOp : FIRRTLOp<"done", [Terminator]> {
  let summary = "FIRRTL termination operation";
  let description = [{
    "firrtl.done" marks the end of a region in the FIRRTL dialect.
  }];

  let arguments = (ins);
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

def InstanceOp : FIRRTLOp<"instance", []> {
  let summary = "Instantiate an instance of a module";
  let description = [{
    This represents an instance of a module.  The result is always a bundle
    that is a flattened form of the referenced modules inputs and outputs.
  }];

  let arguments = (ins FlatSymbolRefAttr:$moduleName,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
     $moduleName attr-dict `:` type($result)
  }];
}

def NodeOp : FIRRTLOp<"node", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "No-op to name a value";
  let description = [{
    A node is simply a named intermediate value in a circuit. The node must
    be initialized to a value with a passive type and cannot be connected to.
    Nodes are often used to split a complicated compound expression into named
    subexpressions.

    ```
      %result = firrt.node %input : t1
    ```
    }];

  let arguments = (ins AnyType:$input, OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
     $input attr-dict `:` type($input)
  }];
}

def RegOp : FIRRTLOp<"reg", []> {
  let summary = "Define a new register";
  let description = [{
    Declare a new register:
    ```
      %name = firrt.reg %clockVal : t1
    ```
    }];

  let arguments = (ins AnyType:$clockVal, OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
     operands attr-dict `:` functional-type(operands, $result)
  }];
}

def RegInitOp : FIRRTLOp<"reginit", []> {
  let summary = "Define a new register with an initializer";
  let description = [{
    Declare a new register:
    ```
      %name = firrt.reg %clockVal, %resetSignal, %resetValue : t1
    ```
    }];

  let arguments = (ins AnyType:$clockVal, AnyType:$resetSignal, 
                       AnyType:$resetValue, OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
     operands attr-dict `:` functional-type(operands, $result)
  }];
}

def WireOp : FIRRTLOp<"wire", [NoSideEffect]> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = firrt.wire : t1
    ```
    }];

  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
     attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

def ConstantOp : FIRRTLOp<"constant", [NoSideEffect]> {
  let summary = "Produce a constant value";
  let description = [{
    The constant operation produces a constant value of SInt or UInt type.
    ```
      %result = firrt.constant 42 : t1
    ```
    }];

  let arguments = (ins I64Attr:$value, OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  // TODO: Could drop the result type, inferring it from the source.
  let assemblyFormat = [{
     $value attr-dict `:` type($result)
  }];
}

def SubfieldOp : FIRRTLOp<"subfield", [NoSideEffect]> {
  let summary = "Extract a subfield of another value";
  let description = [{
    The subfield expression refers to a subelement of an expression with a
    bundle type.
    ```
      %result = firrt.subfield %input "fieldname" : t1, t2
    ```
    }];

  let arguments = (ins AnyType:$input, StrAttr:$fieldname,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  // TODO: Could drop the result type, inferring it from the source.
  let assemblyFormat = [{
     $input `(` $fieldname `)` attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = [{
    /// Compute the result of a Subfield operation on a value of the specified
    /// type and extracting the specified field name.  If the request is
    /// invalid, then a null type is returned.
    static FIRRTLType getResultType(FIRRTLType inType, StringRef fieldName);
  }];
}

def SubindexOp : FIRRTLOp<"subindex", [NoSideEffect]> {
  let summary = "Extract an element of a vector value";
  let description = [{
    The subindex expression statically refers, by index, to a subelement
    of an expression with a vector type. The index must be a non-negative
    integer and cannot be equal to or exceed the length of the vector it 
    indexes.
    ```
      %result = firrt.subindex %input[index] : t1, t2
    ```
    }];

  let arguments = (ins AnyType:$input, I32Attr:$index,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  // TODO: Could drop the result type, inferring it from the source.
  let assemblyFormat = [{
     $input `[` $index `]` attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = [{
    /// Compute the result of a Subindex operation on a value of the specified
    /// type.  If the request is invalid, then a null type is returned.
    static FIRRTLType getResultType(FIRRTLType inType, unsigned fieldIdx);
  }];
}

def SubaccessOp : FIRRTLOp<"subaccess", [NoSideEffect]> {
  let summary = "Extract a dynamic element of a vector value";
  let description = [{
    The subaccess expression dynamically refers to a subelement of a 
    vector-typed expression using a calculated index. The index must be an
    expression with an unsigned integer type.
    ```
      %result = firrt.subaccess %input[%idx] : t1, t2
    ```
    }];

  let arguments = (ins AnyType:$input, AnyType:$index,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  // TODO: Could drop the result type, inferring it from the source.
  let assemblyFormat = [{
     $input `[` $index `]` attr-dict `:` functional-type(operands, $result)
  }];

  let extraClassDeclaration = [{
    /// Compute the result of a Subaccess operation on a value of the specified
    /// type.  If the request is invalid, then a null type is returned.
    static FIRRTLType getResultType(FIRRTLType baseType, FIRRTLType indexType);
  }];
}


//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

def ConnectOp : FIRRTLOp<"connect", []> {
  let summary = "Connect two signals";
  let description = [{
    Connect Operation:
    ```
      firrt.connect %lhs, %rhs : t1, t2
    ```
    }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs);
  
  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:` type($lhs) `,` type($rhs)
  }];
}

def PartialConnectOp : FIRRTLOp<"partialconnect", []> {
  let summary = "Connect two signals";
  let description = [{
    Connect two values with fewer constraints:
    ```
      firrt.partialconnect %lhs, %rhs : t1, t2
    ```
    }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs);
  
  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:` type($lhs) `,` type($rhs)
  }];
}

def InvalidOp : FIRRTLOp<"invalid", []> {
  let summary = "Invalid statement";
  let description = [{
    Invalidate Operation:
    ```
       %result = firrt.invalid : i8
    ```

    The invalid operation produces a value of unspecified value.  Drivers may
    be attached to it with `firrtl.connect`.
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs);
  
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand)
  }];
}

def PrintFOp : FIRRTLOp<"printf", []> {
  let summary = "Formatted Print Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$cond, StrAttr:$formatString, 
                       Variadic<AnyType>:$operands);
  let results = (outs);
  
  let assemblyFormat = [{
    $clock `,` $cond `,` $formatString attr-dict ($operands^ `:` type($operands))?
  }];
}

def SkipOp : FIRRTLOp<"skip", [NoSideEffect]> {
  let summary = "Skip statement";
  let description = [{
    Skip Statement:
    ```
       %firrt.skip
    ```

    This is a no-op statement.
  }];

  let arguments = (ins);
  let results = (outs);
  
  let assemblyFormat = [{
    attr-dict
  }];
}

def StopOp : FIRRTLOp<"stop", []> {
  let summary = "Stop Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$cond, I32Attr:$exitCode);
  let results = (outs);
  
  let assemblyFormat = [{
    $clock `,` $cond `,` $exitCode attr-dict
  }];
}

def WhenOp : FIRRTLOp<"when", [SingleBlockImplicitTerminator<"DoneOp">]> {
  let summary = "When Statement";
  let description = [{
    The "firrtl.when" operation represents a conditional.  Connections within
    a conditional statement that connect to previously declared components hold
    only when the given condition is high. The condition must have a 1-bit
    unsigned integer type.
  }];
  let arguments = (ins AnyType:$condition);
  let results = (outs);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value condition, bool withElseRegion">
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      Block &body = thenRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }

    bool hasElseRegion() {
      return !elseRegion().empty();
    }

    void createElseRegion();

    OpBuilder getElseBodyBuilder() {
      assert(hasElseRegion() && "Unexpected empty 'else' region.");
      Block &body = elseRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
  }];
  
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Primitive Operations
//===----------------------------------------------------------------------===//

class PrimOp<string mnemonic, list<OpTrait> traits = []> :
    FIRRTLOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:`
       `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

// Base class for binary primitive operators.
class BinaryPrimOp<string mnemonic, string resultTypeFunction, 
                   list<OpTrait> traits = []> :
    PrimOp<mnemonic, traits> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:`
       `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];

  let extraClassDeclaration = !cast<code>(!strconcat(!cast<string>([{
    /// Return the result for inputs with the specified type, returning a null
    /// type if the input types are invalid.
    static FIRRTLType getResultType(FIRRTLType lhs, FIRRTLType rhs) {
      return }]), resultTypeFunction, !cast<string>([{(lhs, rhs);
    }
    static FIRRTLType getResultType(ArrayRef<FIRRTLType> inputs,
                                    ArrayRef<int32_t> integers) {
      if (inputs.size() != 2 || !integers.empty())
        return {};
      return getResultType(inputs[0], inputs[1]);
    }
  }])));
}

def AddPrimOp : BinaryPrimOp<"add", "getAddSubResult", [Commutative]>;
def SubPrimOp : BinaryPrimOp<"sub", "getAddSubResult", []>;
def MulPrimOp : BinaryPrimOp<"mul", "getMulResult", [Commutative]>;
def DivPrimOp : BinaryPrimOp<"div", "getDivResult", []>;
def RemPrimOp : BinaryPrimOp<"rem", "getRemResult", []>;

def AndPrimOp : BinaryPrimOp<"and", "getBitwiseBinaryResult", [Commutative]>;
def OrPrimOp  : BinaryPrimOp<"or",  "getBitwiseBinaryResult", [Commutative]>;
def XorPrimOp : BinaryPrimOp<"xor", "getBitwiseBinaryResult", [Commutative]>;

// Comparison Operations
def LEQPrimOp : BinaryPrimOp<"leq", "getCompareResult", []>;
def LTPrimOp  : BinaryPrimOp<"lt",  "getCompareResult", []>;
def GEQPrimOp : BinaryPrimOp<"geq", "getCompareResult", []>;
def GTPrimOp  : BinaryPrimOp<"gt",  "getCompareResult", []>;
def EQPrimOp  : BinaryPrimOp<"eq",  "getCompareResult", [Commutative]>;
def NEQPrimOp : BinaryPrimOp<"neq", "getCompareResult", [Commutative]>;

def CatPrimOp  : BinaryPrimOp<"cat", "getCatResult", []>;
def DShlPrimOp : BinaryPrimOp<"dshl", "getDShlResult", []>;
def DShrPrimOp : BinaryPrimOp<"dshr", "getDShrResult", []>;

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Base class for binary primitive operators.
class UnaryPrimOp<string mnemonic, string resultTypeFunction, 
                  list<OpTrait> traits = []> :
    PrimOp<mnemonic, traits> {
  let arguments = (ins AnyType:$input, OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = !cast<code>(!strconcat(!cast<string>([{
    /// Return the result for inputs with the specified type, returning a null
    /// type if the input types are invalid.
    static FIRRTLType getResultType(FIRRTLType input) {
      return }]), resultTypeFunction, !cast<string>([{(input);
    }
    static FIRRTLType getResultType(ArrayRef<FIRRTLType> inputs,
                                    ArrayRef<int32_t> integers) {
      if (inputs.size() != 1 || !integers.empty())
        return {};
      return getResultType(inputs[0]);
    }
  }])));
}

def AsSIntPrimOp  : UnaryPrimOp<"asSInt", "getAsSIntResult", []>;
def AsUIntPrimOp  : UnaryPrimOp<"asUInt", "getAsUIntResult", []>;
def AsClockPrimOp : UnaryPrimOp<"asClock", "getAsClockResult", []>;
def CvtPrimOp     : UnaryPrimOp<"cvt", "getCvtResult", []>;
def NegPrimOp     : UnaryPrimOp<"neg", "getNegResult", []>;
def NotPrimOp     : UnaryPrimOp<"not", "getNotResult", []>;

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

def BitsPrimOp : PrimOp<"bits", []> {
  let arguments = (ins AnyType:$input, I32Attr:$hi, I32Attr:$lo,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $input $hi `to` $lo attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = [{
    /// Return the result for inputs with the specified type, returning a null
    /// type if the input types are invalid.
    static FIRRTLType getResultType(FIRRTLType input, int32_t high,
                                    int32_t low);
    static FIRRTLType getResultType(ArrayRef<FIRRTLType> inputs,
                                    ArrayRef<int32_t> integers) {
      if (inputs.size() != 1 || integers.size() != 2)
        return {};
      return getResultType(inputs[0], integers[0], integers[1]);
    }
  }];
}

def MuxPrimOp : PrimOp<"mux", []> {
  let arguments = (ins AnyType:$sel, AnyType:$high, AnyType:$low,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, $result)
  }];

  let extraClassDeclaration = [{
    /// Return the result for inputs with the specified type, returning a null
    /// type if the input types are invalid.
    static FIRRTLType getResultType(FIRRTLType sel, FIRRTLType high,
                                    FIRRTLType low);
    static FIRRTLType getResultType(ArrayRef<FIRRTLType> inputs,
                                    ArrayRef<int32_t> integers) {
      if (inputs.size() != 3 || integers.size() != 0)
        return {};
      return getResultType(inputs[0], inputs[1], inputs[2]);
    }
  }];
}

class ShiftPrimOp<string mnemonic> : PrimOp<mnemonic, []> {
  let arguments = (ins AnyType:$input, I32Attr:$amount,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $input `,` $amount attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = [{
    /// Return the result for inputs with the specified type, returning a null
    /// type if the input types are invalid.
    static FIRRTLType getResultType(FIRRTLType input, int32_t amount);
    static FIRRTLType getResultType(ArrayRef<FIRRTLType> inputs,
                                    ArrayRef<int32_t> integers) {
      if (inputs.size() != 1 || integers.size() != 1)
        return {};
      return getResultType(inputs[0], integers[0]);
    }
  }];
}

def ShlPrimOp : ShiftPrimOp<"shl">;
def ShrPrimOp : ShiftPrimOp<"shr">;

def TailPrimOp : PrimOp<"tail", []> {
  let arguments = (ins AnyType:$input, I32Attr:$amount,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $input $amount attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = [{
    /// Return the result for inputs with the specified type, returning a null
    /// type if the input types are invalid.
    static FIRRTLType getResultType(FIRRTLType input, int32_t amount);
    static FIRRTLType getResultType(ArrayRef<FIRRTLType> inputs,
                                    ArrayRef<int32_t> integers) {
      if (inputs.size() != 1 || integers.size() != 1)
        return {};
      return getResultType(inputs[0], integers[0]);
    }
  }];
}

#endif // FIRRTL_OPS
