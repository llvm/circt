//===- OpExpressions.td - FIRRTL expression ops ------------*- tablegen -*-===//
//
// This describes the MLIR ops for FIRRTL expressions.
//
//===----------------------------------------------------------------------===//

def ConstantOp : FIRRTLOp<"constant", [NoSideEffect, ConstantLike]> {
  let summary = "Produce a constant value";
  let description = [{
    The constant operation produces a constant value of SInt or UInt type.
    ```
      %result = firrt.constant 42 : t1
    ```
    }];

  let arguments = (ins APIntAttr:$value);
  let results = (outs IntType:$result);
  
  let assemblyFormat = [{
     `(` $value `)` attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "IntType type, const APInt &value">
  ];
  let hasFolder = 1;
  let verifier = [{ return ::verify(*this); }];
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

  let arguments = (ins AnyType:$input, StrAttr:$fieldname);
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

  let arguments = (ins AnyType:$input, I32Attr:$index);
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

  let arguments = (ins AnyType:$input, AnyType:$index);
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
// Primitive Operations
//===----------------------------------------------------------------------===//

class PrimOp<string mnemonic, list<OpTrait> traits = []> :
    FIRRTLOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
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
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
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
def SubPrimOp : BinaryPrimOp<"sub", "getAddSubResult">;
def MulPrimOp : BinaryPrimOp<"mul", "getMulResult", [Commutative]>;
def DivPrimOp : BinaryPrimOp<"div", "getDivResult">;
def RemPrimOp : BinaryPrimOp<"rem", "getRemResult">;

let hasFolder = 1 in
def AndPrimOp : BinaryPrimOp<"and", "getBitwiseBinaryResult", [Commutative]>;
let hasFolder = 1 in
def OrPrimOp  : BinaryPrimOp<"or",  "getBitwiseBinaryResult", [Commutative]>;
let hasFolder = 1 in
def XorPrimOp : BinaryPrimOp<"xor", "getBitwiseBinaryResult", [Commutative]>;

// Comparison Operations
def LEQPrimOp : BinaryPrimOp<"leq", "getCompareResult">;
def LTPrimOp  : BinaryPrimOp<"lt",  "getCompareResult">;
def GEQPrimOp : BinaryPrimOp<"geq", "getCompareResult">;
def GTPrimOp  : BinaryPrimOp<"gt",  "getCompareResult">;
def EQPrimOp  : BinaryPrimOp<"eq",  "getCompareResult", [Commutative]>;
def NEQPrimOp : BinaryPrimOp<"neq", "getCompareResult", [Commutative]>;

def CatPrimOp  : BinaryPrimOp<"cat", "getCatResult">;
def DShlPrimOp : BinaryPrimOp<"dshl", "getDShlResult">;
def DShrPrimOp : BinaryPrimOp<"dshr", "getDShrResult">;

def ValidIfPrimOp : BinaryPrimOp<"validif", "getValidIfResult">;

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Base class for binary primitive operators.
class UnaryPrimOp<string mnemonic, string resultTypeFunction, 
                  list<OpTrait> traits = []> :
    PrimOp<mnemonic, traits> {
  let arguments = (ins AnyType:$input);
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

def AsSIntPrimOp       : UnaryPrimOp<"asSInt", "getAsSIntResult">;
def AsUIntPrimOp       : UnaryPrimOp<"asUInt", "getAsUIntResult">;
def AsAsyncResetPrimOp : UnaryPrimOp<"asAsyncReset", "getAsAsyncResetResult">;
def AsClockPrimOp      : UnaryPrimOp<"asClock", "getAsClockResult">;
def CvtPrimOp          : UnaryPrimOp<"cvt", "getCvtResult">;
def NegPrimOp          : UnaryPrimOp<"neg", "getNegResult">;
def NotPrimOp          : UnaryPrimOp<"not", "getNotResult">;

def AndRPrimOp         : UnaryPrimOp<"andr", "getReductionResult">;
def OrRPrimOp          : UnaryPrimOp<"orr", "getReductionResult">;
def XorRPrimOp         : UnaryPrimOp<"xorr", "getReductionResult">;

// This is an MLIR specific pseudo op.
def AsPassivePrimOp    : UnaryPrimOp<"asPassive", "getAsPassiveResult">;


//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

def BitsPrimOp : PrimOp<"bits"> {
  let arguments = (ins AnyType:$input, I32Attr:$hi, I32Attr:$lo);
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

def HeadPrimOp : PrimOp<"head"> {
  let arguments = (ins AnyType:$input, I32Attr:$amount);
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

def MuxPrimOp : PrimOp<"mux"> {
  let arguments = (ins AnyType:$sel, AnyType:$high, AnyType:$low);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, $result)
  }];

  let hasFolder = 1;

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

def PadPrimOp : PrimOp<"pad"> {
  let arguments = (ins AnyType:$input, I32Attr:$amount);
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

class ShiftPrimOp<string mnemonic> : PrimOp<mnemonic> {
  let arguments = (ins AnyType:$input, I32Attr:$amount);
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

def TailPrimOp : PrimOp<"tail"> {
  let arguments = (ins AnyType:$input, I32Attr:$amount);
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
