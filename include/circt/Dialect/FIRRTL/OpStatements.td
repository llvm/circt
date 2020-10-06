//===- OpStatements.td - FIRRTL statement ops --------------*- tablegen -*-===//
//
// This describes the MLIR ops for FIRRTL statements.
//
//===----------------------------------------------------------------------===//

def MemoryPortOp : FIRRTLOp<"memoryport", []> {
  let summary = "Access a memory";

  let arguments = (ins AnyType:$memory, AnyType:$index, AnyType:$clock,
                       MemDirAttr:$direction, OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $direction $memory `,` $index `,` $clock attr-dict `:`
       functional-type(operands, $result)
  }];
}

def AttachOp : FIRRTLOp<"attach"> {
  let summary = "Analog Attach Statement";

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs);

  let assemblyFormat = [{
    `(` $operands `)` attr-dict `:` type($operands)
  }];
  // TODO: add verifier that the operands are all Analog
}

def ConnectOp : FIRRTLOp<"connect", []> {
  let summary = "Connect two signals";
  let description = [{
    Connect Operation:
    ```
      firrtl.connect %lhs, %rhs : t1, t2
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
      firrtl.partialconnect %lhs, %rhs : t1, t2
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
       %result = firrtl.invalid : i8
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
    $clock `,` $cond `,` $formatString attr-dict (`(` $operands^ `)`
    `:` type($operands))?
  }];
}

def SkipOp : FIRRTLOp<"skip", [NoSideEffect]> {
  let summary = "Skip statement";
  let description = [{
    Skip Statement:
    ```
       %firrtl.skip
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
    OpBuilder<"Value condition, bool withElseRegion">
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
