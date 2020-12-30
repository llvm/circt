//===- FIRRTLStatements.td - FIRRTL statement ops ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for FIRRTL statements.
//
//===----------------------------------------------------------------------===//

def MemoryPortOp : FIRRTLOp<"memoryport", []> {
  let summary = "Access a memory";

  let arguments = (ins FIRRTLType:$memory, IntType:$index, ClockType:$clock,
                       MemDirAttr:$direction, OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);

  let assemblyFormat = [{
    $direction $memory `,` $index `,` $clock attr-dict `:`
       functional-type(operands, $result)
  }];
}

def AttachOp : FIRRTLOp<"attach"> {
  let summary = "Analog Attach Statement";

  let arguments = (ins Variadic<AnalogType>:$operands);
  let results = (outs);

  let assemblyFormat = "$operands attr-dict `:` type($operands)";
}

def ConnectOp : FIRRTLOp<"connect", []> {
  let summary = "Connect two signals";
  let description = [{
    Connect Operation:
    ```
      firrtl.connect %dest, %src : t1, t2
    ```
    }];

  let arguments = (ins FIRRTLType:$dest, FIRRTLType:$src);
  let results = (outs);

  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` type($dest) `,` type($src)
  }];

  let verifier = [{ return ::verifyConnectOp(*this); }];
}

def PartialConnectOp : FIRRTLOp<"partialconnect", []> {
  let summary = "Connect two signals";
  let description = [{
    Connect two values with fewer constraints:
    ```
      firrtl.partialconnect %dest, %src : t1, t2
    ```
    }];

  let arguments = (ins FIRRTLType:$dest, FIRRTLType:$src);
  let results = (outs);

  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` type($dest) `,` type($src)
  }];
}

def InvalidOp : FIRRTLOp<"invalid", []> {
  let summary = "Invalid statement";
  let description = [{
    Invalidate Operation:
    ```
      firrtl.invalid %result : !firrtl.uint<1>
    ```

    The invalid operation produces a value of unspecified value.
  }];

  let arguments = (ins FIRRTLType:$operand);
  let results = (outs);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand)
  }];
}

def PrintFOp : FIRRTLOp<"printf", []> {
  let summary = "Formatted Print Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$cond, StrAttr:$formatString,
                       Variadic<FIRRTLType>:$operands);
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

def AssertOp : FIRRTLOp<"assert", []> {
  let summary = "Assert Verification Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$predicate, UInt1Type:$enable,
    StrAttr:$message);
  let results = (outs);

  let assemblyFormat = [{
    $clock `,` $predicate `,` $enable `,` $message attr-dict
  }];
}

def AssumeOp : FIRRTLOp<"assume", []> {
  let summary = "Assume Verification Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$predicate, UInt1Type:$enable,
    StrAttr:$message);
  let results = (outs);

  let assemblyFormat = [{
    $clock `,` $predicate `,` $enable `,` $message attr-dict
  }];
}

def CoverOp : FIRRTLOp<"cover", []> {
  let summary = "Cover Verification Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$predicate, UInt1Type:$enable,
    StrAttr:$message);
  let results = (outs);

  let assemblyFormat = [{
    $clock `,` $predicate `,` $enable `,` $message attr-dict
  }];
}

def WhenOp : FIRRTLOp<"when", [SingleBlockImplicitTerminator<"DoneOp">,
                               NoRegionArguments, RecursiveSideEffects]> {
  let summary = "When Statement";
  let description = [{
    The "firrtl.when" operation represents a conditional.  Connections within
    a conditional statement that connect to previously declared components hold
    only when the given condition is high. The condition must have a 1-bit
    unsigned integer type.
  }];
  let arguments = (ins UInt1Type:$condition);
  let results = (outs);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "Value":$condition, "bool":$withElseRegion,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>
  ];

  let assemblyFormat = [{
    $condition $thenRegion (`else` $elseRegion^)? attr-dict-with-keyword
  }];

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
}
