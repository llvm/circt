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

def AttachOp : FIRRTLOp<"attach"> {
  let summary = "Analog Attach Statement";

  let arguments = (ins Variadic<AnalogType>:$operands);
  let results = (outs);

  let assemblyFormat = "$operands attr-dict `:` type($operands)";

  let hasCanonicalizeMethod = true;
}

def ConnectOp : FIRRTLOp<"connect"> {
  let summary = "Connect two signals";
  let description = [{
    Connect Operation:
    ```
      firrtl.connect %dest, %src : t1, t2
    ```
    }];

  let arguments = (ins FIRRTLType:$dest, FIRRTLType:$src);
  let results = (outs);

  let assemblyFormat =
    "$dest `,` $src  attr-dict `:` type($dest) `,` type($src)";

  let verifier = "return ::verifyConnectOp(*this);";
  let hasCanonicalizeMethod = true;
}

def PartialConnectOp : FIRRTLOp<"partialconnect"> {
  let summary = "Connect two signals";
  let description = [{
    Connect two values with fewer constraints:
    ```
      firrtl.partialconnect %dest, %src : t1, t2
    ```
    }];

  let arguments = (ins FIRRTLType:$dest, FIRRTLType:$src);
  let results = (outs);
  let hasCanonicalizeMethod = true;

  let verifier = "return ::verifyPartialConnectOp(*this);";

  let assemblyFormat =
    "$dest `,` $src  attr-dict `:` type($dest) `,` type($src)";
}

def PrintFOp : FIRRTLOp<"printf"> {
  let summary = "Formatted Print Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$cond, StrAttr:$formatString,
                       Variadic<FIRRTLType>:$operands, StrAttr:$name);
  let results = (outs);

  let assemblyFormat = [{
    $clock `,` $cond `,` $formatString custom<PrintfAttrs>(attr-dict)
    (`(` $operands^ `)` `:` type($operands))?
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

  let assemblyFormat = "attr-dict";
}

def StopOp : FIRRTLOp<"stop"> {
  let summary = "Stop Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$cond, I32Attr:$exitCode,
                       StrAttr:$name);
  let results = (outs);

  let assemblyFormat = "$clock `,` $cond `,` $exitCode custom<StopAttrs>(attr-dict)";
}

def AssertOp : FIRRTLOp<"assert"> {
  let summary = "Assert Verification Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$predicate, UInt1Type:$enable,
    StrAttr:$message, StrAttr:$name);
  let results = (outs);

  let assemblyFormat =
    "$clock `,` $predicate `,` $enable `,` $message custom<VerifAttrs>(attr-dict)";
}

def AssumeOp : FIRRTLOp<"assume"> {
  let summary = "Assume Verification Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$predicate, UInt1Type:$enable,
    StrAttr:$message, StrAttr:$name);
  let results = (outs);

  let assemblyFormat =
    "$clock `,` $predicate `,` $enable `,` $message custom<VerifAttrs>(attr-dict)";
}

def CoverOp : FIRRTLOp<"cover"> {
  let summary = "Cover Verification Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$predicate, UInt1Type:$enable,
    StrAttr:$message, StrAttr:$name);
  let results = (outs);

  let assemblyFormat =
    "$clock `,` $predicate `,` $enable `,` $message custom<VerifAttrs>(attr-dict)";
}

def WhenOp : FIRRTLOp<"when", [SingleBlock, NoTerminator, NoRegionArguments,
                               RecursiveSideEffects]> {
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
    OpBuilder<(ins "Value":$condition, "bool":$withElseRegion,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>
  ];

  let assemblyFormat =
    "$condition $thenRegion (`else` $elseRegion^)? attr-dict-with-keyword";

  let extraClassDeclaration = [{

    Block &getThenBlock() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      return thenRegion().front();
    }

    OpBuilder getThenBodyBuilder() {
      Block &body = getThenBlock();
      return OpBuilder::atBlockEnd(&body);
    }

    bool hasElseRegion() {
      return !elseRegion().empty();
    }

    void createElseRegion();

    Block &getElseBlock() {
      assert(hasElseRegion() && "Unexpected empty 'else' region.");
      return elseRegion().front();
    }

    OpBuilder getElseBodyBuilder() {
      Block &body = getElseBlock();
      return OpBuilder::atBlockEnd(&body);
    }
  }];
}

def ForceOp : FIRRTLOp<"force", [SameTypeOperands]> {
  let summary = "Force procedural statement";
  let description = "Maps to the corresponding `sv.force` operation.";
  let arguments = (ins FIRRTLType:$dest, FIRRTLType:$src);
  let results = (outs);
  let assemblyFormat =
    "$dest `,` $src attr-dict `:` type($dest) `,` type($src)";
}
