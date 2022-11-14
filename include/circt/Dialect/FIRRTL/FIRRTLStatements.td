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

  let arguments = (ins Variadic<AnalogType>:$attached);
  let results = (outs);

  let assemblyFormat = "$attached attr-dict `:` qualified(type($attached))";

  let hasCanonicalizeMethod = true;
}

def ConnectOp : FIRRTLOp<"connect", [FConnectLike]> {
  let summary = "Connect two signals";
  let description = [{
    Connect Operation:
    ```
      firrtl.connect %dest, %src : t1, t2
    ```
    }];

  let arguments = (ins ConnectableType:$dest, ConnectableType:$src);
  let results = (outs);

  let assemblyFormat =
    "$dest `,` $src  attr-dict `:` qualified(type($dest)) `,` qualified(type($src))";

  let hasVerifier = 1;
  let hasCanonicalizer = true;
}

def StrictConnectOp : FIRRTLOp<"strictconnect", [SameTypeOperands, FConnectLike]> {
  let summary = "Connect two signals";
  let description =  [{
    Connect two values with strict constraints:
    ```
      firrtl.strictconnect %dest, %src : t1
    ```
    }];

  let arguments = (ins SizedBaseOrRefTypeOrForeignType:$dest,
                       SizedBaseOrRefTypeOrForeignType:$src);
  let results = (outs);
  let hasCanonicalizeMethod = true;

  let hasVerifier = 1;

  let assemblyFormat =
    "$dest `,` $src  attr-dict `:` qualified(type($dest))";
}

def PrintFOp : FIRRTLOp<"printf"> {
  let summary = "Formatted Print Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$cond, StrAttr:$formatString,
                       Variadic<FIRRTLBaseType>:$substitutions, StrAttr:$name);
  let results = (outs);

  let assemblyFormat = [{
    $clock `,` $cond `,` $formatString custom<PrintfAttrs>(attr-dict)
    (`(` $substitutions^ `)` `:` qualified(type($substitutions)))?
  }];
}

def SkipOp : FIRRTLOp<"skip", [Pure]> {
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

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"AtPosEdge", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"AtNegEdge", 1, "negedge">;
/// AtEdge(v) is syntactic sugar for "AtPosEdge(v) or AtNegEdge(v)".
def AtEdge   : I32EnumAttrCase<"AtEdge", 2, "edge">;

def EventControlAttr : I32EnumAttr<"EventControl", "edge control trigger",
                                   [AtPosEdge, AtNegEdge, AtEdge]>  {}

class VerifOp<string mnemonic, list<Trait> traits = []> :
    FIRRTLOp<mnemonic, traits> {
  let arguments = (ins
    ClockType:$clock,
    UInt1Type:$predicate,
    UInt1Type:$enable,
    StrAttr:$message,
    Variadic<AnyType>:$substitutions,
    StrAttr:$name,
    DefaultValuedAttr<BoolAttr,"false">:$isConcurrent,
    DefaultValuedAttr<EventControlAttr,"EventControl::AtPosEdge">:$eventControl
  );

  let hasCanonicalizer = true;

  let assemblyFormat = [{
    $clock `,` $predicate `,` $enable `,`
    $message (`(` $substitutions^ `)` `:` qualified(type($substitutions)))?
    custom<VerifAttrs>(attr-dict)
  }];
}

def AssertOp : VerifOp<"assert"> {
  let summary = "Assert Verification Statement";
}

def AssumeOp : VerifOp<"assume"> {
  let summary = "Assume Verification Statement";
}

def CoverOp : VerifOp<"cover"> {
  let summary = "Cover Verification Statement";
}

def WhenOp : FIRRTLOp<"when", [SingleBlock, NoTerminator, NoRegionArguments,
                               RecursiveMemoryEffects, RecursivelySpeculatable]> {
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
      assert(!getThenRegion().empty() && "Unexpected empty 'then' region.");
      return getThenRegion().front();
    }

    OpBuilder getThenBodyBuilder() {
      Block &body = getThenBlock();
      return OpBuilder::atBlockEnd(&body);
    }

    bool hasElseRegion() {
      return !getElseRegion().empty();
    }

    void createElseRegion();

    Block &getElseBlock() {
      assert(hasElseRegion() && "Unexpected empty 'else' region.");
      return getElseRegion().front();
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
  let arguments = (ins FIRRTLBaseType:$dest, FIRRTLBaseType:$src);
  let results = (outs);
  let assemblyFormat =
    "$dest `,` $src attr-dict `:` qualified(type($dest)) `,` qualified(type($src))";
}

def ProbeOp : FIRRTLOp<"probe"> {
  let summary = "FIRRTL Value Probe";
  let description = [{
    Captures values without binding to any accidental name.
  }];
  let arguments = (ins SymbolNameAttr:$inner_sym, Variadic<FIRRTLBaseType>:$captured);
  let results = (outs);

  let assemblyFormat = "$inner_sym attr-dict ( `,` $captured^  `:` qualified(type($captured)))?";
}
