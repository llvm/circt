//===- SVStatements.td - SV statements ops -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for System Verilog statements.
//
//===----------------------------------------------------------------------===//

def HasRegionTerminator : SingleBlockImplicitTerminator<"YieldOp">;

//===----------------------------------------------------------------------===//
// Control flow like-operations
//===----------------------------------------------------------------------===//

def IfDefOp : SVOp<"ifdef", [HasRegionTerminator, NoRegionArguments,
                             RecursiveSideEffects]> {
  let summary = "'ifdef MACRO' block";
   
   let description = [{
    This operation is an #ifdef block, or an #ifndef block if `string` starts
    with a `!` character.
    }];

  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
  let arguments = (ins StrAttr:$cond);
  let results = (outs);
  
  let assemblyFormat = [{ $cond $thenRegion (`else` $elseRegion^)? attr-dict }];

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "StringRef":$cond,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>
  ];

  let extraClassDeclaration = [{
    /// Returns true if an else block exists.
    bool hasElse() { return !elseRegion().empty(); }

    Block *getThenBlock() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      return &thenRegion().front();
    }

    Block *getElseBlock() {
      assert(hasElse() && "Empty 'else' region.");
      return &elseRegion().front();
    }
  }];
}

def IfOp : SVOp<"if", [HasRegionTerminator, NoRegionArguments,
                       RecursiveSideEffects]> {
  let summary = "'if (cond)' block";
  
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins I1:$cond);
  let results = (outs);
  
  let assemblyFormat = [{ $cond $body attr-dict }];

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "Value":$cond,
                      CArg<"std::function<void()>", "{}">:$bodyCtor)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &body().front(); }
  }];
}

// Edge behavior enum for always block.  See SV Spec 9.4.2.

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"AtPosEdge", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"AtNegEdge", 1, "negedge">;
/// AtEdge(v) is syntactic sugar for "AtPosEdge(v) or AtNegEdge(v)".
def AtEdge   : I32EnumAttrCase<"AtEdge", 2, "edge">;

def EventControlAttr : I32EnumAttr<"EventControl", "edge control trigger",
                                   [AtPosEdge, AtNegEdge, AtEdge]>;


def AlwaysOp : SVOp<"always", [HasRegionTerminator, NoRegionArguments,
                               RecursiveSideEffects]> {
  let summary = "'always @' block";
  let description = "See SV Spec 9.2, and 9.4.2.2.";
  
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins TypedArrayAttrBase<EventControlAttr, "events">:$events,
                       Variadic<I1>:$clocks);
  let results = (outs);
  
  let assemblyFormat = "custom<EventList>($events, $clocks) $body attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "ArrayRef<EventControl>":$event, "ArrayRef<Value>":$cond,
                      CArg<"std::function<void()>", "{}">:$bodyCtor)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &body().front(); }

    struct Condition {
      EventControl event;
      Value value;
    };

    size_t getNumConditions() { return events().size(); }
    Condition getCondition(size_t idx);
  }];

  // Check that we have the same number of events and conditions.
  let verifier = [{ return ::verifyAlwaysOp(*this); }];
}

def InitialOp : SVOp<"initial", [HasRegionTerminator, NoRegionArguments,
                                 RecursiveSideEffects]> {
  let summary = "'initial' block";
  let description = "See SV Spec 9.2.1.";

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = [{ $body attr-dict }];

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins CArg<"std::function<void()>", "{}">:$bodyCtor)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &body().front(); }
  }];
}

//===----------------------------------------------------------------------===//
// Other Statements
//===----------------------------------------------------------------------===//

def YieldOp
  : SVOp<"yield", [NoSideEffect, Terminator,
                   ParentOneOf<["IfDefOp, IfOp", "AlwaysOp", "InitialOp"]>]> {
  let summary = "terminator for control-flow operation regions";
  let arguments = (ins);

   let assemblyFormat = [{attr-dict}];
}

def BPAssignOp : SVOp<"bpassign", [InOutTypeConstraint<"src", "dest">]> {
  let summary = "Blocking procedural assignment";
  let description = [{
    A SystemVerilog blocking procedural assignment statement 'x = y;'.  These
    occur in initial, always, task, and function blocks.  The statement is
    executed before any following statements are. See SV Spec 10.4.1.
  }];
  let arguments = (ins InOutType:$dest, InOutElementType:$src);
  let results = (outs);
  let assemblyFormat = [{ $dest `,` $src  attr-dict `:` type($src) }];
}

def PAssignOp : SVOp<"passign", [InOutTypeConstraint<"src", "dest">]> {
  let summary = "Nonblocking procedural assignment";
  let description = [{
    A SystemVerilog nonblocking procedural assignment statement 'x <= y;'. 
    These occur in initial, always, task, and function blocks.  The statement
    can be scheduled without blocking procedural flow.  See SV Spec 10.4.2.
  }];
  let arguments = (ins InOutType:$dest, InOutElementType:$src);
  let results = (outs);
  let assemblyFormat = [{ $dest `,` $src  attr-dict `:` type($src) }];
}

def AliasOp : SVOp<"alias"> {
  let summary = "SystemVerilog 'alias' statement";
  let description = [{
    An alias statement declares multiple names for the same physical net, or
    bits within a net.  Aliases always have at least two operands.
  }];

  let arguments = (ins Variadic<InOutType>:$operands);
  let results = (outs);

  let assemblyFormat = "$operands attr-dict `:` type($operands)";

  // Verify that we have at least two operands.
  let verifier = [{ return ::verifyAliasOp(*this); }];
}

// TODO: This needs to model the file descriptor to write on.  This is currently
// hard coded to 32'h80000002.
def FWriteOp : SVOp<"fwrite"> {
  let summary = "'$fwrite' statement";
   
  let arguments = (ins StrAttr:$string, Variadic<AnyType>:$operands);
  let results = (outs);

  let assemblyFormat = [{
    $string attr-dict (`(` $operands^ `)` `:` type($operands))?
  }];
}

def FinishOp : SVOp<"finish"> {
  let summary = "'$finish' statement";
  let description = [{
    Finishes a simulation and exits the simulation process.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = [{ attr-dict }];
}

def FatalOp : SVOp<"fatal"> {
  let summary = "'$fatal' statement";
  let description = [{
    Run-time fatal assertion error.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = [{ attr-dict }];
}

def VerbatimOp : SVOp<"verbatim"> {
  let summary = "verbatim opaque text emitted to SystemVerilog file.";
  let summary = "sv.verbatim allows operand substitutions with {{0}} syntax.";
  let arguments = (ins StrAttr:$string, Variadic<AnyType>:$operands);

  let assemblyFormat =
    "$string attr-dict (`(` $operands^ `)` `:` type($operands))?";

  let builders = [
    OpBuilderDAG<(ins "StringRef":$string),
                 "build(odsBuilder, odsState, string, ValueRange{});">
  ];
}

//===----------------------------------------------------------------------===//
// Verification Statements
//===----------------------------------------------------------------------===//

def AssertOp : SVOp<"assert"> {
  let summary = "immediate assertion statement";
  let description = [{
    Assert that a specific condition is always true.
  }];

  let arguments = (ins AnyType:$predicate);
  let results = (outs);
  let assemblyFormat = "attr-dict $predicate `:` type($predicate)";
}

def AssumeOp : SVOp<"assume"> {
  let summary = "assume property statement";
  let description = [{
    Assume that a specific property is always true.
  }];

  let arguments = (ins AnyType:$property);
  let results = (outs);
  let assemblyFormat = "attr-dict $property `:` type($property)";
}

def CoverOp : SVOp<"cover"> {
  let summary = "functional coverage statement";
  let description = [{
    Assert that a specific property happens during the course of execution.
  }];

  let arguments = (ins AnyType:$property);
  let results = (outs);
  let assemblyFormat = "attr-dict $property `:` type($property)";
}
