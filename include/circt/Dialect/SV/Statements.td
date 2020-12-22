//===- Statements.td - SV statements ops -------------------*- tablegen -*-===//
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

def IfDefOp : SVOp<"ifdef", [HasRegionTerminator]> {
  let summary = "'ifdef MACRO' block";
   
   let description = [{
    This operation is an #ifdef block, or an #ifndef block if `string` starts
    with a `!` character.
    }];

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins StrAttr:$cond);
  let results = (outs);
  
  let assemblyFormat = [{ $cond regions attr-dict }];

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "StringRef":$cond,
                      CArg<"std::function<void()>", "{}">:$bodyCtor)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &body().front(); }
  }];
}

def IfOp : SVOp<"if", [HasRegionTerminator]> {
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

// TODO: This should be generalized, e.g. with an enum to specify the edge
// trigger behavior.
def AlwaysAtPosEdgeOp : SVOp<"alwaysat_posedge", [HasRegionTerminator]> {
  let summary = "'always @(posedge ...)'' block";
  
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins I1:$clock);
  let results = (outs);
  
  let assemblyFormat = [{ $clock $body attr-dict }];

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

//===----------------------------------------------------------------------===//
// Other Statements
//===----------------------------------------------------------------------===//

def YieldOp
  : SVOp<"yield", [NoSideEffect, Terminator,
                   ParentOneOf<["IfDefOp, IfOp", "AlwaysAtPosEdgeOp"]>]> {
  let summary = "terminator for control-flow operation regions";
  let arguments = (ins);

   let assemblyFormat = [{attr-dict}];
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

  let assemblyFormat = [{ attr-dict $predicate `:` type($predicate) }];
}

def AssumeOp : SVOp<"assume"> {
  let summary = "assume property statement";
  let description = [{
    Assume that a specific property is always true.
  }];

  let arguments = (ins AnyType:$property);
  let results = (outs);

  let assemblyFormat = [{ attr-dict $property `:` type($property) }];
}

def CoverOp : SVOp<"cover"> {
  let summary = "functional coverage statement";
  let description = [{
    Assert that a specific property happens during the course of execution.
  }];

  let arguments = (ins AnyType:$property);
  let results = (outs);

  let assemblyFormat = [{ attr-dict $property `:` type($property) }];
}
