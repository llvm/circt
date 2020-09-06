//===- Statements.td - SV statements ops -------------------*- tablegen -*-===//
//
// This describes the ops for System Verilog statements.
//
//===----------------------------------------------------------------------===//

class HasCustomParserPrinter {
  code printer = [{ return ::print$cppClass(p, *this); }];
  code parser = [{ return ::parse$cppClass(parser, result); }];
}

def HasRegionTerminator : SingleBlockImplicitTerminator<"YieldOp">;

//===----------------------------------------------------------------------===//
// Control flow like-operations
//===----------------------------------------------------------------------===//

def IfDefOp : SVOp<"ifdef", [HasRegionTerminator]>, HasCustomParserPrinter {
  let summary = "'ifdef MACRO' block";
   
   let description = [{
    This operation is an #ifdef block, or an #ifndef block if `string` starts
    with a `!` character.
    }];

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins StrAttr:$cond);
  let results = (outs);
  
  // TODO: ODS doesn't support parsing/printing regions yet :-(
  // let assemblyFormat = [{ $clock attr-dict $body }];*/

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "StringRef cond">
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &body().front(); }
  }];
}

def IfOp : SVOp<"if", [HasRegionTerminator]>,
           HasCustomParserPrinter {
  let summary = "'if (cond)' block";
  
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins I1:$cond);
  let results = (outs);
  
  // TODO: ODS doesn't support parsing/printing regions yet :-(
  //let assemblyFormat = [{ $cond attr-dict $body }];

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value cond">
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &body().front(); }
  }];
}

// TODO: This should be generalized, e.g. with an enum to specify the edge
// trigger behavior.
def AlwaysAtPosEdgeOp : SVOp<"alwaysat_posedge", [HasRegionTerminator]>,
                        HasCustomParserPrinter {
  let summary = "'always @(posedge ...)'' block";
  
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins I1:$clock);
  let results = (outs);
  
  // TODO: ODS doesn't support parsing/printing regions yet :-(
  //let assemblyFormat = [{ $clock attr-dict $body }];

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value cond">
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

// TODO: This needs to model the file descriptor to write on as well as the
// arguments.
def FWriteOp : SVOp<"fwrite", []> {
  let summary = "'$fwrite' statement";
   
  let arguments = (ins StrAttr:$string);
  let results = (outs);

  let assemblyFormat = [{
    $string attr-dict
  }];
}
