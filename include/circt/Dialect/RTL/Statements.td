//===- Statements.td - RTL statements ops ------------------*- tablegen -*-===//
//
// This describes the MLIR ops for statements.
//
//===----------------------------------------------------------------------===//

def ConnectOp : RTLOp<"connect", [SameTypeOperands]> {
  let summary = "Connect two signals";
  let description = [{
    Connect Operation:
    ```
      rtl.connect %dest, %src : t1
    ```
    }];

  let arguments = (ins AnyType:$dest, AnyType:$src);
  let results = (outs);
  
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` type($dest)
  }];
}

def WireOp : RTLOp<"wire", [DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = rtl.wire : t1
    ```
    }];

  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs InOutType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "::mlir::Type":$elementType,
                      CArg<"StringAttr", "StringAttr()">:$name)>
  ];

  // We handle the name in a custom way, so we use a customer parser/printer.
  let printer = [{ printWireOp(p, *this); }];
  let parser = [{ return parseWireOp(parser, result); }];
}

def ReadInOutOp
 : RTLOp<"read_inout", [NoSideEffect, InOutTypeConstraint<"result", "input">]> {
  let summary = "Get the value of from something of inout type (e.g. a wire or"
                " inout port) as the value itself.";
  let arguments = (ins InOutType:$input);
  let results = (outs InOutElementType:$result);

  let builders = [
    OpBuilderDAG<(ins "Value":$input)>
  ];

  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];

  let verifier = [{ return ::verifyReadInOutOp(*this); }];
}
