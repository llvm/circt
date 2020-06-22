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

def WireOp : RTLOp<"wire"> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = rtl.wire : t1
    ```
    }];

  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
     attr-dict `:` type($result)
  }];
}
