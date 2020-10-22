//===- OpDeclarations.td - FIRRTL declaration ops ----------*- tablegen -*-===//
//
// This describes the MLIR ops for FIRRTL declarations.
//
//===----------------------------------------------------------------------===//

def InstanceOp : FIRRTLOp<"instance"> {
  let summary = "Instantiate an instance of a module";
  let description = [{
    This represents an instance of a module.  The result is always a bundle
    that is a flattened form of the referenced modules inputs and outputs.
  }];

  let arguments = (ins FlatSymbolRefAttr:$moduleName,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);
  
  let assemblyFormat = [{
     $moduleName attr-dict `:` type($result)
  }];
}

def CMemOp : FIRRTLOp<"cmem", [/*MemAlloc*/]> {
  let summary = "Define a new cmem";
  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);
  
  let assemblyFormat = [{
     attr-dict `:` type($result)
  }];
}

def SMemOp : FIRRTLOp<"smem", [/*MemAlloc*/]> {
  let summary = "Define a new smem";
  let arguments = (ins RUWAttr:$ruw, OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);

  let assemblyFormat = [{
    $ruw attr-dict `:` type($result)
  }];
}

def MemOp : FIRRTLOp<"mem", [/*MemAlloc*/]> {
  let summary = "Define a new mem";
  let arguments = (ins I32Attr:$readLatency,
                       Confined<I32Attr, [IntMinValue<1>]>:$writeLatency,
                       I64Attr:$depth, RUWAttr:$ruw,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);

  let assemblyFormat = [{
    $ruw attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    enum class PortKind { Read, Write, ReadWrite };

    /// Return the type of a mem given a list of named ports and their kind.
    /// This returns a null type if there are duplicate port names.
    static FIRRTLType getTypeForPortList(uint64_t depth, FIRRTLType dataType,
                              ArrayRef<std::pair<Identifier, PortKind>> ports);

    /// Return the name and kind of ports supported by this memory.
    void getPorts(SmallVectorImpl<std::pair<Identifier, PortKind>> &result);

    /// Return the kind of the specified port or None if the name is invalid.
    Optional<PortKind> getPortKind(StringRef portName);

    /// Return the data-type field of the memory, the type of each element.  If
    /// the Mem has no read/write ports, this will return null.
    FIRRTLType getDataTypeOrNull();
  }];
}

def NodeOp : FIRRTLOp<"node", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "No-op to name a value";
  let description = [{
    A node is simply a named intermediate value in a circuit. The node must
    be initialized to a value with a passive type and cannot be connected to.
    Nodes are often used to split a complicated compound expression into named
    subexpressions.

    ```
      %result = firrtl.node %input : t1
    ```
    }];

  let arguments = (ins FIRRTLType:$input, OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);
  
  let assemblyFormat = [{
     $input attr-dict `:` type($input)
  }];
}

def RegOp : FIRRTLOp<"reg", [/*MemAlloc*/]> {
  let summary = "Define a new register";
  let description = [{
    Declare a new register:
    ```
      %name = firrtl.reg %clockVal : t1
    ```
    }];

  let arguments = (ins ClockType:$clockVal, OptionalAttr<StrAttr>:$name);
  let results = (outs PassiveType:$result);

  let assemblyFormat = [{
     operands attr-dict `:` functional-type(operands, $result)
  }];
}

def RegInitOp : FIRRTLOp<"reginit", [/*MemAlloc*/]> {
  let summary = "Define a new register with an initializer";
  let description = [{
    Declare a new register:
    ```
      %name = firrtl.reginit %clockVal, %resetSignal, %resetValue : t1
    ```
    }];

  let arguments = (ins ClockType:$clockVal, ResetType:$resetSignal,
                       PassiveType:$resetValue, OptionalAttr<StrAttr>:$name);
  let results = (outs PassiveType:$result);

  let assemblyFormat = [{
     operands attr-dict `:` functional-type(operands, $result)
  }];
}

def WireOp : FIRRTLOp<"wire", []> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = firrtl.wire : t1
    ```
    }];

  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);
  
  let assemblyFormat = [{
     attr-dict `:` type($result)
  }];
}

