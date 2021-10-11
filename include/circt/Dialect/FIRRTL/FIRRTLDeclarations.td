//===- FIRRTLDeclarations.td - FIRRTL declaration ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for FIRRTL declarations.
//
//===----------------------------------------------------------------------===//

def InstanceOp : FIRRTLOp<"instance", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Instantiate an instance of a module";
  let description = [{
    This represents an instance of a module.  The results are the modules inputs
    and outputs.  The inputs have flip type, the outputs do not.
  }];

  let arguments = (ins FlatSymbolRefAttr:$moduleName, StrAttr:$name,
                       AnnotationArrayAttr:$annotations,
                       PortAnnotationsAttr:$portAnnotations,
                       BoolAttr:$lowerToBind);
  let results = (outs Variadic<FIRRTLType>:$results);

  let assemblyFormat =
    "$moduleName custom<InstanceOp>(attr-dict) (`:` type($results)^ )?";

  let verifier = "return ::verifyInstanceOp(*this);";

  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "::mlir::StringRef":$moduleName,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>", "{}">:$annotations,
                   CArg<"ArrayRef<Attribute>", "{}">:$portAnnotations,
                   CArg<"bool","false">:$lowerToBind)>,

    /// Constructor when you have the target module in hand.
    OpBuilder<(ins "FModuleLike":$module,
                   "mlir::StringRef":$name,
                   CArg<"ArrayRef<Attribute>", "{}">:$annotations,
                   CArg<"ArrayRef<Attribute>", "{}">:$portAnnotations,
                   CArg<"bool","false">:$lowerToBind)>
  ];

  let extraClassDeclaration = [{
    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    FModuleLike getReferencedModule();
    FModuleLike getReferencedModule(SymbolTable& symtbl);

    /// Return the port name for the specified result number.
    StringAttr getPortName(size_t resultNo) {
      return getReferencedModule().getPortNameAttr(resultNo);
    }
    StringRef getPortNameStr(size_t resultNo) {
      return getPortName(resultNo).getValue();
    }

    /// Hooks for port annotations.
    ArrayAttr getPortAnnotation(unsigned portIdx);
    void setAllPortAnnotations(ArrayRef<Attribute> annotations);
  }];
}

def CombMemOp : FIRRTLOp<"combmem"> {
  let summary = "Define a new combinational memory";
  let description = [{
    Define a new behavioral combinational memory. Combinational memories have a
    write latency of 1 and a read latency of 0.
  }];
  let arguments = (ins StrAttr:$name, AnnotationArrayAttr:$annotations);
  let results = (outs CMemoryType:$result);
  let assemblyFormat = "custom<ImplicitSSAName>(attr-dict) `:` type($result)";
  let hasCanonicalizer = true;
}

def SeqMemOp : FIRRTLOp<"seqmem"> {
  let summary = "Define a new sequential memory";
  let description = [{
    Define a new behavioral sequential memory.  Sequential memories have a
    write latency and a read latency of 1.
  }];
  let arguments = (ins RUWAttr:$ruw, StrAttr:$name,
                       AnnotationArrayAttr:$annotations);
  let results = (outs CMemoryType:$result);
  let assemblyFormat = "$ruw custom<SeqMemOp>(attr-dict) `:` type($result)";
  let hasCanonicalizer = true;
}

def MemoryPortOp : FIRRTLOp<"memoryport", [InferTypeOpInterface,
         DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Declares a memory port on a memory";

  let summary = "Defines a memory port on CHIRRTL memory";
  let description = [{
    This operation defines a new memory port on a `seqmem` or `combmem`CHISEL.
    `data` is the data returned from the memory port.  

    The memory port requires an access point, which sets the enable condition
    of the port, the clock, and the address.  This is done by passing the the
    `port` argument to a `firrtl.memoryport.access operation`.

  }];

  let arguments = (ins CMemoryType:$memory, MemDirAttr:$direction,
                    StrAttr:$name, AnnotationArrayAttr:$annotations);

  let results = (outs FIRRTLType:$data, CMemoryPortType:$port);

  let assemblyFormat = [{
    $direction $memory custom<MemoryPortOp>(attr-dict) `:`
       functional-type(operands, results)
  }];
  let hasCanonicalizer = true;

  let builders = [
    OpBuilder<(ins "::mlir::Type":$dataType, "::mlir::Value":$memory,
                   "MemDirAttr":$direction, CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations)>
  ];

  let verifier = "return ::verifyMemoryPortOp(*this);";

  let extraClassDeclaration = [{
    /// Get the assocated access op.
    MemoryPortAccessOp getAccess();

    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          Optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];
}

def MemoryPortAccessOp : FIRRTLOp<"memoryport.access"> {
  let summary = "Enables a memory port";

  let description = [{
    This operation is used to conditionally enable a memory port, and associate
    it with a `clock` and `index`.  The memory port will be actuve on the
    positive edge of the clock. The index is the address of the memory
    accessed. See the FIRRTL rational for more information about why this
    operation exists.
  }];

  let arguments = (ins CMemoryPortType:$port, IntType:$index,
                   ClockType:$clock);

  let assemblyFormat = [{
    $port `[` $index `]` `,` $clock attr-dict `:` type(operands)
  }];
}

def MemOp : FIRRTLOp<"mem"> {
  let summary = "Define a new mem";
  let arguments =
    (ins Confined<I32Attr, [IntMinValue<0>]>:$readLatency,
         Confined<I32Attr, [IntMinValue<1>]>:$writeLatency,
         Confined<I64Attr, [IntMinValue<1>]>:$depth, RUWAttr:$ruw,
         StrArrayAttr:$portNames, StrAttr:$name,
         AnnotationArrayAttr:$annotations,
         PortAnnotationsAttr:$portAnnotations);
  let results = (outs Variadic<FIRRTLType>:$results);

  let assemblyFormat = "$ruw custom<MemOp>(attr-dict) `:` type($results)";

  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "uint32_t":$readLatency, "uint32_t":$writeLatency,
                   "uint64_t":$depth, "RUWAttr":$ruw,
                   "ArrayRef<Attribute>":$portNames,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>", "{}">:$annotations,
                   CArg<"ArrayRef<Attribute>", "{}">:$portAnnotations)>
  ];

  let verifier = "return ::verifyMemOp(*this);";

  let hasCanonicalizeMethod = true;

  let extraClassDeclaration = [{
    enum class PortKind { Read, Write, ReadWrite };

    using NamedPort = std::pair<StringAttr, MemOp::PortKind>;

    /// Return the type of a port given the memory depth, type, and kind
    static BundleType getTypeForPort(uint64_t depth, FIRRTLType dataType,
                                     PortKind portKind, size_t maskBits = 0);

    /// Return the name and kind of ports supported by this memory.
    SmallVector<NamedPort> getPorts();

    /// Return the kind of the specified port.
    PortKind getPortKind(StringRef portName);

    /// Return the kind of the specified port number.
    PortKind getPortKind(size_t resultNo);

    /// Return the data-type field of the memory, the type of each element.
    FIRRTLType getDataType();

    /// Return the number of mask bits.
    size_t getMaskBits();

    /// Return the port name for the specified result number.
    StringAttr getPortName(size_t resultNo);
    StringRef getPortNameStr(size_t resultNo) {
      return getPortName(resultNo).getValue();
    }

    /// Return the port type for the specified result number.
    FIRRTLType getPortType(size_t resultNo);

    // Return the result for this instance that corresponds to the specified
    // port name.
    Value getPortNamed(StringRef name) {
      return getPortNamed(StringAttr::get(getContext(), name));
    }
    Value getPortNamed(StringAttr name);

    /// Hooks for port annotations.
    ArrayAttr getPortAnnotation(unsigned portIdx);
    void setAllPortAnnotations(ArrayRef<Attribute> annotations);

    /// Get the number of read ports, write ports and read write ports.
    void getNumPorts(size_t &numReadPorts, size_t &numWritePorts, size_t &numReadWritePorts);

    // Extract the relevant attributes from the MemOp and return a FirMemory object.
    FirMemory analyzeMemOp();
  }];
}

def NodeOp : FIRRTLOp<"node",
                      [SameOperandsAndResultType, InferTypeOpInterface]> {
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

  let arguments = (ins PassiveType:$input, StrAttr:$name,
                       AnnotationArrayAttr:$annotations);
  let results = (outs FIRRTLType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType,
                   "::mlir::Value":$input,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations), [{
      return build($_builder, $_state, elementType, input, name,
                   $_builder.getArrayAttr(annotations));
    }]>
  ];

  let assemblyFormat = "$input custom<ImplicitSSAName>(attr-dict) `:` type($input)";

  let hasCanonicalizer = true;

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          Optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
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

  let arguments =
    (ins ClockType:$clockVal, StrAttr:$name, AnnotationArrayAttr:$annotations);
  let results = (outs PassiveType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations), [{
      return build($_builder, $_state, elementType, clockVal, name,
                   $_builder.getArrayAttr(annotations));
    }]>
  ];

  let assemblyFormat =
    "operands custom<ImplicitSSAName>(attr-dict) `:` type($result)";
  let hasCanonicalizeMethod = true;
}

def RegResetOp : FIRRTLOp<"regreset", [/*MemAlloc*/]> {
  let summary = "Define a new register with a reset";
  let description = [{
    Declare a new register:
    ```
      %name = firrtl.regreset %clockVal, %resetSignal, %resetValue : t1
    ```
    }];

  let arguments = (
    ins ClockType:$clockVal, AnyResetType:$resetSignal, PassiveType:$resetValue,
        StrAttr:$name, AnnotationArrayAttr:$annotations);
  let results = (outs PassiveType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   "::mlir::Value":$resetSignal, "::mlir::Value":$resetValue,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations), [{
      return build($_builder, $_state, elementType, clockVal, resetSignal,
                   resetValue, name,
                   $_builder.getArrayAttr(annotations));
    }]>
  ];

  let assemblyFormat =
     "operands custom<ImplicitSSAName>(attr-dict) `:` type($resetSignal) `,` type($resetValue) `,` type($result)";

  let hasCanonicalizer = true;
}

def WireOp : FIRRTLOp<"wire", []> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = firrtl.wire : t1
    ```
    }];

  let arguments = (ins StrAttr:$name, AnnotationArrayAttr:$annotations);
  let results = (outs FIRRTLType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType,
                      CArg<"StringRef", "{}">:$name,
                      CArg<"ArrayRef<Attribute>","{}">:$annotations), [{
      return build($_builder, $_state, elementType, name,
                   $_builder.getArrayAttr(annotations));
    }]>
  ];

  let assemblyFormat = "custom<ImplicitSSAName>(attr-dict) `:` type($result)";
  let hasCanonicalizer = true;
}
