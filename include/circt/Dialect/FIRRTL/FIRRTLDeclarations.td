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

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLDECLARATIONS_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLDECLARATIONS_TD

include "FIRRTLAttributes.td"
include "FIRRTLDialect.td"
include "FIRRTLEnums.td"
include "FIRRTLOpInterfaces.td"
include "FIRRTLTypes.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class ReferableDeclOp<string mnemonic, list<Trait> traits = []> :
  FIRRTLOp<mnemonic, traits # [HasCustomSSAName, InnerSymbol, FNamableOp]> {

}

def InstanceOp : ReferableDeclOp<"instance", [HasParent<"firrtl::FModuleOp, firrtl::WhenOp">,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<HWInstanceLike, ["instanceName", "instanceNameAttr"]>]> {
  let summary = "Instantiate an instance of a module";
  let description = [{
    This represents an instance of a module.  The results are the modules inputs
    and outputs.  The inputs have flip type, the outputs do not.

    Examples:
    ```mlir
    %0 = firrtl.instance foo @Foo(in io: !firrtl.uint)
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$moduleName, StrAttr:$name, NameKindAttr:$nameKind,
                       APIntAttr:$portDirections, StrArrayAttr:$portNames,
                       AnnotationArrayAttr:$annotations,
                       PortAnnotationsAttr:$portAnnotations,
                       UnitAttr:$lowerToBind,
                       OptionalAttr<InnerSymAttr>:$inner_sym);

  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "::mlir::StringRef":$moduleName,
                   "::mlir::StringRef":$name,
                   "::circt::firrtl::NameKindEnum":$nameKind,
                   "::mlir::ArrayRef<Direction>":$portDirections,
                   "::mlir::ArrayRef<Attribute>":$portNames,
                   CArg<"ArrayRef<Attribute>", "{}">:$annotations,
                   CArg<"ArrayRef<Attribute>", "{}">:$portAnnotations,
                   CArg<"bool","false">:$lowerToBind,
                   CArg<"StringAttr", "StringAttr()">:$inner_sym)>,
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "::mlir::StringRef":$moduleName,
                   "::mlir::StringRef":$name,
                   "::circt::firrtl::NameKindEnum":$nameKind,
                   "::mlir::ArrayRef<Direction>":$portDirections,
                   "::mlir::ArrayRef<Attribute>":$portNames,
                   "ArrayRef<Attribute>":$annotations,
                   "ArrayRef<Attribute>":$portAnnotations,
                   "bool":$lowerToBind,
                   "hw::InnerSymAttr":$inner_sym)>,

    /// Constructor when you have the target module in hand.
    OpBuilder<(ins "FModuleLike":$module,
                   "mlir::StringRef":$name,
                   CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                   CArg<"ArrayRef<Attribute>", "{}">:$annotations,
                   CArg<"ArrayRef<Attribute>", "{}">:$portAnnotations,
                   CArg<"bool","false">:$lowerToBind,
                   CArg<"StringAttr", "StringAttr()">:$inner_sym)>
  ];

  let extraClassDeclaration = [{
    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    FModuleLike getReferencedModule(SymbolTable& symtbl);

    /// Return the port direction for the specified result number.
    Direction getPortDirection(size_t resultNo) {
      return direction::get(getPortDirections()[resultNo]);
    }

    /// Return the port name for the specified result number.
    StringAttr getPortName(size_t resultNo) {
      return getPortNames()[resultNo].cast<StringAttr>();
    }
    StringRef getPortNameStr(size_t resultNo) {
      return getPortName(resultNo).getValue();
    }

    /// Hooks for port annotations.
    ArrayAttr getPortAnnotation(unsigned portIdx);
    void setAllPortAnnotations(ArrayRef<Attribute> annotations);

    /// Builds a new `InstanceOp` with the ports listed in `portIndices` erased,
    /// and updates any users of the remaining ports to point at the new
    /// instance.
    InstanceOp erasePorts(OpBuilder &builder, const llvm::BitVector &portIndices);

    /// Clone the instance op and add ports.  This is usually used in
    /// conjuction with adding ports to the referenced module. This will emit
    /// the new InstanceOp to the same location.
    InstanceOp cloneAndInsertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports);
  }];
}

def MemOp : ReferableDeclOp<"mem"> {
  let summary = "Define a new mem";
  let arguments =
    (ins ConfinedAttr<I32Attr, [IntMinValue<0>]>:$readLatency,
         ConfinedAttr<I32Attr, [IntMinValue<1>]>:$writeLatency,
         ConfinedAttr<I64Attr, [IntMinValue<1>]>:$depth, RUWAttr:$ruw,
         StrArrayAttr:$portNames, StrAttr:$name, NameKindAttr:$nameKind,
         AnnotationArrayAttr:$annotations,
         PortAnnotationsAttr:$portAnnotations,
         OptionalAttr<InnerSymAttr>:$inner_sym,
         OptionalAttr<UI32Attr>:$groupID,
         OptionalAttr<MemoryInitAttr>:$init);
  let results = (outs Variadic<FIRRTLType>:$results);

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
    $ruw `` custom<MemOp>(attr-dict) `:` qualified(type($results))
  }];

  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "uint32_t":$readLatency, "uint32_t":$writeLatency,
                   "uint64_t":$depth, "RUWAttr":$ruw,
                   "ArrayRef<Attribute>":$portNames,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                   CArg<"ArrayRef<Attribute>", "{}">:$annotations,
                   CArg<"ArrayRef<Attribute>", "{}">:$portAnnotations,
                   CArg<"StringAttr", "StringAttr()">:$inner_sym)>,
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes, "uint32_t":$readLatency,
                   "uint32_t":$writeLatency, "uint64_t":$depth, "RUWAttr":$ruw,
                   "ArrayRef<Attribute>":$portNames, "StringRef":$name,
                   "NameKindEnum":$nameKind, "ArrayRef<Attribute>":$annotations,
                   "ArrayRef<Attribute>":$portAnnotations,
                   "hw::InnerSymAttr":$inner_sym)>
  ];

  let hasVerifier = 1;

  let hasCanonicalizer = true;

  let extraClassDeclaration = [{
    enum class PortKind { Read, Write, ReadWrite, Debug };

    using NamedPort = std::pair<StringAttr, MemOp::PortKind>;

    /// Return the type of a port given the memory depth, type, and kind
    static FIRRTLType getTypeForPort(uint64_t depth, FIRRTLBaseType dataType,
                                     PortKind portKind, size_t maskBits = 0);

    /// Return the name and kind of ports supported by this memory.
    SmallVector<NamedPort> getPorts();

    /// Return the kind of the specified port.
    PortKind getPortKind(StringRef portName);

    /// Return the kind of the specified port number.
    PortKind getPortKind(size_t resultNo);

    /// Return the data-type field of the memory, the type of each element.
    FIRRTLBaseType getDataType();

    /// Return the number of mask bits.
    size_t getMaskBits();

    /// Return the port name for the specified result number.
    StringAttr getPortName(size_t resultNo);
    StringRef getPortNameStr(size_t resultNo) {
      return getPortName(resultNo).getValue();
    }

    /// Return the port type for the specified result number.
    FIRRTLBaseType getPortType(size_t resultNo);

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
    void getNumPorts(size_t &numReadPorts, size_t &numWritePorts,
                      size_t &numReadWritePorts, size_t &numDbgsPorts);

    // Extract the relevant attributes from the MemOp and return a FirMemory object.
    FirMemory getSummary();
  }];
}

def NodeOp : ReferableDeclOp<"node",
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
                       NameKindAttr:$nameKind,
                       AnnotationArrayAttr:$annotations,
                       OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs FIRRTLBaseType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$input,
                  CArg<"StringRef", "{}">:$name,
                  CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                  CArg<"ArrayRef<Attribute>", "{}">:$annotations,
                  CArg<"StringAttr", "StringAttr()">:$inner_sym), [{
      return build($_builder, $_state, elementType, input, name, nameKind,
                   $_builder.getArrayAttr(annotations),
                   inner_sym ? hw::InnerSymAttr::get(inner_sym) : hw::InnerSymAttr());
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$input,
                  "StringRef":$name, "NameKindEnum":$nameKind,
                  "::mlir::ArrayAttr":$annotations,
                  CArg<"StringAttr", "StringAttr()">:$inner_sym), [{
      return build($_builder, $_state, elementType, input, name, nameKind,
                   annotations,
                   inner_sym ? hw::InnerSymAttr::get(inner_sym) : hw::InnerSymAttr());
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
    $input `` custom<ImplicitSSAName>(attr-dict) `:` qualified(type($input))
  }];

  let hasCanonicalizer = true;
  let hasFolder = 1;
}

def RegOp : ReferableDeclOp<"reg"> {
  let summary = "Define a new register";
  let description = [{
    Declare a new register:

    ```
    %name = firrtl.reg %clockVal : t1
    ```
    }];

  let arguments = (
    ins ClockType:$clockVal, StrAttr:$name, NameKindAttr:$nameKind,
        AnnotationArrayAttr:$annotations,
        OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs AnyRegisterType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations,
                   CArg<"StringAttr", "StringAttr()">:$inner_sym), [{
      return build($_builder, $_state, elementType, clockVal, name,
                   nameKind, $_builder.getArrayAttr(annotations),
                   inner_sym ? hw::InnerSymAttr::get(inner_sym) : hw::InnerSymAttr());
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   "StringRef":$name, "NameKindEnum":$nameKind,
                   "::mlir::ArrayAttr":$annotation, "StringAttr":$inner_sym), [{
      return build($_builder, $_state, elementType, clockVal, name,
                   nameKind, annotation,
                   inner_sym ? hw::InnerSymAttr::get(inner_sym) : hw::InnerSymAttr());
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
    operands `` custom<ImplicitSSAName>(attr-dict) `:` qualified(type($result))
  }];
  let hasCanonicalizeMethod = true;
}

def RegResetOp : ReferableDeclOp<"regreset"> {
  let summary = "Define a new register with a reset";
  let description = [{
    Declare a new register:
    ```
      %name = firrtl.regreset %clockVal, %resetSignal, %resetValue : t1
    ```
    }];

  let arguments = (
    ins ClockType:$clockVal, AnyResetType:$resetSignal,
        AnyRegisterType:$resetValue,
        StrAttr:$name, NameKindAttr:$nameKind,
        AnnotationArrayAttr:$annotations,
        OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs AnyRegisterType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   "::mlir::Value":$resetSignal, "::mlir::Value":$resetValue,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations,
                   CArg<"StringAttr", "StringAttr()">:$inner_sym), [{
      return build($_builder, $_state, elementType, clockVal, resetSignal,
                   resetValue, name, nameKind,
                   $_builder.getArrayAttr(annotations),
                   inner_sym ? hw::InnerSymAttr::get(inner_sym) : hw::InnerSymAttr());
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   "::mlir::Value":$resetSignal, "::mlir::Value":$resetValue,
                   "StringRef":$name, "NameKindEnum":$nameKind,
                    "::mlir::ArrayAttr":$annotation, "StringAttr":$inner_sym), [{
      return build($_builder, $_state, elementType, clockVal, resetSignal,
                   resetValue, name, nameKind, annotation,
                   inner_sym ? hw::InnerSymAttr::get(inner_sym) : hw::InnerSymAttr());
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
    operands `` custom<ImplicitSSAName>(attr-dict)
    `:` qualified(type($resetSignal)) `,` qualified(type($resetValue)) `,` qualified(type($result))
  }];

  let hasCanonicalizer = true;
  let hasVerifier = 1;
}

def WireOp : ReferableDeclOp<"wire"> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = firrtl.wire : t1
    ```
    }];

  let arguments = (ins StrAttr:$name, NameKindAttr:$nameKind,
                       AnnotationArrayAttr:$annotations,
                       OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs AnyType:$result);

  let hasCanonicalizer = true;

  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType,
                      CArg<"StringRef", "{}">:$name,
                      CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                      CArg<"ArrayRef<Attribute>","{}">:$annotations,
                      CArg<"StringAttr", "StringAttr()">:$inner_sym), [{
      return build($_builder, $_state, elementType, name, nameKind,
                   $_builder.getArrayAttr(annotations),
                   inner_sym ? hw::InnerSymAttr::get(inner_sym) : hw::InnerSymAttr());
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType, "StringRef":$name,
                   "NameKindEnum":$nameKind, "::mlir::ArrayAttr":$annotations,
                   CArg<"StringAttr", "StringAttr()">:$inner_sym), [{
      return build($_builder, $_state, elementType, name, nameKind, annotations,
                   inner_sym ? hw::InnerSymAttr::get(inner_sym) : hw::InnerSymAttr());
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind) ``
    custom<ImplicitSSAName>(attr-dict) `:`
    qualified(type($result))
  }];
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLDECLARATIONS_TD
