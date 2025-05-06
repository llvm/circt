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
include "mlir/Interfaces/SideEffectInterfaces.td"

/// A declaration that can be referred to by it's name/identity.
class ReferableDeclOp<string mnemonic, list<Trait> traits = []> :
  FIRRTLOp<mnemonic, traits # [HasCustomSSAName,
                               DeclareOpInterfaceMethods<InnerSymbol,["getTargetResultIndex"]>,
                               FNamableOp]> {
}

/// A referable declaration of some actual hardware object, which must be
/// located in a hardware-creating context, such as the body of a module.
class HardwareDeclOp<string mnemonic, list <Trait> traits = []> :
  ReferableDeclOp<mnemonic, traits # [
    ParentOneOf<[
      "firrtl::ContractOp",
      "firrtl::FModuleOp",
      "firrtl::LayerBlockOp",
      "firrtl::MatchOp",
      "firrtl::WhenOp",
      "sv::IfDefOp",
    ]>
  ]> {}

def InstanceOp : HardwareDeclOp<"instance", [
    DeclareOpInterfaceMethods<FInstanceLike>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<InstanceGraphInstanceOpInterface, [
      "getModuleName"
    ]>,
  ]> {
  let summary = "Instantiate a module";
  let description = [{
    This represents an instance of a module.  The results are the modules inputs
    and outputs.  The inputs have flip type, the outputs do not.

    Examples:
    ```mlir
    %0 = firrtl.instance foo @Foo(in io: !firrtl.uint)
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$moduleName, StrAttr:$name,
      NameKindAttr:$nameKind, DenseBoolArrayAttr:$portDirections,
      StrArrayAttr:$portNames, AnnotationArrayAttr:$annotations,
      PortAnnotationsAttr:$portAnnotations, LayerArrayAttr:$layers,
      UnitAttr:$lowerToBind, UnitAttr:$doNotPrint,
      OptionalAttr<InnerSymAttr>:$inner_sym);

  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;

  let builders =
      [OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
           "::mlir::StringRef":$moduleName, "::mlir::StringRef":$name,
           "::circt::firrtl::NameKindEnum":$nameKind,
           "::mlir::ArrayRef<Direction>":$portDirections,
           "::mlir::ArrayRef<Attribute>":$portNames,
           CArg<"ArrayRef<Attribute>", "{}">:$annotations,
           CArg<"ArrayRef<Attribute>", "{}">:$portAnnotations,
           CArg<"::mlir::ArrayRef<Attribute>", "{}">:$layers,
           CArg<"bool", "false">:$lowerToBind,
           CArg<"bool", "false">:$doNotPrint,
           CArg<"StringAttr", "StringAttr()">:$innerSym)>,
       OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
           "::mlir::StringRef":$moduleName, "::mlir::StringRef":$name,
           "::circt::firrtl::NameKindEnum":$nameKind,
           "::mlir::ArrayRef<Direction>":$portDirections,
           "::mlir::ArrayRef<Attribute>":$portNames,
           "ArrayRef<Attribute>":$annotations,
           "ArrayRef<Attribute>":$portAnnotations,
           "::mlir::ArrayRef<Attribute>":$layers, "bool":$lowerToBind,
           "bool":$doNotPrint, "hw::InnerSymAttr":$innerSym)>,

       /// Constructor when you have the target module in hand.
       OpBuilder<(ins "FModuleLike":$module, "mlir::StringRef":$name,
           CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
           CArg<"ArrayRef<Attribute>", "{}">:$annotations,
           CArg<"ArrayRef<Attribute>", "{}">:$portAnnotations,
           CArg<"bool", "false">:$lowerToBind,
           CArg<"bool", "false">:$doNotPrint,
           CArg<"hw::InnerSymAttr", "hw::InnerSymAttr()">:$innerSym)>,

       /// Constructor when you have a port info list in hand.
       OpBuilder<(ins "ArrayRef<PortInfo>":$ports,
           "::mlir::StringRef":$moduleName, "mlir::StringRef":$name,
           CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
           CArg<"ArrayRef<Attribute>", "{}">:$annotations,
           CArg<"ArrayRef<Attribute>", "{}">:$layers,
           CArg<"bool", "false">:$lowerToBind,
           CArg<"bool", "false">:$doNotPrint,
           CArg<"hw::InnerSymAttr", "hw::InnerSymAttr()">:$innerSym)>];

  let extraClassDeclaration = [{
    /// Return the port direction for the specified result number.
    Direction getPortDirection(size_t resultNo) {
      return direction::get(getPortDirections()[resultNo]);
    }

    /// Return the port name for the specified result number.
    StringAttr getPortName(size_t resultNo) {
      return cast<StringAttr>(getPortNames()[resultNo]);
    }
    StringRef getPortNameStr(size_t resultNo) {
      return getPortName(resultNo).getValue();
    }

    /// Hooks for port annotations.
    ArrayAttr getPortAnnotation(unsigned portIdx);
    void setAllPortAnnotations(ArrayRef<Attribute> annotations);

    /// Builds a new instance with the ports listed in `portIndices` erased, and
    /// updates any users of the remaining ports to point at the new instance.
    InstanceOp erasePorts(OpBuilder &builder, const llvm::BitVector &portIndices);

    /// Clone the instance op and add ports.  This is usually used in
    /// conjuction with adding ports to the referenced module. This will emit
    /// the new InstanceOp to the same location.
    InstanceOp cloneAndInsertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports);

    //===------------------------------------------------------------------===//
    // Instance graph methods
    //===------------------------------------------------------------------===//

    // Quick lookup of the referenced module using the instance graph.
    template <typename T = ::circt::igraph::ModuleOpInterface>
    T getReferencedModule(::circt::igraph::InstanceGraph &instanceGraph) {
      auto moduleNameAttr = getModuleNameAttr().getAttr();
      auto *node =  instanceGraph.lookup(moduleNameAttr);
      if (!node)
        return nullptr;
      Operation *moduleOp = node->getModule();
      return dyn_cast_or_null<T>(moduleOp);
    }
  }];

  let hasVerifier = true;
}

def InstanceChoiceOp : HardwareDeclOp<"instance_choice", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Creates an instance of a module based on a option";

  let description = [{
    The instance choice operation creates an instance choosing the target based
    on the value of an option if one is specified, instantiating a default
    target otherwise.

    The port lists of all instance targets must match exactly.

    Examples:
    ```mlir
    %0 = firrtl.instance_choice foo @Foo alternatives @Opt { @FPGA -> @FPGAFoo }
      (in io: !firrtl.uint)
    ```
  }];

  let arguments = (ins FlatSymbolRefArrayAttr:$moduleNames,
                       SymbolRefArrayAttr:$caseNames,
                       StrAttr:$name, NameKindAttr:$nameKind,
                       DenseBoolArrayAttr:$portDirections, StrArrayAttr:$portNames,
                       AnnotationArrayAttr:$annotations,
                       PortAnnotationsAttr:$portAnnotations,
                       LayerArrayAttr:$layers,
                       OptionalAttr<InnerSymAttr>:$inner_sym);

  let results = (outs Variadic<AnyType>:$results);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Return the port direction for the specified result number.
    Direction getPortDirection(size_t resultNo) {
      return direction::get(getPortDirections()[resultNo]);
    }

    /// Return the port name for the specified result number.
    StringAttr getPortName(size_t resultNo) {
      return cast<StringAttr>(getPortNames()[resultNo]);
    }

    /// Builds a new instance with the ports listed in `portIndices` erased, and
    /// updates any users of the remaining ports to point at the new instance.
    InstanceChoiceOp erasePorts(OpBuilder &builder,
                                const llvm::BitVector &portIndices);

    /// Return the default target attribute.
    FlatSymbolRefAttr getDefaultTargetAttr() {
      return llvm::cast<FlatSymbolRefAttr>(getModuleNamesAttr()[0]);
    }

    /// Return the name of the option group.
    StringAttr getOptionNameAttr() {
      return llvm::cast<SymbolRefAttr>(getCaseNames()[0]).getRootReference();
    }

    /// Return the target for an option or the default if none specified.
    FlatSymbolRefAttr getTargetOrDefaultAttr(OptionCaseOp option);

    /// Return the list of case-module mappings.
    SmallVector<std::pair<SymbolRefAttr, FlatSymbolRefAttr>, 1>
    getTargetChoices();
  }];

  let builders = [
    OpBuilder<(ins "FModuleLike":$defaultModule,
                   "ArrayRef<std::pair<OptionCaseOp, FModuleLike>>":$cases,
                   "mlir::StringRef":$name,
                   CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                   CArg<"ArrayRef<Attribute>", "{}">:$annotations,
                   CArg<"ArrayRef<Attribute>", "{}">:$portAnnotations,
                   CArg<"StringAttr", "StringAttr()">:$innerSym)>
  ];
}


def MemOp : HardwareDeclOp<"mem", [DeclareOpInterfaceMethods<CombDataflow>]> {
  let summary = "Define a new mem";
  let arguments =
    (ins ConfinedAttr<I32Attr, [IntMinValue<0>]>:$readLatency,
         ConfinedAttr<I32Attr, [IntMinValue<1>]>:$writeLatency,
         ConfinedAttr<I64Attr, [IntMinValue<1>]>:$depth, RUWAttr:$ruw,
         StrArrayAttr:$portNames, StrAttr:$name, NameKindAttr:$nameKind,
         AnnotationArrayAttr:$annotations,
         PortAnnotationsAttr:$portAnnotations,
         OptionalAttr<InnerSymAttr>:$inner_sym,
         OptionalAttr<MemoryInitAttr>:$init,
         OptionalAttr<StrAttr>:$prefix);
  let results = (outs Variadic<FIRRTLType>:$results);

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
    $ruw `` custom<MemOp>(attr-dict) `:` qualified(type($results))
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = true;

  let extraClassDeclaration = [{
    enum class PortKind { Read, Write, ReadWrite, Debug };

    using NamedPort = std::pair<StringAttr, MemOp::PortKind>;

    /// Return the address width of the memory.
    size_t getAddrBits();

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

def NodeOp : HardwareDeclOp<"node", [
      AllTypesMatch<["input","result"]>,
      DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
      Forceable]> {
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
                       OptionalAttr<InnerSymAttr>:$inner_sym,
                       UnitAttr:$forceable); // ReferenceKinds
  let results = (outs PassiveType:$result, Optional<RefType>:$ref);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                   CArg<"ArrayRef<Attribute>", "{}">:$annotations,
                   CArg<"StringAttr", "StringAttr()">:$innerSym,
                   CArg<"bool", "false">:$forceable), [{
      return build($_builder, $_state, input, name, nameKind,
                   $_builder.getArrayAttr(annotations),
                   innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(),
                   forceable);
    }]>,
    OpBuilder<(ins "::mlir::Value":$input,
                   "StringRef":$name, "NameKindEnum":$nameKind,
                   "::mlir::ArrayAttr":$annotations,
                   CArg<"StringAttr", "StringAttr()">:$innerSym,
                   CArg<"bool", "false">:$forceable), [{
      return build($_builder, $_state, input, name, nameKind,
                   annotations,
                   innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(),
                   forceable);
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
    $input (`forceable` $forceable^)? `` custom<FIRRTLImplicitSSAName>(attr-dict) `:` qualified(type($input))
  }];

  let hasCanonicalizer = true;
  let hasFolder = 1;
}

def RegOp : HardwareDeclOp<"reg", [Forceable, DeclareOpInterfaceMethods<CombDataflow>]> {
  let summary = "Define a new register";
  let description = [{
    Declare a new register:

    ```
    %name = firrtl.reg %clockVal : !firrtl.clock, t1
    ```
    }];

  let arguments = (
    ins ClockType:$clockVal, StrAttr:$name, NameKindAttr:$nameKind,
        AnnotationArrayAttr:$annotations,
        OptionalAttr<InnerSymAttr>:$inner_sym,
        UnitAttr:$forceable);
  let results = (outs AnyRegisterType:$result, Optional<RefType>:$ref);

  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes, "::mlir::ValueRange":$operands,
                   CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>","{}">:$attributes), [{
      assert(operands.size() == 1u && "mismatched number of parameters");
      odsState.addOperands(operands);
      odsState.addAttributes(attributes);
      assert(resultTypes.size() >= 1u && "mismatched number of return types");
      odsState.addTypes(resultTypes);
    }]>,
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes, "::mlir::ValueRange":$operands,
                   CArg<"const Properties &","{}">:$properties), [{
      assert(operands.size() == 1u && "mismatched number of parameters");
      odsState.addOperands(operands);
      odsState.useProperties(const_cast<Properties &>(properties));
      assert(resultTypes.size() >= 1u && "mismatched number of return types");
      odsState.addTypes(resultTypes);
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType,
                   "::mlir::Value":$clockVal,
                   "::mlir::StringAttr":$name,
                   "::circt::firrtl::NameKindEnumAttr":$nameKind,
                   "::mlir::ArrayAttr":$annotations,
                   "::circt::hw::InnerSymAttr":$innerSym,
                   "::mlir::UnitAttr":$forceable), [{
      $_state.addOperands(clockVal);
      $_state.addAttribute(getNameAttrName($_state.name), name);
      $_state.addAttribute(getNameKindAttrName($_state.name), nameKind);
      $_state.addAttribute(getAnnotationsAttrName($_state.name), annotations);
      if (innerSym) {
        $_state.addAttribute(getInnerSymAttrName($_state.name), innerSym);
      }
      $_state.addTypes(elementType);
      if (forceable) {
        $_state.addAttribute(getForceableAttrName($_state.name), forceable);
        if (auto type = detail::getForceableResultType(true, elementType))
          $_state.addTypes(type);
      }
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType,
                   "::mlir::Value":$clockVal,
                   "::llvm::StringRef":$name,
                   "::circt::firrtl::NameKindEnum":$nameKind,
                   "::mlir::ArrayAttr":$annotations,
                   "::circt::hw::InnerSymAttr":$innerSym,
                   "bool":$forceable), [{
      return build($_builder, $_state, elementType, clockVal,
                   $_builder.getStringAttr(name),
                   ::circt::firrtl::NameKindEnumAttr::get(odsBuilder.getContext(), nameKind),
                   annotations, innerSym,
                   forceable ? $_builder.getUnitAttr() : nullptr);
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations,
                   CArg<"StringAttr", "StringAttr()">:$innerSym,
                   CArg<"bool", "false">:$forceable), [{
      return build($_builder, $_state, elementType,
                   clockVal, $_builder.getStringAttr(name), nameKind,
                   $_builder.getArrayAttr(annotations),
                   innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(),
                   forceable);
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   "StringRef":$name, "NameKindEnum":$nameKind,
                   "::mlir::ArrayAttr":$annotation, "StringAttr":$innerSym,
                   CArg<"bool", "false">:$forceable), [{
      return build($_builder, $_state, elementType,
                   clockVal, $_builder.getStringAttr(name), nameKind, annotation,
                   innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(),
                   forceable);
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
    operands (`forceable` $forceable^)? `` custom<FIRRTLImplicitSSAName>(attr-dict) `:` type($clockVal) `,` qualified(type($result)) (`,` qualified(type($ref))^)?

  }];
  let hasCanonicalizeMethod = true;
}

def RegResetOp : HardwareDeclOp<"regreset", [Forceable, DeclareOpInterfaceMethods<CombDataflow>]> {
  let summary = "Define a new register with a reset";
  let description = [{
    Declare a new register:
    ```
      %name = firrtl.regreset %clockVal, %resetSignal, %resetValue : !firrtl.clock, t1
    ```
    }];

  let arguments = (
    ins ClockType:$clockVal, AnyResetType:$resetSignal,
        FIRRTLBaseType:$resetValue,
        StrAttr:$name, NameKindAttr:$nameKind,
        AnnotationArrayAttr:$annotations,
        OptionalAttr<InnerSymAttr>:$inner_sym,
        UnitAttr:$forceable);
  let results = (outs AnyRegisterType:$result, Optional<RefType>:$ref);

  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes, "::mlir::ValueRange":$operands,
                   CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>","{}">:$attributes), [{
      assert(operands.size() == 3u && "mismatched number of parameters");
      odsState.addOperands(operands);
      odsState.addAttributes(attributes);
      assert(resultTypes.size() >= 1u && "mismatched number of return types");
      odsState.addTypes(resultTypes);
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType,
                   "::mlir::Value":$clockVal,
                   "::mlir::Value":$resetSignal, "::mlir::Value":$resetValue,
                   "::mlir::StringAttr":$name,
                   "::circt::firrtl::NameKindEnumAttr":$nameKind,
                   "::mlir::ArrayAttr":$annotations,
                   "::circt::hw::InnerSymAttr":$innerSym,
                   "::mlir::UnitAttr":$forceable), [{
      $_state.addOperands(clockVal);
      odsState.addOperands(resetSignal);
      odsState.addOperands(resetValue);
      $_state.addAttribute(getNameAttrName($_state.name), name);
      $_state.addAttribute(getNameKindAttrName($_state.name), nameKind);
      $_state.addAttribute(getAnnotationsAttrName($_state.name), annotations);
      if (innerSym) {
        $_state.addAttribute(getInnerSymAttrName($_state.name), innerSym);
      }
      $_state.addTypes(elementType);
      if (forceable) {
        $_state.addAttribute(getForceableAttrName($_state.name), forceable);
        if (auto type = detail::getForceableResultType(true, elementType))
          $_state.addTypes(type);
      }
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   "::mlir::Value":$resetSignal, "::mlir::Value":$resetValue,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations,
                   CArg<"StringAttr", "StringAttr()">:$innerSym,
                   CArg<"bool", "false">:$forceable), [{
      return build($_builder, $_state, elementType,
                   clockVal, resetSignal, resetValue,
                   $_builder.getStringAttr(name),
                   ::circt::firrtl::NameKindEnumAttr::get(odsBuilder.getContext(), nameKind),
                   $_builder.getArrayAttr(annotations),
                   innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(),
                   forceable ? $_builder.getUnitAttr() : nullptr);
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType, "::mlir::Value":$clockVal,
                   "::mlir::Value":$resetSignal, "::mlir::Value":$resetValue,
                   "StringRef":$name, "NameKindEnum":$nameKind,
                    "::mlir::ArrayAttr":$annotation, "StringAttr":$innerSym,
                    CArg<"bool", "false">:$forceable), [{
      return build($_builder, $_state, elementType,
                   clockVal, resetSignal, resetValue,
                   $_builder.getStringAttr(name),
                   ::circt::firrtl::NameKindEnumAttr::get(odsBuilder.getContext(), nameKind),
                   annotation,
                   innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(),
                   forceable ? $_builder.getUnitAttr() : nullptr);
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
    operands (`forceable` $forceable^)? `` custom<FIRRTLImplicitSSAName>(attr-dict)
    `:` type($clockVal) `,` qualified(type($resetSignal)) `,` qualified(type($resetValue)) `,` qualified(type($result)) (`,` qualified(type($ref))^)?

  }];

  let hasCanonicalizer = true;
  let hasVerifier = 1;
}

def WireOp : HardwareDeclOp<"wire", [
    Forceable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = firrtl.wire : t1
    ```
    }];

  let arguments = (ins StrAttr:$name, NameKindAttr:$nameKind,
                       AnnotationArrayAttr:$annotations,
                       OptionalAttr<InnerSymAttr>:$inner_sym,
                       UnitAttr:$forceable); // ReferenceKinds
  let results = (outs AnyType:$result, Optional<RefType>:$ref);

  let hasCanonicalizer = true;
  let skipDefaultBuilders = true;

  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes, "::mlir::ValueRange":$operands,
                   CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>","{}">:$attributes), [{
      assert(operands.size() == 0u && "mismatched number of parameters");
      odsState.addOperands(operands);
      odsState.addAttributes(attributes);
      assert(resultTypes.size() >= 1u && "mismatched number of return types");
      odsState.addTypes(resultTypes);
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType,
                   "::mlir::StringAttr":$name,
                   "::circt::firrtl::NameKindEnumAttr":$nameKind,
                   "::mlir::ArrayAttr":$annotations,
                   "::circt::hw::InnerSymAttr":$innerSym,
                   "::mlir::UnitAttr":$forceable), [{
      $_state.addAttribute(getNameAttrName($_state.name), name);
      $_state.addAttribute(getNameKindAttrName($_state.name), nameKind);
      $_state.addAttribute(getAnnotationsAttrName($_state.name), annotations);
      if (innerSym) {
        $_state.addAttribute(getInnerSymAttrName($_state.name), innerSym);
      }
      $_state.addTypes(elementType);
      if (forceable) {
        $_state.addAttribute(getForceableAttrName($_state.name), forceable);
        if (auto type = detail::getForceableResultType(true, elementType))
          $_state.addTypes(type);
      }
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType,
                   "::llvm::StringRef":$name,
                   "::circt::firrtl::NameKindEnum":$nameKind,
                   "::mlir::ArrayAttr":$annotations,
                   "::circt::hw::InnerSymAttr":$innerSym,
                   "bool":$forceable), [{
      return build($_builder, $_state, elementType,
                   $_builder.getStringAttr(name),
                   ::circt::firrtl::NameKindEnumAttr::get(odsBuilder.getContext(), nameKind),
                   annotations, innerSym,
                   forceable ? $_builder.getUnitAttr() : nullptr);
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType,
                      CArg<"StringRef", "{}">:$name,
                      CArg<"NameKindEnum", "NameKindEnum::DroppableName">:$nameKind,
                      CArg<"ArrayRef<Attribute>","{}">:$annotations,
                      CArg<"StringAttr", "StringAttr()">:$innerSym,
                      CArg<"bool", "false">:$forceable), [{
      return build($_builder, $_state, elementType,
                   name, nameKind,
                   $_builder.getArrayAttr(annotations),
                   innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(),
                   forceable);
    }]>,
    OpBuilder<(ins "::mlir::Type":$elementType, "StringRef":$name,
                   "NameKindEnum":$nameKind, "::mlir::ArrayAttr":$annotations,
                   CArg<"StringAttr", "StringAttr()">:$innerSym,
                   CArg<"bool", "false">:$forceable), [{
      return build($_builder, $_state, elementType,
                   name, nameKind, annotations,
                   innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(),
                   forceable);
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
    (`forceable` $forceable^)? `` custom<FIRRTLImplicitSSAName>(attr-dict) `:`
    qualified(type($result)) (`,` qualified(type($ref))^)?
  }];
}

def ContractOp : FIRRTLOp<"contract", [
  SingleBlock,
  NoTerminator,
  AllTypesMatch<["inputs", "outputs"]>,
]> {
  let summary = "Contract declaration";
  let description = [{
    The `firrtl.contract` operation defines values that adhere to a set of
    formal assertions and assumptions outlined in the contract's body.

    See the `verif.contract` operation for more details.
  }];
  let arguments = (ins Variadic<PassiveType>:$inputs);
  let results = (outs Variadic<PassiveType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    ($inputs^ `:` type($inputs))? attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Property Ops
//===----------------------------------------------------------------------===//

def ObjectOp :  FIRRTLOp<"object", [
    ParentOneOf<["firrtl::FModuleOp, firrtl::ClassOp"]>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<FInstanceLike, [
      "getReferencedOperation",
      "getReferencedModuleName",
      "getReferencedModuleNameAttr",
    ]>,
    DeclareOpInterfaceMethods<InstanceGraphInstanceOpInterface, [
      "getReferencedModuleSlow"
    ]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, [
      "getAsmResultNames"
    ]>
]> {
  let summary = "Instantiate a class to produce an object";
  let description = [{
    This represents an instance of a class.  The results is the instantiated
    object.

    Examples:
    ```mlir
    %0 = firrtl.object @Foo(in io: !firrtl.uint)
    ```
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs ClassType:$result);
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "ClassLike":$klass, "StringRef":$name)>
  ];
  let extraClassDeclaration = [{
    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    ClassLike getReferencedClass(const SymbolTable &symbolTable);

    StringAttr getClassNameAttr();
    StringRef getClassName();
  }];
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>(attr-dict) custom<ClassInterface>(type($result))
  }];
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLDECLARATIONS_TD
