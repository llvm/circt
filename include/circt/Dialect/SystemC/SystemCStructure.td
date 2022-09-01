//===- SystemCStructure.td - SystemC structure ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes ops with regions for SystemC to structure other ops, such as
// modules, methods, etc.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWOpInterfaces.td"

def SCModuleOp : SystemCOp<"module", [
    Symbol,
    HWModuleLike,
    FunctionOpInterface,
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    RegionKindInterface,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    HasParent<"mlir::ModuleOp">
  ]> {
  let summary = "Define a SystemC SC_MODULE.";
  let description = [{
    Represents the SC_MODULE macro as described in IEEE 1666-2011 §5.2.5.
    Models input, output and inout ports as module arguments (as opposed to
    `sc_signal`s which are modeled by a separate `systemc.signal` operation),
    but are nonetheless emitted as regular struct fields.
  }];

  let arguments = (ins StrArrayAttr:$portNames);
  let regions = (region SizedRegion<1>: $body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<hw::PortInfo>":$ports,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "StringAttr":$name, "const hw::ModulePortInfo &":$ports,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "StringAttr":$name, "ArrayAttr":$portNames,
                   "ArrayRef<Type>":$portTypes,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { 
      return RegionKind::Graph;
    }

    /// Returns the type of this function.
    mlir::FunctionType getFunctionType() {
      return getFunctionTypeAttr().getValue().cast<mlir::FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    // Use FunctionOpInterface traits's getBody method.
    using mlir::detail::FunctionOpInterfaceTrait<SCModuleOp>::getBody;

    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getBody().front(); }

    /// Return the symbol name of this module as string.
    StringRef getModuleName();

    using PortDirectionRange = llvm::iterator_range<
      llvm::filter_iterator<Region::BlockArgListType::iterator,
                            std::function<bool(const BlockArgument &)>>>;

    void getPortInfoList(SmallVectorImpl<hw::PortInfo> &portInfoList);
    PortDirectionRange getPortsOfDirection(hw::PortDirection direction);
    PortDirectionRange getInputPorts() {
      return getPortsOfDirection(hw::PortDirection::INPUT);
    }
    PortDirectionRange getOutputPorts() {
      return getPortsOfDirection(hw::PortDirection::OUTPUT);
    }
    PortDirectionRange getInOutPorts() {
      return getPortsOfDirection(hw::PortDirection::INOUT);
    }

    // Return the Ctor operation in this module's body or create one if none
    // exists yet.
    systemc::CtorOp getOrCreateCtor();

    // Return the Destructor operation in this module's body or create one if
    // none exists yet.
    systemc::DestructorOp getOrCreateDestructor();
  }];
}

def CtorOp : SystemCOp<"ctor", [SingleBlock, NoTerminator,
                                HasParent<"SCModuleOp">]> {
  let summary = "A constructor definition.";
  let description = [{
    Represents the SC_CTOR macro as described in IEEE 1666-2011 §5.2.7.
    The name of the module being constructed does not have to be passed
    to this operation, but is automatically added during emission.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";

  let builders = [
    OpBuilder<(ins), [{
      Region *region = $_state.addRegion();
      region->push_back(new Block);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getBody().front(); }
  }];

  let hasVerifier = true;
  let skipDefaultBuilders = 1;
}

def SCFuncOp : SystemCOp<"func", [
  HasCustomSSAName,
  SystemCNameDeclOpInterface,
  SingleBlock,
  NoTerminator,
  HasParent<"SCModuleOp">
]> {
  let summary = "A (void)->void member function of a SC_MODULE.";
  let description = [{
    This operation does not represent a specific SystemC construct, but a
    regular C++ member function with no arguments and a void return type.
    These are used to implement module-internal logic and are registered to the
    module using the SC_METHOD, SC_THREAD, and SC_CTHREAD macros.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs NullaryVoidFunctionType:$handle);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "StringAttr":$name), [{
      $_state.addAttribute(getNameAttrName($_state.name), name);
      Region *region = $_state.addRegion();
      $_state.addTypes(FunctionType::get($_builder.getContext(), {}, {}));
      region->push_back(new Block);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getBody().front(); }
  }];

  let assemblyFormat = "custom<ImplicitSSAName>($name) attr-dict-with-keyword $body";

  let hasVerifier = true;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// Operations that model C++-level features.
//===----------------------------------------------------------------------===//

def DestructorOp : SystemCOp<"cpp.destructor", [
    SingleBlock,
    NoTerminator,
    HasParent<"SCModuleOp">
]> {
  let summary = "A C++ destructor definition.";
  let description = [{
    This operation models a C++ destructor of a class or struct. It is not an
    operation modelling some abstract SystemC construct, but still required to
    support more complex functionality such as having a pointer to an external
    object inside a SystemC module, e.g., for interoperability purposes.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";

  let builders = [
    OpBuilder<(ins), [{
      Region *region = $_state.addRegion();
      region->push_back(new Block);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getBody().front(); }
  }];

  let hasVerifier = true;
  let skipDefaultBuilders = 1;
}

def InstanceDeclOp : SystemCOp<"instance.decl", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<HWInstanceLike>,
  HasCustomSSAName,
  SystemCNameDeclOpInterface,
  HasParent<"SCModuleOp">
]> {
  let summary = "Declares a SystemC module instance.";
  let description = [{
    Declares an instantiation of a SystemC module inside another SystemC module
    by value. The instance handle returned by this operation can then be used
    to initialize it in the constructor and to access its fields.
    More information can be found in IEEE 1666-2011 §4.1.1.
  }];

  let arguments = (ins StrAttr:$name, FlatSymbolRefAttr:$moduleName);
  let results = (outs ModuleType:$instanceHandle);

  let assemblyFormat = [{
    custom<ImplicitSSAName>($name) $moduleName attr-dict
    `:` type($instanceHandle)
  }];

  let extraClassDeclaration = [{
    /// Get the systemc::ModuleType that represents the instantiated module.
    systemc::ModuleType getInstanceType() {
      return getInstanceHandle().getType().cast<systemc::ModuleType>();
    }

    StringRef getInstanceName() { return getName(); }

    /// Lookup the module for the symbol. This returns null on invalid IR.
    Operation *getReferencedModule(const hw::HWSymbolCache *cache);

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// An InstanceDeclOp may not optionally define a symbol.
    bool isOptionalSymbol() { return false; }
  }];
}
