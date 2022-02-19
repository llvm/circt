//===- MSFTOps.td - MSFT operations definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWTypes.td"

// Base class for the operation in this dialect.
class MSFTOp<string mnemonic, list<Trait> traits = []> :
    Op<MSFTDialect, mnemonic, traits>;

def InstanceOp : MSFTOp<"instance", [
        Symbol,
        ParentOneOf<["::circt::hw::HWModuleOp", "MSFTModuleOp"]>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Instantiate a module";

  let arguments = (ins SymbolNameAttr:$sym_name,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs,
                       OptionalAttr<ParamDeclArrayAttr>:$parameters,
                       OptionalAttr<SymbolRefAttr>:$targetDesignPartition);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    // Return the name of the specified result or empty string if it cannot be
    // determined.
    StringAttr getResultName(size_t i);

    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    Operation *getReferencedModule();

    /// Instance name is the same as the symbol name. This may change in the
    /// future.
    StringRef instanceName() {
      return sym_name();
    }
    StringAttr instanceNameAttr() {
      return sym_nameAttr();
    }
    /// Check that the operands and results match the module specified.
    LogicalResult verifySignatureMatch(const circt::hw::ModulePortInfo&);

    // Update the results.
    InstanceOp getWithNewResults(MSFTModuleOp mod,
                                 ArrayRef<unsigned> newToOldMap);
  }];

  /// sym keyword for optional symbol simplifies parsing
  let assemblyFormat = [{
    $sym_name $moduleName `(` $inputs `)` custom<ParameterList>($parameters) attr-dict
      `:` functional-type($inputs, results)
  }];
}

def OneOrNoBlocksRegion : Region<
  CPred<"::llvm::hasNItemsOrLess($_self, 1)">,
  "region with at most 1 block">;

def MSFTModuleOp : MSFTOp<"module",
      [IsolatedFromAbove, FunctionOpInterface, Symbol, RegionKindInterface,
       HasParent<"mlir::ModuleOp">,
       SingleBlockImplicitTerminator<"OutputOp">,
       OpAsmOpInterface]>{
  let summary = "MSFT HW Module";
  let description = [{
    A lot like `hw.module`, but with a few differences:
      - Can exist without a body. The body is filled in by a generator post op
      creation.
      - Provides methods for mutation.
  }];
  let arguments = (ins
      StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
      DictionaryAttr:$parameters,
      OptionalAttr<StrAttr>:$fileName);
  let results = (outs);
  let regions = (region OneOrNoBlocksRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "hw::ModulePortInfo":$ports,
                   "ArrayRef<NamedAttribute>":$params)>
  ];

  let extraClassDeclaration = [{
    using mlir::detail::FunctionOpInterfaceTrait<MSFTModuleOp>::front;
    using mlir::detail::FunctionOpInterfaceTrait<MSFTModuleOp>::getBody;

    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) {
      return RegionKind::Graph;
    }

    // Decode information about the input and output ports on this module.
    ::circt::hw::ModulePortInfo getPorts();

    // Adds input and output ports. Returns a list of new block arguments for
    // the new inputs.
    SmallVector<BlockArgument> addPorts(
      ArrayRef<std::pair<StringAttr, Type>> inputs,
      ArrayRef<std::pair<StringAttr, Value>> outputs);

    // Remove the ports at the specified indexes. Returns the new to old result
    // mapping.
    SmallVector<unsigned>
    removePorts(llvm::BitVector inputs, llvm::BitVector outputs);

    // Get the module's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the module's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    Block *getBodyBlock() { return &body().front(); }
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);

    /// Returns the type of this function.
    FunctionType getType() { 
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def MSFTModuleExternOp : MSFTOp<"module.extern",
      [Symbol, HasParent<"mlir::ModuleOp">]> {
  let summary = "MSFT external Module";
  let description = [{
    Identical to `hw.module.extern`, and trivially lowers to that. This op
    exists so that we can use `msft.instance` to refer to both `msft.module` and
    `msft.module.extern`, rather than mixing `hw.instance` with `msft.instance`.
  }];
  let arguments = (ins StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
                       ParamDeclArrayAttr:$parameters,
                       OptionalAttr<StrAttr>:$verilogName);
  let regions = (region SizedRegion<0>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "const hw::ModulePortInfo &":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let hasCustomAssemblyFormat = 1;
  let verifier = "return ::verify$cppClass(*this);";

  let extraClassDeclaration = [{
    /// Decode information about the input and output ports on this module.
    hw::ModulePortInfo getPorts();
  }];
}

def DesignPartitionOp : MSFTOp<"partition",
                               [Symbol, HasParent<"msft::MSFTModuleOp">]> {
  let summary = "A target \"module\" for moving entities";
  let description = [{
    Sometimes EDA tools require designs to have a module hierarchy which doesn't
    match the logical structure a designer would like to have. "Design
    partitions" allow the designer to "tag" entities (instances, registers, etc.)
    with a target design partition. During lowering, CIRCT will modify the
    hierarchy to move the tagged entities into the design partition module. The
    target design partition can then be used by subsequent EDA tools.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$verilogName);
  let assemblyFormat = "$sym_name `,` $verilogName attr-dict";
}

def OutputOp : MSFTOp<"output", [Terminator, HasParent<"MSFTModuleOp">,
                                NoSideEffect, ReturnLike]> {
  let summary = "termination operation";

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [
    OpBuilder<(ins)>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` qualified(type($operands)))?";
}

def DeclPhysicalRegionOp : MSFTOp<"physical_region",
    [Symbol, HasParent<"mlir::ModuleOp">]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    PhysicalBoundsArray:$bounds
  );

  let assemblyFormat = [{
    $sym_name `,` $bounds attr-dict
  }];
}

def EntityExternOp : MSFTOp<"entity.extern",
    [Symbol, HasParent<"mlir::ModuleOp">]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    AnyAttr:$metadata);
  let assemblyFormat = [{
    $sym_name $metadata attr-dict
  }];
}

def PDPhysLocationOp : MSFTOp<"pd.location",
      [DeclareOpInterfaceMethods<DynInstDataOpInterface>]> {
  let summary = "Specify a location for an instance";
  let description = [{
    Used to specify a specific location on an FPGA to place a dynamic instance.
    Supports specifying the location of a subpath for extern modules and device
    primitives. Intended to live as a child of `instance.dynamic` initially
    without the `ref` field. The dynamic instance lowering will fill in `ref`
    with the symol of the `hw.globalref` op corresponding to the lowered dynamic
    instance.
  }];
  let arguments = (ins PhysLocation:$loc,
                       OptionalAttr<StrAttr>:$subPath,
                       OptionalAttr<FlatSymbolRefAttr>:$ref);
  let assemblyFormat = [{
    ($ref^)? custom<PhysLoc>($loc) (`path` `:` $subPath^)? attr-dict
  }];
}

def PDPhysRegionOp : MSFTOp<"pd.physregion", 
      [DeclareOpInterfaceMethods<DynInstDataOpInterface>]> {
  let summary = "Specify a physical region for an instance";
  let description = [{
    Annotate a particular entity within an op with the region of the devices
    on an FPGA to which it should mapped. The physRegionRef must refer to a
    DeclPhysicalRegion operation.
  }];
  let arguments = (ins FlatSymbolRefAttr:$physRegionRef,
                       OptionalAttr<StrAttr>:$subPath,
                       OptionalAttr<FlatSymbolRefAttr>:$ref);
  let assemblyFormat = [{
    ($ref^)? $physRegionRef (`path` `:` $subPath^)? attr-dict
  }];
}

def DynamicInstanceOp : MSFTOp<"instance.dynamic",
                               [HasParent<"mlir::ModuleOp">, NoTerminator]> {

  let summary = "A module instance in the instance hierarchy";
  let description = [{
    Represents an instance (as in instance in the instance hierarchy) referred
    to henceforth as a dynamic instance. Specified with a path through the
    instance hierarchy (which in the future will be replaced with an AppID).
    Lowers to a `hw.globalref` but unlike a global ref, does not require all of
    the ops participating in the globalref to contain a back pointer attribute.
    Allows users to efficiently add placements to a large number of dynamic
    instances which happen to map to a small number of static instances by
    bulk-adding the necessary `hw.globalref` attributes.

    During the lowering, moves the operations in the body to the top level and
    gives them the symbol of the globalref which was created to replace the
    dynamic instance.
  }];
  let arguments = (ins NameRefArrayAttr:$appid);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $appid $body attr-dict
  }];

  let extraClassDeclaration = [{
    bool isRootedIn(Operation *top);
  }];
}
