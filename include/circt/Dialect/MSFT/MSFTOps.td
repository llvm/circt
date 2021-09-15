//===- MSFTOps.td - MSFT operations definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class MSFTOp<string mnemonic, list<OpTrait> traits = []> :
    Op<MSFTDialect, mnemonic, traits>;

def InstanceOp : MSFTOp<"instance", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Create an instance of a module";

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs,
                       OptionalAttr<DictionaryAttr>:$parameters);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    // Return the name of the specified result or empty string if it cannot be
    // determined.
    StringAttr getResultName(size_t i);

    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    Operation *getReferencedModule();
  }];

  /// sym keyword for optional symbol simplifies parsing
  let assemblyFormat = [{
    $instanceName $moduleName `(` $inputs `)` attr-dict
      `:` functional-type($inputs, results)
  }];
}

def OneOrNoBlocksRegion : Region<
  CPred<"::llvm::hasNItemsOrLess($_self, 1)">,
  "region with at most 1 block">;

def MSFTModuleOp : MSFTOp<"module",
      [IsolatedFromAbove, FunctionLike, Symbol, RegionKindInterface,
       SingleBlockImplicitTerminator<"::circt::hw::OutputOp">,
       HasParent<"mlir::ModuleOp">]>{
  let summary = "HW Module";
  let description = [{
    The "hw.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments = (ins
      StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
      DictionaryAttr:$parameters);
  let results = (outs);
  let regions = (region OneOrNoBlocksRegion:$body);

  // let skipDefaultBuilders = 1;
  // let builders = [
  //   OpBuilder<(ins "StringAttr":$name, "ArrayRef<ModulePortInfo>":$ports,
  //                  CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  // ];

  let extraClassDeclaration = [{
    using FunctionLike::front;
    using FunctionLike::getBody;

    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) {
      return RegionKind::Graph;
    }

    // Decode information about the input and output ports on this module.
    SmallVector<::circt::hw::ModulePortInfo> getPorts();

    // Get the module's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the module's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<MSFTModuleOp>;

    /// Returns the number of arguments, implementing OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }
    /// Returns the number of results, implementing OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  public:
  }];

  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let verifier = "return ::verify$cppClass(*this);";
}
