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
        Symbol,
        ParentOneOf<["::circt::hw::HWModuleOp", "MSFTModuleOp"]>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Instantiate a module";

  let arguments = (ins SymbolNameAttr:$sym_name,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs);
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
  }];

  /// sym keyword for optional symbol simplifies parsing
  let assemblyFormat = [{
    $sym_name $moduleName `(` $inputs `)` attr-dict
      `:` functional-type($inputs, results)
  }];
}

def OneOrNoBlocksRegion : Region<
  CPred<"::llvm::hasNItemsOrLess($_self, 1)">,
  "region with at most 1 block">;

def MSFTModuleOp : MSFTOp<"module",
      [IsolatedFromAbove, FunctionLike, Symbol, RegionKindInterface,
       HasParent<"mlir::ModuleOp">]>{
  let summary = "MSFT HW Module";
  let description = [{
    A lot like `hw.module`, but with a few differences:
      - Can exist without a body. The body is filled in by a generator post op
      creation.
      - MSFT-specific methods and arguments will be added later on.
  }];
  let arguments = (ins
      StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
      DictionaryAttr:$parameters);
  let results = (outs);
  let regions = (region OneOrNoBlocksRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<hw::PortInfo>":$ports,
                   "ArrayRef<NamedAttribute>":$params)>
  ];

  let extraClassDeclaration = [{
    using FunctionLike::front;
    using FunctionLike::getBody;

    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) {
      return RegionKind::Graph;
    }

    // Decode information about the input and output ports on this module.
    ::circt::hw::ModulePortInfo getPorts();

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
}


def OutputOp : MSFTOp<"output", [Terminator, HasParent<"MSFTModuleOp">,
                                NoSideEffect, ReturnLike]> {
  let summary = "termination operation";

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
