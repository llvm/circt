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

def SCModuleOp : SystemCOp<"module", [
    Symbol,
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

  let arguments = (ins PortDirectionsAttr:$portDirections, StrArrayAttr:$portNames);
  let regions = (region SizedRegion<1>: $body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;

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

    void getPortsOfDirection(PortDirection direction,
                             llvm::SmallVector<Value> &values);
    void getInputs(llvm::SmallVector<Value> &inputs) {
      getPortsOfDirection(PortDirection::Input, inputs);
    }
    void getOutputs(llvm::SmallVector<Value> &outputs) {
      getPortsOfDirection(PortDirection::Output, outputs);
    }
    void getInOuts(llvm::SmallVector<Value> &inouts) {
      getPortsOfDirection(PortDirection::InOut, inouts);
    }
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
  let assemblyFormat = "$body attr-dict";
}
