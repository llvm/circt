//===-- Passes.td - HW pass definition file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_PASSES_TD
#define CIRCT_DIALECT_HW_PASSES_TD

include "mlir/Pass/PassBase.td"

def PrintInstanceGraph : Pass<"hw-print-instance-graph", "mlir::ModuleOp"> {
  let summary = "Print a DOT graph of the module hierarchy.";
  let constructor =  "circt::hw::createPrintInstanceGraphPass()";
}

def PrintHWModuleGraph : Pass<"hw-print-module-graph", "mlir::ModuleOp"> {
  let summary = "Print a DOT graph of the HWModule's within a top-level module.";
  let constructor =  "circt::hw::createPrintHWModuleGraphPass()";
  
  let options = [
    Option<"verboseEdges", "verbose-edges", "bool", "false",
      "Print information on SSA edges (types, operand #, ...)">,
  ];
}

def FlattenIO : Pass<"hw-flatten-io", "mlir::ModuleOp"> {
  let summary = "Flattens hw::Structure typed in- and output ports.";
  let constructor =  "circt::hw::createFlattenIOPass()";
  
  let options = [
    Option<"recursive", "recursive", "bool", "false",
      "Recursively flatten nested structs.">,
  ];
}

def HWSpecialize : Pass<"hw-specialize", "mlir::ModuleOp"> {
  let summary = "Specializes instances of parametric hw.modules";
  let constructor = "circt::hw::createHWSpecializePass()";
  let description = [{
    Any `hw.instance` operation instantiating a parametric `hw.module` will
    trigger a specialization procedure which resolves all parametric types and
    values within the module based on the set of provided parameters to the
    `hw.instance` operation. This specialized module is created as a new
    `hw.module` and the referring `hw.instance` operation is rewritten to
    instantiate the newly specialized module.
  }];
}

def Externalize: Pass<"hw-externalize", "mlir::ModuleOp"> {
  let summary = "Run module externalization on the target op";
  let constructor = "circt::hw::createExternalizePass()";
  let dependentDialects = ["circt::hw::HWDialect"];
  let options = [
    Option<"opName", "op", "std::string", "",
           "Name of the operation to externalize">,
    Option<"moduleName", "module-name", "std::string", "",
           "Name of the external module">,
    Option<"instName", "instance-name", "std::string", "",
           "Name of the generated instances">,
    ListOption<"portNames", "port-names", "std::string",
           "A 'id=name' list of interface-required ports (id) to provided external module port (name)s",
           "llvm::cl::ZeroOrMore,">,
  ];
  let statistics = [
    Statistic<"numOperationsCnverted", "num-ops-converted",
      "Number of operations to external module instances">
  ];
}

#endif // CIRCT_DIALECT_HW_PASSES_TD
