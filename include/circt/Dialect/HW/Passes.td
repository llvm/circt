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

def MergeIdenticalPorts : Pass<"hw-merge-identical-ports", "mlir::ModuleOp"> {
  let summary = "Merges module ports which are always driven by the same inputs";
  let constructor = "circt::hw::createMergeIdenticalPortsPass()";
  let description = [{
    This pass merges ports for any hw::HWMutableModuleLike which, at all of
    its instantiation points, are always driven by the same inputs.
    Similarly, it removes any ports of modules which are driven by the
    same values.

    By doing so, we may reduce the i/o port list of a module, and by extension
    provide more opportunities for other optimizations such as CSE.

    In case a module intentionally has identically driven ports (i.e. for
    adhering to an external interface), a module can be marked with a
    `hw.merge_identical_ports.ignore` unit attribute, making this pass
    ignore the module.

    This pass currently only optimizes input ports when _all_ instantiations
    of a module show the same identical-driven pattern. A future
    improvement/change of the pass could be to duplicate modules to allow
    for specializing modules based on their instantiation sites - a change which
    will increase the # of unique modules, but may prove useful if enough
    optimization sites are exposed.
  }];
}

#endif // CIRCT_DIALECT_HW_PASSES_TD
