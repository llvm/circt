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

def HWOutlineOps : Pass<"hw-outline-ops", "mlir::ModuleOp"> {
  let summary = "Outlines operations into separate HW modules.";
  let description = [{
    This pass outlines all of the selected operations into separate modules,
    uniquifying the modules. This is useful for debugging and cutting down RTL
    size for certain high level ops.

    Notes:
      - Does not support ops with regions.
      - In uniquifying the modules, dialect attributes are ignored. Only
        properties are respected.
      - Dialect attributes are, however, inherited by the module instances.
  }];

  let options = [
    ListOption<"opNames", "op-names", "std::string",
      "List of operation names to outline. If empty, this pass is a no-op.">
  ];
}

def FlattenIO : Pass<"hw-flatten-io", "mlir::ModuleOp"> {
  let summary = "Flattens hw::Structure typed in- and output ports.";
  let constructor =  "circt::hw::createFlattenIOPass()";
  
  let options = [
    Option<"recursive", "recursive", "bool", "true",
      "Recursively flatten nested structs.">,
    Option<"flattenExtern", "flatten-extern", "bool", "false",
      "Flatten the extern modules also.">,
    Option<"joinChar", "join-char", "char", "'.'",
      "Use a custom character to construct the flattened names.">,
  ];
}

def FlattenModules : Pass<"hw-flatten-modules", "mlir::ModuleOp"> {
  let summary = "Eagerly inline private modules";
  let description = [{
    This pass eagerly inlines private HW modules into their instantiation sites.
    This is necessary for verification purposes, as model checking backends do not 
    require or support the use of module hierarchy. For simulation, module hierarchies 
    degenerate into a purely cosmetic construct, at which point it is beneficial
    to fully flatten the module hierarchy to simplify further analysis and 
    optimization of state transfer arcs.
  }];
  let constructor = "circt::hw::createFlattenModulesPass()";
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

def VerifyInnerRefNamespace : Pass<"hw-verify-irn"> {
  let summary = "Verify InnerRefNamespaceLike operations, if not self-verifying.";
  let constructor =  "circt::hw::createVerifyInnerRefNamespacePass()";
}

/**
 * Tutorial Pass, doesn't do anything interesting
 */
def FooWires : Pass<"hw-foo-wires", "hw::HWModuleOp"> {
  let summary = "Change all wires' name to foo_<n>.";
  let description = [{
    Very basic pass that numbers all of the wires in a given module.
    The wires' names are then all converte to foo_<that number>.
  }];
  let constructor = "circt::hw::createFooWiresPass()";
}

#endif // CIRCT_DIALECT_HW_PASSES_TD
