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
}

def PrintHWModuleGraph : Pass<"hw-print-module-graph", "mlir::ModuleOp"> {
  let summary = "Print a DOT graph of the HWModule's within a top-level module.";
  let options = [
    Option<"verboseEdges", "verbose-edges", "bool", "false",
      "Print information on SSA edges (types, operand #, ...)">,
  ];
}

def FlattenIO : Pass<"hw-flatten-io", "mlir::ModuleOp"> {
  let summary = "Flattens hw::Structure typed in- and output ports.";
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
}

def HWSpecialize : Pass<"hw-specialize", "mlir::ModuleOp"> {
  let summary = "Specializes instances of parametric hw.modules";
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
}

def HWAggregateToComb : Pass<"hw-aggregate-to-comb", "hw::HWModuleOp"> {
  let summary = "Lower aggregate operations to comb operations";
  let description = [{
    This pass lowers aggregate *operations* to comb operations within modules.
    Note that this pass does not lower ports. Ports lowering is handled
    by FlattenIO.

    This pass will change the behavior of out-of-bounds access of arrays,
    specifically the last element of the array is used as a value for
    out-of-bounds access.
  }];
  let dependentDialects = ["comb::CombDialect"];
}

def HWInliner : Pass<"hw-inliner", "mlir::ModuleOp"> {
  let summary = "Inline private HW modules";
  let description = [{
    This pass aggressively inlines private HW modules into their instantiation
    sites based on heuristics. The pass will inline modules that are:
    - Empty (only contain hw.output)
    - Have no outputs
    - Have only one use
    - Are small (less than 8 operations)
    - Do not contain state (seq.firreg operations)

    The pass operates in post-order on the instance graph to ensure that
    nested modules are processed before their parents. It also handles
    hierarchical path operations and inner symbol renaming.
  }];
  let statistics = [
    Statistic<"numInlined", "modules-inlined", "Number of modules that were inlined">
  ];
  let options = [
    Option<"inlineEmpty", "inline-empty", "bool", "true",
           "Inline modules that are empty (only contain hw.output)">,
    Option<"inlineNoOutputs", "inline-no-outputs", "bool", "true",
           "Inline modules that have no output ports">,
    Option<"inlineSingleUse", "inline-single-use", "bool", "true",
           "Inline modules that have only one use">,
    Option<"inlineSmall", "inline-small", "bool", "true",
           "Inline modules that are small (fewer than smallThreshold operations)">,
    Option<"smallThreshold", "small-threshold", "unsigned", "8",
           "Maximum number of operations for a module to be considered small">,
    Option<"inlineWithState", "inline-with-state", "bool", "false",
           "Allow inlining of modules that contain state (seq.firreg operations)">
  ];
}

#endif // CIRCT_DIALECT_HW_PASSES_TD
