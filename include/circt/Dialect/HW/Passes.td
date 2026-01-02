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

    By default, all private modules are inlined. The pass supports heuristics to
    control which modules are inlined based on their characteristics.
  }];
  let options = [
    Option<"inlineEmpty", "hw-inline-empty", "bool", "true",
           "Inline modules that are empty (only contain hw.output)">,
    Option<"inlineNoOutputs", "hw-inline-no-outputs", "bool", "true",
           "Inline modules that have no output ports">,
    Option<"inlineSingleUse", "hw-inline-single-use", "bool", "true",
           "Inline modules that have only one use">,
    Option<"inlineSmall", "hw-inline-small", "bool", "true",
           "Inline modules that are small (fewer than smallThreshold operations)">,
    Option<"smallThreshold", "hw-small-threshold", "unsigned", "8",
           "Maximum number of operations for a module to be considered small">,
    Option<"inlineWithState", "hw-inline-with-state", "bool", "false",
           "Allow inlining of modules that contain state (seq.firreg operations)">,
    Option<"inlineAll", "hw-inline-all", "bool", "true",
           "Inline all private modules regardless of heuristics (default behavior)">
  ];
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

def HWParameterizeConstantPorts : Pass<"hw-parameterize-constant-ports",
                                       "mlir::ModuleOp"> {
  let summary = "Parametize constant ports on private modules";
  let description = [{
    This pass converts input ports on private modules into parameters when all
    instances pass constant values (or parameter values) to those ports.

    By converting constant ports to parameters, synthesis pipelines can
    recognize these values as compile-time constants through local analysis
    alone, without requiring inter-module analysis. This enables more precise
    timing information and better optimization of each instance independently,
    since the constant values are immediately visible at the module interface.
  }];
}

def HWBypassInnerSymbols : Pass<"hw-bypass-inner-symbols", "hw::HWModuleOp"> {
  let summary = "Pass through values through inner symbols";
  let description = [{
    This pass moves inner symbols from ports to wires, then bypasses wire
    operations with inner symbols by replacing uses with their inputs while
    keeping the wire to preserve the symbol. This enables optimizations to
    cross symbol boundaries while maintaining symbol references.

    Warning: This transformation assumes that values associated with inner
    symbols are not mutated through inner symbols (e.g. force). This assumption
    may not hold in simulation, but is safe in synthesis. This pass treats
    inner symbols differently from the optimization-blocking semantics that
    other parts of CIRCT use, so it is opt-in and should only be used when
    the above assumptions hold.
  }];

  let statistics = [
    Statistic<"numPortsMoved", "num-ports-moved",
              "Number of inner symbols moved from ports to wires">
  ];
}

def HWIMConstProp : Pass<"hw-imconstprop", "mlir::ModuleOp"> {
  let summary = "Inter-module constant propagation";
  let description = [{
    This pass performs inter-module constant propagation for HW modules.
    It propagates constant values across module boundaries through instances,
    and folds operations with constant operands.
  }];
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect"];
  let statistics = [
    Statistic<"numValuesFolded", "num-values-folded",
              "Number of values folded to a constant">,
    Statistic<"numOpsErased", "num-ops-erased",
              "Number of dead ops erased">
  ];
}

#endif // CIRCT_DIALECT_HW_PASSES_TD
