//===-- Passes.td - HW pass definition file ------0000------*- tablegen -*-===//
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

def HWFlattenStructs : Pass<"hw-flatten-structs", "mlir::ModuleOp"> {
  let summary = "Flatten struct types";
  let constructor = "circt::hw::createHWFlattenStructsPass()";
  let description = [{
    This pass breaks down `hw.struct` types into their scalar elements, and
    expands module and instance ports accordingly.

    Example:
    ```
    hw.module @foo(%in: !hw.struct<a: i8, b: i8>) ->
                  (out: !hw.struct<a: i8, b: i8>) {
      hw.output %in : !hw.struct<a: i8, b: i8>
    }
    ```
    is transformed to
    ```
    hw.module @foo(%in_a: i8, %in_b: i8) -> (out_a: i8, out_b: i8) {
      hw.output %in_a, %in_b : i8, i8
    }
    ```
  }];
}

#endif // CIRCT_DIALECT_HW_PASSES_TD
