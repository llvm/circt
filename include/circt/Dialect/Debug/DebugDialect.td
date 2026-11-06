//===- DebugDialect.td - Debug dialect definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGDIALECT_TD
#define CIRCT_DIALECT_DEBUG_DEBUGDIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def DebugDialect : Dialect {
  let name = "dbg";
  let summary = "Facilities to represent debug information";
  let cppNamespace = "circt::debug";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;

  // Facts that belong to a module or a root debug op and
  // have no op to live on. Each is named `dbg.<name>`.
  let discardableAttrs = (ins
    // On a root debug op: the module that dedup merged it from.
    "::mlir::StringAttr":$merged_from,

    // On a module: the location of itself and of each module merged into it.
    // On an inlined scope: the same for the inlined module.
    "::mlir::DictionaryAttr":$module_locs,

    // On a module made by lowering, not in the source.
    "::mlir::UnitAttr":$lowered,

    // On a module in the source but made by the compiler, e.g. for a layer.
    "::mlir::UnitAttr":$artificial,

    // On a memory module that debug info describes.
    "::mlir::UnitAttr":$memory,

    // Class of a public module: `name` and `ext`, as on `#dbg.instance`.
    "::mlir::DictionaryAttr":$moduleinfo,

    // On an instance: its `#dbg.instance` descriptions. On an inlined scope:
    // those of the inlined instance, if dedup left more than one.
    "::mlir::ArrayAttr":$instance
  );

  let extraClassDeclaration = [{
    void registerOps();
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGDIALECT_TD
