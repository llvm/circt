//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STANDALONE_PASSES
#define CIRCT_STANDALONE_PASSES

include "mlir/Pass/PassBase.td"

def RenameHWModule
    : Pass<"circt-standalone-rename-hw-module", "::mlir::ModuleOp"> {
  let summary = "Rename a hw.module named `bar` to `foo`";
  let description = [{
    This pass is intentionally small and CIRCT-specific: it finds a top-level
    `hw.module @bar` and renames it to `hw.module @foo`.
  }];
  let dependentDialects = ["::circt::hw::HWDialect"];
}

#endif // CIRCT_STANDALONE_PASSES
