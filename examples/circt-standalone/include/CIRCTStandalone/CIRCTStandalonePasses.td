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

def RenameWires
    : Pass<"circt-standalone-rename-wires", "hw::HWModuleOp"> {
  let summary = "Rename all hw.wire operations in a module";
  let description = [{
    This pass is intentionally small and CIRCT-specific: it walks a hardware
    module and renames each `hw.wire` operation to `foo_<n>`.
  }];
  let dependentDialects = ["::circt::hw::HWDialect"];
}

#endif // CIRCT_STANDALONE_PASSES
