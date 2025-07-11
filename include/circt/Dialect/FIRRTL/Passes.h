//===- Passes.h - FIRRTL pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_PASSES_H
#define CIRCT_DIALECT_FIRRTL_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace firrtl {

/// Configure which aggregate values will be preserved by the LowerTypes pass.
namespace PreserveAggregate {
enum PreserveMode {
  /// Don't preserve aggregate at all. This has been default behaivor and
  /// compatible with SFC.
  None,

  /// Preserve only 1d vectors of ground type (e.g. UInt<2>[3]).
  OneDimVec,

  /// Preserve only vectors (e.g. UInt<2>[3][3]).
  Vec,

  /// Preserve all aggregate values.
  All,
};
}

/// Configure which values will be explicitly preserved by the DropNames pass.
namespace PreserveValues {
enum PreserveMode {
  /// Strip all names. No name on declaration is preserved.
  Strip,
  /// Don't explicitly preserve any named values. Every named operation could
  /// be optimized away by the compiler. Unlike `Strip` names could be preserved
  /// until the end.
  None,
  // Explicitly preserved values with meaningful names.  If a name begins with
  // an "_" it is not considered meaningful.
  Named,
  // Explicitly preserve all values.  No named operation should be optimized
  // away by the compiler.
  All,
};
} // namespace PreserveValues

enum class CompanionMode {
  // Lower companions to SystemVerilog binds.
  Bind,
  // Lower companions to explicit instances. Used when assertions or other
  // debugging constructs from the companion are to be included in the design.
  Instantiate,
  // Drop companion modules, eliminating them from the design.
  Drop,
};

#define GEN_PASS_DECL
#include "circt/Dialect/FIRRTL/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLowerBundleVectorTypesPass();

std::unique_ptr<mlir::Pass>
createLowerIntmodulesPass(bool fixupEICGWrapper = false);

std::unique_ptr<mlir::Pass>
createMemToRegOfVecPass(bool replSeqMem = false, bool ignoreReadEnable = false);

std::unique_ptr<mlir::Pass>
createGrandCentralPass(CompanionMode companionMode = CompanionMode::Bind);

std::unique_ptr<mlir::Pass>
createMergeConnectionsPass(bool enableAggressiveMerging = false);

std::unique_ptr<mlir::Pass>
createDropNamesPass(PreserveValues::PreserveMode mode = PreserveValues::None);

std::unique_ptr<mlir::Pass>
createResolveTracesPass(mlir::StringRef outputAnnotationFilename = "");

std::unique_ptr<mlir::Pass>
createLinkCircuitsPass(mlir::StringRef baseCircuitName = "",
                       bool noMangle = false);

std::unique_ptr<mlir::Pass>
createSpecializeOptionPass(bool selectDefaultInstanceChoice = false);

std::unique_ptr<mlir::Pass>
createAssignOutputDirsPass(mlir::StringRef outputDir = "");

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/FIRRTL/Passes.h.inc"

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_PASSES_H
