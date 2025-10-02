//===- AssignOutputDirs.cpp - Assign Output Directories ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/Debug.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "firrtl-assign-output-dirs"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_ASSIGNOUTPUTDIRS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

using hw::OutputFileAttr;

// If moduleOutputDir is a relative path, convert it to an absolute path, by
// interpreting moduleOutputDir as relative to the outputDir.
static void makeAbsolute(StringRef outputDir,
                         SmallString<64> &moduleOutputDir) {
  auto sep = llvm::sys::path::get_separator();
  if (!moduleOutputDir.empty())
    assert(moduleOutputDir.ends_with(sep));
  fs::make_absolute(outputDir, moduleOutputDir);
  path::remove_dots(moduleOutputDir, true);
  moduleOutputDir += sep;
}

// If outputDir is a prefix of moduleOutputDir, then make moduleOutputDir
// relative to outputDir. Otherwise, leave moduleOutputDir as absolute.
static void tryMakeRelative(StringRef outputDir,
                            SmallString<64> &moduleOutputDir) {
  if (moduleOutputDir.starts_with(outputDir))
    moduleOutputDir.erase(moduleOutputDir.begin(),
                          moduleOutputDir.begin() + outputDir.size());
}

static void makeCommonPrefix(StringRef outputDir, SmallString<64> &a,
                             OutputFileAttr attr) {
  if (attr) {
    SmallString<64> b(attr.getDirectory());
    makeAbsolute(outputDir, b);
    makeCommonPrefix(a, b);
  } else {
    makeCommonPrefix(a, outputDir);
  }
}

static OutputFileAttr getOutputFile(igraph::ModuleOpInterface op) {
  return op->getAttrOfType<hw::OutputFileAttr>("output_file");
}

namespace {
struct AssignOutputDirsPass
    : public circt::firrtl::impl::AssignOutputDirsBase<AssignOutputDirsPass> {
  using Base::Base;

  AssignOutputDirsPass(StringRef outputDir) {
    if (!outputDir.empty())
      outputDirOption = std::string(outputDir);
  }

  void runOnOperation() override;
};
} // namespace

void AssignOutputDirsPass::runOnOperation() {
#ifndef NDEBUG
  ScopedDebugPassLogger _(this);
#endif
  SmallString<64> outputDir(outputDirOption);
  if (fs::make_absolute(outputDir)) {
    emitError(mlir::UnknownLoc::get(&getContext()),
              "failed to convert the output directory to an absolute path");
    signalPassFailure();
    return;
  }
  path::remove_dots(outputDir, true);
  auto sep = path::get_separator();
  if (!outputDir.ends_with(sep))
    outputDir.append(sep);

  bool changed = false;

  LLVM_DEBUG(llvm::dbgs() << "Updating modules:\n");
  getAnalysis<InstanceGraph>().walkInversePostOrder([&](auto &node) {
    FModuleLike moduleLike =
        dyn_cast<FModuleLike>(node.getModule().getOperation());
    if (!moduleLike || !isa<FModuleOp, FExtModuleOp>(moduleLike))
      return;
    if (moduleLike->getAttrOfType<hw::OutputFileAttr>("output_file") ||
        moduleLike.isPublic())
      return;

    // Get the output directory of the first parent, and then fold the current
    // output directory with the LCA of all other discovered output
    // directories.
    SmallString<64> moduleOutputDir;
    auto i = node.usesBegin();
    auto e = node.usesEnd();
    for (; i != e; ++i) {
      auto parent = (*i)->getParent()->getModule();
      auto file = getOutputFile(parent);
      if (file) {
        moduleOutputDir = file.getDirectory();
        makeAbsolute(outputDir, moduleOutputDir);
      } else {
        moduleOutputDir = outputDir;
      }
      ++i;
      break;
    }
    for (; i != e; ++i) {
      auto parent = (*i)->getParent()->getModule();
      makeCommonPrefix(outputDir, moduleOutputDir, getOutputFile(parent));
    }

    tryMakeRelative(outputDir, moduleOutputDir);
    if (!moduleOutputDir.empty()) {
      auto f =
          hw::OutputFileAttr::getAsDirectory(&getContext(), moduleOutputDir);
      moduleLike->setAttr("output_file", f);
      changed = true;
      LLVM_DEBUG({
        llvm::dbgs() << "  - name: " << moduleLike.getName() << "\n"
                     << "    directory: " << f.getFilename() << "\n";
      });
    }
  });

  if (!changed)
    markAllAnalysesPreserved();
}
