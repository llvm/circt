//===- AssignOutputDirs.cpp - Assign Output Directories ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "firrtl-assign-output-dirs"

using namespace circt;
using namespace firrtl;
namespace path = llvm::sys::path;

using hw::OutputFileAttr;

static StringRef lca(StringRef a, OutputFileAttr file) {
  if (!file)
    return StringRef();

  auto b = file.getDirectory();
  if (llvm::sys::path::is_absolute(b))
    return StringRef();

  for (auto i = path::begin(b), e = path::end(b); i != e; ++i)
    if (*i == "..")
      return StringRef();

  size_t i = 0;
  size_t e = std::min(a.size(), b.size());
  for (; i < e; ++i)
    if (a[i] != b[i])
      break;

  auto dir = a.substr(0, i);
  if (dir.ends_with(llvm::sys::path::get_separator()))
    return dir;

  return llvm::sys::path::parent_path(dir);
}

static OutputFileAttr getOutputFile(Operation *op) {
  return op->getAttrOfType<hw::OutputFileAttr>("output_file");
}

namespace {
class AssignOutputDirsPass : public AssignOutputDirsBase<AssignOutputDirsPass> {
  void runOnOperation() override;
};
} // namespace

void AssignOutputDirsPass::runOnOperation() {
  auto falseAttr = BoolAttr::get(&getContext(), false);
  bool changed = false;

  DenseSet<InstanceGraphNode *> visited;
  for (auto *root : getAnalysis<InstanceGraph>()) {
    for (auto *node : llvm::inverse_post_order_ext(root, visited)) {
      auto module = dyn_cast<FModuleOp>(node->getModule().getOperation());
      if (!module || module->getAttrOfType<hw::OutputFileAttr>("output_file") ||
          module.isPublic())
        continue;
      StringRef outputDir;
      auto i = node->usesBegin();
      auto e = node->usesEnd();
      for (; i != e; ++i) {
        if (auto parent = dyn_cast<FModuleOp>((*i)->getParent()->getModule())) {
          auto file = getOutputFile(parent);
          if (file)
            outputDir = file.getDirectory();
          ++i;
          break;
        }
      }
      for (; i != e; ++i) {
        if (outputDir.empty())
          break;
        if (auto parent =
                dyn_cast<FModuleOp>((*i)->getParent()->getModule<FModuleOp>()))
          outputDir = lca(outputDir, getOutputFile(parent));
      }
      if (!outputDir.empty()) {
        auto s = StringAttr::get(&getContext(), outputDir);
        auto f = hw::OutputFileAttr::get(s, falseAttr, falseAttr);
        module->setAttr("output_file", f);
        changed = true;
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createAssignOutputDirsPass() {
  return std::make_unique<AssignOutputDirsPass>();
}
