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

//===----------------------------------------------------------------------===//
// Directory Utilities
//===----------------------------------------------------------------------===//

static SmallString<128> canonicalize(StringRef directory) {
  SmallString<128> native;
  if (directory.empty())
    return native;

  llvm::sys::path::native(directory, native);
  auto separator = llvm::sys::path::get_separator();
  if (!native.ends_with(separator))
    native += separator;
  return native;
}

static StringAttr canonicalize(const StringAttr directory) {
  if (!directory)
    return nullptr;
  if (directory.empty())
    return nullptr;
  return StringAttr::get(directory.getContext(),
                         canonicalize(directory.getValue()));
}

//===----------------------------------------------------------------------===//
// Output Directory Priority Table
//===----------------------------------------------------------------------===//

namespace {

struct OutputDirInfo {
  OutputDirInfo(StringAttr name, size_t depth = SIZE_MAX,
                size_t parent = SIZE_MAX)
      : name(name), depth(depth), parent(parent) {}
  StringAttr name;
  size_t depth;
  size_t parent;
};

/// A table that helps decide which directory a floating module must be placed.
/// Given two candidate output directories, the table can answer the question,
/// which directory should a resource go.
///
/// Output directories are organized into a tree, which represents the relative
/// "specificity" of a directory. If a resource could be placed in more than one
/// directory, then it is output in the least-common-ancestor of the
/// candidate output directories, which represents the "most specific" place
/// a resource could go, which is still general enough to cover all uses.
class OutputDirTable {
public:
  LogicalResult initialize(CircuitOp);

  /// Given two directory names, returns the least-common-ancestor directory.
  /// If the LCA is the toplevel output directory (which is considered the most
  /// general), return null.
  StringAttr lca(StringAttr, StringAttr);

  unsigned getNumLcaComputations() const { return numLcaComputations; }

private:
  DenseMap<StringAttr, size_t> indexTable;
  std::vector<OutputDirInfo> infoTable;
  unsigned numLcaComputations = 0;
};
} // namespace

LogicalResult OutputDirTable::initialize(CircuitOp circuit) {
  auto err = [&]() { return emitError(circuit.getLoc()); };

  // Stage 1: Build a table mapping child directories to their parents.
  indexTable[nullptr] = 0;
  infoTable.emplace_back(nullptr, 0, SIZE_MAX);
  AnnotationSet annos(circuit);
  for (auto anno : annos) {
    if (anno.isClass(ouputDirPrecedenceAnnoClass)) {
      auto nameField = anno.getMember<StringAttr>("name");
      if (!nameField)
        return err() << "output directory declaration missing name";
      if (nameField.empty())
        return err() << "output directory name cannot be empty";
      auto name = canonicalize(nameField);

      auto parentField = anno.getMember<StringAttr>("parent");
      if (!parentField)
        return err() << "output directory declaration missing parent";
      auto parent = canonicalize(parentField);

      auto parentIdx = infoTable.size();
      {
        auto [it, inserted] = indexTable.try_emplace(parent, parentIdx);
        if (inserted)
          infoTable.emplace_back(parent, SIZE_MAX, SIZE_MAX);
        else
          parentIdx = it->second;
      }

      {
        auto [it, inserted] = indexTable.try_emplace(name, infoTable.size());
        if (inserted) {
          infoTable.emplace_back(name, SIZE_MAX, parentIdx);
        } else {
          auto &child = infoTable[it->second];
          assert(child.name == name);
          if (child.parent != SIZE_MAX)
            return err() << "output directory " << name
                         << " declared multiple times";
          child.parent = parentIdx;
        }
      }
    }
  }
  for (auto &info : infoTable)
    if (info.parent == SIZE_MAX)
      info.parent = 0;

  // Stage 2: Set the depth/priority of each directory, and check for cycles.
  SmallVector<size_t> stack;
  BitVector seen(infoTable.size(), false);
  for (unsigned i = 0, e = infoTable.size(); i < e; ++i) {
    auto *current = &infoTable[i];
    if (current->depth != SIZE_MAX)
      continue;
    seen.reset();
    seen.set(i);
    while (true) {
      seen.set(i);
      auto *current = &infoTable[i];
      auto *parent = &infoTable[current->parent];
      if (seen[current->parent])
        return emitError(circuit.getLoc())
               << "circular precedence between output directories "
               << current->name << " and " << parent->name;
      if (parent->depth == SIZE_MAX) {
        stack.push_back(i);
        i = current->parent;
        continue;
      }
      current->depth = parent->depth + 1;
      if (stack.empty())
        break;
      i = stack.back();
      stack.pop_back();
    }
  }
  return success();
}

StringAttr OutputDirTable::lca(StringAttr nameA, StringAttr nameB) {
  if (!nameA || !nameB)
    return nullptr;
  if (nameA == nameB)
    return nameA;

  auto lookupA = indexTable.find(nameA);
  if (lookupA == indexTable.end())
    return nullptr;

  auto lookupB = indexTable.find(nameB);
  if (lookupB == indexTable.end())
    return nullptr;

  ++numLcaComputations;

  auto a = infoTable[lookupA->second];
  auto b = infoTable[lookupB->second];

  while (a.depth > b.depth)
    a = infoTable[a.parent];
  while (b.depth > a.depth)
    b = infoTable[b.parent];
  while (a.name != b.name) {
    a = infoTable[a.parent];
    b = infoTable[b.parent];
  }
  return a.name;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class AssignOutputDirsPass : public AssignOutputDirsBase<AssignOutputDirsPass> {
  void runOnOperation() override;
};
} // namespace

static StringAttr getOutputDir(Operation *op) {
  auto outputFile = op->getAttrOfType<hw::OutputFileAttr>("output_file");
  if (!outputFile)
    return nullptr;
  return outputFile.getDirectoryAttr();
}

void AssignOutputDirsPass::runOnOperation() {
  auto falseAttr = BoolAttr::get(&getContext(), false);
  auto circuit = getOperation();
  OutputDirTable outDirTable;
  if (failed(outDirTable.initialize(circuit)))
    return signalPassFailure();

  DenseSet<InstanceGraphNode *> visited;
  for (auto *root : getAnalysis<InstanceGraph>()) {
    for (auto *node : llvm::inverse_post_order_ext(root, visited)) {
      auto module = dyn_cast<FModuleOp>(node->getModule().getOperation());
      if (!module || module->getAttrOfType<hw::OutputFileAttr>("output_file") ||
          module.isPublic())
        continue;
      StringAttr outputDir;
      auto i = node->usesBegin();
      auto e = node->usesEnd();
      for (; i != e; ++i) {
        if (auto parent = dyn_cast<FModuleOp>((*i)->getParent()->getModule())) {
          outputDir = getOutputDir(parent);
          ++i;
          break;
        }
      }
      for (; i != e; ++i) {
        if (outputDir == nullptr)
          break;
        if (auto parent =
                dyn_cast<FModuleOp>((*i)->getParent()->getModule<FModuleOp>()))
          outputDir = outDirTable.lca(outputDir, getOutputDir(parent));
      }
      if (outputDir)
        module->setAttr("output_file", hw::OutputFileAttr::get(
                                           outputDir, falseAttr, falseAttr));
    }
  }

  numLcaComputations = outDirTable.getNumLcaComputations();
  markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createAssignOutputDirsPass() {
  return std::make_unique<AssignOutputDirsPass>();
}
