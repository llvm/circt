//===- NameModulesByContent.cpp - Name modules by a hash of their body ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Give the modules that share a name up to a numeric suffix (Queue, Queue_1,
// Queue_2, ...) names made from a hash of their own contents instead, so that a
// module keeps its name for as long as its contents do not change, whatever
// order the modules were created in.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SHA256.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_NAMEMODULESBYCONTENT
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

/// The name a module was given before a `_<n>` suffix was added to make it
/// unique: "Queue_12" is "Queue". A name that is nothing but the suffix is
/// kept whole.
static StringRef baseName(StringRef name) {
  auto underscore = name.rfind('_');
  if (underscore == StringRef::npos || underscore == 0 ||
      underscore + 1 == name.size())
    return name;
  auto suffix = name.drop_front(underscore + 1);
  if (!llvm::all_of(suffix, llvm::isDigit))
    return name;
  return name.take_front(underscore);
}

static std::string sha256Hex(StringRef text) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)),
                     /*LowerCase=*/true);
}

namespace {
/// A module to rename, and what its new name is made from.
struct Candidate {
  FModuleOp module;
  /// The module on its own, as `hashedText` prints it.
  std::string text;
  /// SHA-256 of what the name is made from, in hexadecimal.
  std::string hash;
  StringAttr newName;
};

struct NameModulesByContentPass
    : public circt::firrtl::impl::NameModulesByContentBase<
          NameModulesByContentPass> {
  using Base::Base;

  void runOnOperation() override;

private:
  /// The text a module's name is hashed from: the module on its own, printed
  /// with its own name, its annotations and its source locations left out,
  /// and with every module it instantiates named by its name without a
  /// numeric suffix. The modules it instantiates are black boxes here: their
  /// contents, and so their names, do not change this module's name.
  /// Annotations are left out because non-local ones name hierarchical paths
  /// numbered across the whole circuit, locations because they say where a
  /// module was written, not what it is.
  std::string hashedText(FModuleOp module);

  /// Point every instance of `module` and every hierarchical path through it
  /// at `newName`, then rename it. The instance graph is looked up by the old
  /// name, so this must come before any other lookup of the same module.
  LogicalResult rename(FModuleOp module, StringAttr newName,
                       InstanceGraph &instanceGraph, NLATable &nlaTable);
};
} // namespace

std::string NameModulesByContentPass::hashedText(FModuleOp module) {
  auto *context = module.getContext();
  auto noAnnotations = ArrayAttr::get(context, {});
  auto annotationsName = StringAttr::get(context, "annotations");
  auto portAnnotationsName = StringAttr::get(context, "portAnnotations");
  auto blackBox = [&](Attribute name) -> Attribute {
    return FlatSymbolRefAttr::get(
        context, baseName(cast<FlatSymbolRefAttr>(name).getValue()));
  };

  mlir::OwningOpRef<FModuleOp> clone = cast<FModuleOp>(module->clone());
  SymbolTable::setSymbolName(clone.get(), "");
  SmallVector<Attribute> unknownLocations(clone->getNumPorts(),
                                          UnknownLoc::get(context));
  clone->setPortLocationsAttr(ArrayAttr::get(context, unknownLocations));
  clone->walk([&](Operation *op) {
    if (op->hasAttr(annotationsName))
      op->setAttr(annotationsName, noAnnotations);
    if (op->hasAttr(portAnnotationsName))
      op->setAttr(portAnnotationsName, noAnnotations);
    if (auto instance = dyn_cast<InstanceOp>(op))
      instance.setModuleNameAttr(
          cast<FlatSymbolRefAttr>(blackBox(instance.getModuleNameAttr())));
    if (auto choice = dyn_cast<InstanceChoiceOp>(op))
      choice.setModuleNamesAttr(ArrayAttr::get(
          context, llvm::map_to_vector(choice.getModuleNamesAttr(), blackBox)));
  });

  // Generic form, so that every attribute is printed; local scope, so that
  // values are numbered within the module alone; operation locations are not
  // printed by default.
  std::string text;
  llvm::raw_string_ostream os(text);
  clone->print(os, mlir::OpPrintingFlags()
                       .printGenericOpForm()
                       .useLocalScope()
                       .assumeVerified());
  return text;
}

LogicalResult NameModulesByContentPass::rename(FModuleOp module,
                                               StringAttr newName,
                                               InstanceGraph &instanceGraph,
                                               NLATable &nlaTable) {
  auto oldName = module.getModuleNameAttr();
  auto newRef = FlatSymbolRefAttr::get(newName);
  for (auto *use : instanceGraph.lookup(module)->uses()) {
    auto *op = use->getInstance().getOperation();
    if (auto instance = dyn_cast<InstanceOp>(op)) {
      instance.setModuleNameAttr(newRef);
      continue;
    }
    if (auto choice = dyn_cast<InstanceChoiceOp>(op)) {
      SmallVector<Attribute> names;
      for (auto name : choice.getModuleNamesAttr())
        names.push_back(
            cast<FlatSymbolRefAttr>(name).getAttr() == oldName ? newRef : name);
      choice.setModuleNamesAttr(ArrayAttr::get(op->getContext(), names));
      continue;
    }
    return op->emitOpError("instantiates module ")
           << oldName << ", which this pass does not know how to rename";
  }
  nlaTable.renameModule(oldName, newName);
  SymbolTable::setSymbolName(module, newName);
  return success();
}

void NameModulesByContentPass::runOnOperation() {
  auto *context = &getContext();
  auto circuit = getOperation();
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  auto &nlaTable = getAnalysis<NLATable>();
  auto noDedupClass = StringAttr::get(context, noDedupAnnoClass);

  // Only a name that more than one module shares up to a numeric suffix
  // depends on the order the modules were created in. Every module counts,
  // including the ones that keep their names below.
  llvm::StringMap<unsigned> familySize;
  for (auto module : circuit.getOps<FModuleLike>())
    ++familySize[baseName(module.getModuleName())];

  // Public modules keep the name the user gave them; external modules are
  // named by their definition; NoDedup modules were asked to be left alone.
  // The candidates are listed bottom-up, so that a tie below can be broken
  // with the new names of the modules a candidate instantiates.
  SmallVector<Candidate> candidates;
  instanceGraph.walkPostOrder([&](auto &node) {
    auto module = dyn_cast<FModuleOp>(node.getModule().getOperation());
    if (!module || module.isPublic() ||
        AnnotationSet::hasAnnotation(module, noDedupClass) ||
        familySize[baseName(module.getModuleName())] < 2)
      return;
    candidates.push_back({module, {}, {}, {}});
  });
  for (auto &candidate : candidates) {
    candidate.text = hashedText(candidate.module);
    candidate.hash = sha256Hex(candidate.text);
  }

  // Modules that print the same differ only in which modules they
  // instantiate: say a wrapper around one of two queues with the same ports.
  // Tell them apart by the new names of those modules.
  llvm::StringMap<unsigned> sameText;
  for (auto &candidate : candidates)
    ++sameText[candidate.hash];
  DenseMap<StringAttr, StringAttr> newNameOf;
  auto finalName = [&](StringAttr name) {
    auto it = newNameOf.find(name);
    return it == newNameOf.end() ? name : it->second;
  };
  auto shortName = [&](FModuleOp module, StringRef hash) {
    return StringAttr::get(context, baseName(module.getModuleName()) + "_" +
                                        hash.take_front(hashLength));
  };
  for (auto &candidate : candidates) {
    if (sameText[candidate.hash] > 1) {
      std::string children;
      for (auto *record : *instanceGraph.lookup(candidate.module))
        children +=
            (finalName(record->getTarget()->getModule().getModuleNameAttr())
                 .getValue() +
             "\n")
                .str();
      candidate.hash = sha256Hex(candidate.text + children);
    }
    candidate.newName = shortName(candidate.module, candidate.hash);
    newNameOf[candidate.module.getModuleNameAttr()] = candidate.newName;
  }

  // Every name in the circuit now. A new name may not be any of them but the
  // module's own: the modules are renamed one at a time, and two modules must
  // never share a name in between.
  llvm::StringMap<Operation *> current;
  for (auto &op : *circuit.getBodyBlock())
    if (auto name = SymbolTable::getSymbolName(&op))
      current[name.getValue()] = &op;

  // Two candidates with the same new name are either identical, which only
  // dedup groups allow, or differ beyond the digits of the hash kept in the
  // name. Either way there is no name to give them that does not depend on
  // their order, so stop.
  llvm::StringMap<Candidate *> byNewName;
  for (auto &candidate : candidates) {
    auto module = candidate.module;
    if (auto *other = current.lookup(candidate.newName);
        other && other != module.getOperation()) {
      auto diag = module.emitError("cannot rename module '")
                  << module.getModuleName() << "' to '"
                  << candidate.newName.getValue() << "': the name is taken";
      diag.attachNote(other->getLoc()) << "by this";
      return signalPassFailure();
    }
    auto [it, inserted] =
        byNewName.try_emplace(candidate.newName.getValue(), &candidate);
    if (inserted)
      continue;
    auto *other = it->second;
    auto diag = module.emitError("modules '")
                << other->module.getModuleName() << "' and '"
                << module.getModuleName() << "' would both be named '"
                << candidate.newName.getValue() << "': ";
    if (other->hash == candidate.hash)
      diag << "they are identical, and were not deduplicated";
    else
      diag << "their hashes share the first " << hashLength
           << " digits; use a larger hash-length";
    diag.attachNote(other->module.getLoc()) << "the other module";
    return signalPassFailure();
  }

  for (auto &candidate : candidates) {
    if (candidate.newName == candidate.module.getModuleNameAttr())
      continue;
    if (failed(rename(candidate.module, candidate.newName, instanceGraph,
                      nlaTable)))
      return signalPassFailure();
    ++numRenamed;
  }
  // The instance graph indexes modules by name and is not updated above, so
  // no analysis is preserved.
}
