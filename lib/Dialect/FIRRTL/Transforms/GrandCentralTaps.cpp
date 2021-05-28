//===- GrandCentralTaps.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GrandCentralTaps pass.
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gct"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Static information gathered once upfront
//===----------------------------------------------------------------------===//

namespace {
/// Attributes used throughout the annotations.
struct Strings {
  MLIRContext *const cx;
  Strings(MLIRContext *cx) : cx(cx) {}

  Identifier annos = Identifier::get("annotations", cx);
  Identifier fannos = Identifier::get("firrtl.annotations", cx);

  StringAttr dataTapsClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.DataTapsAnnotation");
  StringAttr memTapClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.MemTapAnnotation");

  StringAttr deletedKeyClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.DeletedDataTapKey");
  StringAttr literalKeyClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.LiteralDataTapKey");
  StringAttr referenceKeyClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.ReferenceDataTapKey");
  StringAttr internalKeyClass = StringAttr::get(
      cx, "sifive.enterprise.grandcentral.DataTapModuleSignalKey");
};
} // namespace

//===----------------------------------------------------------------------===//
// Data Taps Implementation
//===----------------------------------------------------------------------===//

// static LogicalResult processDataTapAnnotation(FExtModuleOp module,
//                                               DictionaryAttr anno) {}

/// A port annotated with a data tap key or mem tap.
struct AnnotatedPort {
  unsigned portNum;
  DictionaryAttr anno;
};

/// An extmodule that has annotated ports.
struct AnnotatedExtModule {
  FExtModuleOp extModule;
  SmallVector<AnnotatedPort, 4> portAnnos;
};

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

class GrandCentralTapsPass : public GrandCentralTapsBase<GrandCentralTapsPass> {
  void runOnOperation() override;
};

void GrandCentralTapsPass::runOnOperation() {
  auto circuitOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Running the GCT Data Taps pass\n");

  // Here's a rough idea of what the Scala code is doing:
  // - Gather the `source` of all `keys` of all `DataTapsAnnotation`s throughout
  //   the design.
  // - Convert the sources, which are specified on modules, to the absolute
  //   paths in all instances. E.g. module M tap x will produce a.x and b.x if
  //   there are two instances a and b of module M in the design.
  // - All data tap keys are specified on black box ports.
  // - The code then processes every DataTapsAnnotation separately as follows
  //   (with the targeted blackbox and keys):
  // - Check for collisions between SV keywords and key port names (skip this).
  // - Find all instances of the blackbox, but then just pick the first one (we
  //   should probably do this for each?)
  // - Process each key independently as follows:
  // - Look up the absolute path of the source in the map. Ensure it exists and
  //   is unambiguous. Make it relative to the blackbox instance path.
  // - Look up the port on the black box.
  // - Create a hierarchical SV name and store this as an assignment to the
  //   blackbox port.
  //   - DeletedDataTapKey: skip and don't create a wiring
  //   - LiteralDataTapKey: just the literal
  //   - ReferenceDataTapKey: relative path with "." + source name
  //   - DataTapModuleSignalKey: relative path with "." + internal path
  // - Generate a body for the blackbox module with the signal mapping

  // Gather some string attributes in the context to simplify working with the
  // annotations.
  Strings strings(&getContext());

  // Gather a list of extmodules that have data or mem tap annotations to be
  // expanded.
  SmallVector<AnnotatedExtModule, 4> modules;
  for (auto &op : *circuitOp.getBody()) {
    auto extModule = dyn_cast<FExtModuleOp>(&op);
    if (!extModule)
      continue;

    // Go through the module ports and collect the annotated ones.
    AnnotatedExtModule result{extModule, {}};
    for (unsigned argNum = 0; argNum < extModule.getNumArguments(); ++argNum) {
      auto attrs =
          extModule.getArgAttrOfType<ArrayAttr>(argNum, strings.fannos);
      if (!attrs)
        continue;

      // Go through all annotations on this port and add the data tap key and
      // mem tap ones to the list.
      for (auto attr : attrs) {
        auto anno = attr.dyn_cast<DictionaryAttr>();
        if (!anno)
          continue;
        auto cls = anno.getAs<StringAttr>("class");
        if (cls == strings.memTapClass || cls == strings.deletedKeyClass ||
            cls == strings.literalKeyClass ||
            cls == strings.referenceKeyClass || cls == strings.internalKeyClass)
          result.portAnnos.push_back({argNum, anno});
      }
    }
    if (!result.portAnnos.empty())
      modules.push_back(std::move(result));
  }

  for (auto m : modules) {
    LLVM_DEBUG(llvm::dbgs() << "Extmodule " << m.extModule.getName() << " has "
                            << m.portAnnos.size() << " port annotations\n");
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralTapsPass() {
  return std::make_unique<GrandCentralTapsPass>();
}
