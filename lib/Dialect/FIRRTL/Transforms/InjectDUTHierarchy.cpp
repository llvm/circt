//===- InjectDUTHierarchy.cpp - Add hierarchy above the DUT ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the SiFive transform InjectDUTHierarchy.  This moves all
// the logic inside the DUT into a new module named using an annotation.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"

using namespace circt;
using namespace firrtl;

namespace {
struct InjectDUTHierarchy : public InjectDUTHierarchyBase<InjectDUTHierarchy> {
  void runOnOperation() override;
};
} // namespace

void InjectDUTHierarchy::runOnOperation() {

  CircuitOp circuit = getOperation();

  /// The design-under-test (DUT).  This is kept up-to-date by the pass as the
  /// DUT changes due to internal logic.
  FModuleOp dut;

  /// The wrapper module that is created inside the DUT to house all its logic.
  FModuleOp wrapper;

  /// The name of the new module to create under the DUT.
  StringAttr wrapperName;

  /// Mutable indicator that an error occurred for some reason.  If this is ever
  /// true, then the pass can just signalPassFailure.
  bool error = false;

  AnnotationSet::removeAnnotations(circuit, [&](Annotation anno) {
    if (!anno.isClass(injectDUTHierarchyAnnoClass))
      return false;

    auto name = anno.getMember<StringAttr>("name");
    if (!name) {
      emitError(circuit->getLoc())
          << "contained a malformed "
             "'sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation' "
             "annotation that did not contain a 'name' field";
      error = true;
      return false;
    }

    if (wrapperName) {
      emitError(circuit->getLoc())
          << "contained multiple "
             "'sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation' "
             "annotations when at most one is allowed";
      error = true;
      return false;
    }

    wrapperName = name;
    return true;
  });

  if (error)
    return signalPassFailure();

  // The prerequisites for the pass to run were not met.  Indicate that no work
  // was done and exit.
  if (!wrapperName)
    return markAllAnalysesPreserved();

  // TODO: Combine this logic with GrandCentral and other places that need to
  // find the DUT.  Consider changing the MarkDUTAnnotation scattering to put
  // this information on the Circuit so that we don't have to dig through all
  // the modules to find the DUT.
  for (auto mod : circuit.getOps<FModuleOp>()) {
    if (!AnnotationSet(mod).hasAnnotation(dutAnnoClass))
      continue;
    if (dut) {
      auto diag = emitError(mod.getLoc())
                  << "is marked with a '" << dutAnnoClass << "', but '"
                  << dut.moduleName()
                  << "' also had such an annotation (this should "
                     "be impossible!)";
      diag.attachNote(dut.getLoc()) << "the first DUT was found here";
      error = true;
      break;
    }
    dut = mod;
  }

  if (error)
    return signalPassFailure();

  // If a hierarchy annotation was provided, ensure that a DUT annotation also
  // exists.  The pass could silently ignore this case and do nothing, but it is
  // better to provide an error.
  if (wrapperName && !dut) {
    emitError(circuit->getLoc())
        << "contained a '" << injectDUTHierarchyAnnoClass << "', but no '"
        << dutAnnoClass << "' was provided";
    error = true;
  }

  if (error)
    return signalPassFailure();

  // Create a module that will become the new DUT.  The original DUT is renamed
  // to become the wrapper.  This is done to save copying into the wrapper.
  // While the logical movement is "copy the body of the DUT into a wrapper", it
  // is mechanically more straigthforward to make the DUT the wrappper.  After
  // this block finishes, the "dut" and "wrapper" variables are set correctly.
  // This logic is intentionally put into a block to avoid confusion while the
  // dut and wrapper do not match the logical definition.
  OpBuilder b(circuit.getContext());
  CircuitNamespace circuitNS(circuit);
  {
    b.setInsertionPointAfter(dut);
    auto newDUT = b.create<FModuleOp>(dut.getLoc(), dut.getNameAttr(),
                                      dut.getPorts(), dut.annotations());

    SymbolTable::setSymbolVisibility(newDUT, dut.getVisibility());
    dut.setName(b.getStringAttr(circuitNS.newName(wrapperName.getValue())));

    // The original DUT module is now the wrapper.  The new module we just
    // created becomse the DUT.
    wrapper = dut;
    dut = newDUT;

    // Finish setting up the wrapper.  It can have no annotations.
    AnnotationSet::removePortAnnotations(wrapper,
                                         [](auto, auto) { return true; });
    AnnotationSet::removeAnnotations(wrapper, [](auto) { return true; });
  }

  // Instantiate the wrapper inside the DUT and wire it up.
  b.setInsertionPointToStart(dut.getBody());
  ModuleNamespace dutNS(dut);
  auto wrapperInst = b.create<InstanceOp>(
      b.getUnknownLoc(), wrapper, wrapper.moduleName(),
      NameKindEnum::InterestingName, ArrayRef<Attribute>{},
      ArrayRef<Attribute>{}, false,
      b.getStringAttr(dutNS.newName(wrapper.moduleName())));
  for (auto pair : llvm::enumerate(wrapperInst.getResults())) {
    Value lhs = dut.getArgument(pair.index());
    Value rhs = pair.value();
    if (dut.getPortDirection(pair.index()) == Direction::In)
      std::swap(lhs, rhs);
    b.create<ConnectOp>(b.getUnknownLoc(), lhs, rhs);
  }

  // Get ready to update non-local annotations (NLAs).  This requires both an
  // NLA table and knowledge of what the DUT's port symbols are.
  auto &nlaTable = getAnalysis<NLATable>();
  DenseSet<Attribute> dutPortSyms;
  for (auto port : dut.getPorts()) {
    if (!port.sym)
      continue;
    dutPortSyms.insert(port.sym);
  }

  // Update NLAs involving the DUT.  There are three cases to consider:
  //   1. The DUT or a DUT port is a leaf ref.  Do nothing.
  //   2. The DUT is the root.  Update the root module to be the wrapper.
  //   3. The NLA passes through the DUT.  Remove the original InnerRef and
  //      replace it with two InnerRefs: (1) on the DUT and (2) one the wrapper.
  for (auto nla : llvm::make_early_inc_range(nlaTable.lookup(dut))) {
    // The leaf ref is the DUT or a DUT port.
    if (nla.leafMod() == dut.moduleName())
      if (nla.isModule() || dutPortSyms.contains(nla.ref()))
        continue;

    // The DUT is the root module.
    auto namepath = nla.namepath().getValue();
    if (nla.root() == dut.getNameAttr()) {
      SmallVector<Attribute> newNamepath{hw::InnerRefAttr::get(
          wrapper.getNameAttr(),
          namepath.front().cast<hw::InnerRefAttr>().getName())};
      auto tail = namepath.drop_front();
      newNamepath.append(tail.begin(), tail.end());
      nla->setAttr("namepath", b.getArrayAttr(newNamepath));
      continue;
    }

    // The NLA passes through the DUT.
    auto nlaIdx = std::distance(
        namepath.begin(), llvm::find_if(namepath, [&](Attribute attr) {
          return attr.cast<hw::InnerRefAttr>().getModule() ==
                 dut.moduleNameAttr();
        }));
    auto front = namepath.take_front(nlaIdx);
    auto dutRef = namepath[nlaIdx].cast<hw::InnerRefAttr>();
    auto back = namepath.drop_front(nlaIdx + 1);
    SmallVector<Attribute> newNamepath(front.begin(), front.end());
    newNamepath.push_back(hw::InnerRefAttr::get(dut.moduleNameAttr(),
                                                wrapperInst.inner_symAttr()));
    newNamepath.push_back(
        hw::InnerRefAttr::get(wrapper.moduleNameAttr(), dutRef.getName()));
    newNamepath.append(back.begin(), back.end());
    nla->setAttr("namepath", b.getArrayAttr(newNamepath));
  }
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> circt::firrtl::createInjectDUTHierarchyPass() {
  return std::make_unique<InjectDUTHierarchy>();
}
