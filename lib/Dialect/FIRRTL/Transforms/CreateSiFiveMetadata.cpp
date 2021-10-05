//===- CreateSiFiveMetadata.cpp - Create various metadata -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CreateSiFiveMetadata pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace firrtl;

static const char seqMemAnnoClass[] =
    "sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation";
static const char dutAnnoClass[] = "sifive.enterprise.firrtl.MarkDUTAnnotation";
/// Attribute that indicates where some json files should be dumped.
static const char metadataDirectoryAnnoClass[] =
    "sifive.enterprise.firrtl.MetadataDirAnnotation";

namespace {
class CreateSiFiveMetadataPass
    : public CreateSiFiveMetadataBase<CreateSiFiveMetadataPass> {
  LogicalResult emitRetimeModulesMetadata();
  LogicalResult emitSitestBlackboxMetadata();
  LogicalResult emitMemoryMetadata();
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() override;

  // The set of all modules underneath the design under test module.
  DenseSet<Operation *> dutModuleSet;
  // The design under test module.
  FModuleOp dutMod;
};
} // end anonymous namespace

/// This function collects all the firrtl.mem ops and creates a verbatim op with
/// the relevant memory attributes.
LogicalResult CreateSiFiveMetadataPass::emitMemoryMetadata() {

  // Lambda to get the number of read, write and read-write ports corresponding
  // to a MemOp.
  auto getNumPorts = [&](MemOp op, size_t &numReadPorts, size_t &numWritePorts,
                         size_t &numReadWritePorts) {
    for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
      auto portKind = op.getPortKind(i);
      if (portKind == MemOp::PortKind::Read)
        ++numReadPorts;
      else if (portKind == MemOp::PortKind::Write) {
        ++numWritePorts;
      } else
        ++numReadWritePorts;
    }
  };

  CircuitOp circuitOp = getOperation();
  // The instance graph analysis will be required to print the hierarchy names
  // of the memory.
  auto instancePathCache = InstancePathCache(getAnalysis<InstanceGraph>());

  auto printAttr = [&](llvm::json::OStream &J, Attribute &verifValAttr,
                       std::string &id) {
    if (auto intV = verifValAttr.dyn_cast<IntegerAttr>())
      J.attribute(id, (int64_t)intV.getValue().getZExtValue());
    else if (auto strV = verifValAttr.dyn_cast<StringAttr>())
      J.attribute(id, strV.getValue().str());
    else if (auto arrV = verifValAttr.dyn_cast<ArrayAttr>()) {
      std::string indices;
      J.attributeArray(id, [&] {
        for (auto arrI : llvm::enumerate(arrV)) {
          auto i = arrI.value();
          if (auto intV = i.dyn_cast<IntegerAttr>())
            J.value(std::to_string(intV.getValue().getZExtValue()));
          else if (auto strV = i.dyn_cast<StringAttr>())
            J.value(strV.getValue().str());
        }
      });
    }
  };
  // This lambda, writes to the given Json stream all the relevant memory
  // attributes. Also adds the memory attrbutes to the string for creating the
  // memmory conf file.
  auto createMemMetadata = [&](MemOp memOp, llvm::json::OStream &J,
                               std::string &seqMemConfStr) {
    size_t numReadPorts = 0, numWritePorts = 0, numReadWritePorts = 0;
    // Get the number of read,write ports.
    getNumPorts(memOp, numReadPorts, numWritePorts, numReadWritePorts);
    // Get the memory data width.
    auto width = memOp.getDataType().getBitWidthOrSentinel();
    // Metadata needs to be printed for memories which are candidates for macro
    // replacement. The requirements for macro replacement::
    // 1. read latency and write latency of one.
    // 2. only one readwrite port or write port.
    // 3. zero or one read port.
    // 4. undefined read-under-write behavior.
    if (!((memOp.readLatency() == 1 && memOp.writeLatency() == 1) &&
          (numWritePorts + numReadWritePorts == 1) && (numReadPorts <= 1) &&
          width > 0))
      return;
    // Get the absolute path for the parent memory, to create the hierarchy
    // names.
    auto paths =
        instancePathCache.getAbsolutePaths(memOp->getParentOfType<FModuleOp>());

    AnnotationSet anno = AnnotationSet(memOp);
    DictionaryAttr verifData = {};
    // Get the verification data attached with the memory op, if any.
    if (auto v = anno.getAnnotation(seqMemAnnoClass)) {
      verifData = v.get("data").cast<DictionaryAttr>();
      AnnotationSet::removeAnnotations(memOp, seqMemAnnoClass);
    }

    // Compute the mask granularity.
    auto maskGran = width / memOp.getMaskBits();
    // Now create the config string for the memory.
    std::string portStr;
    if (numWritePorts)
      portStr += "mwrite";
    if (numReadPorts) {
      if (!portStr.empty())
        portStr += ",";
      portStr += "read";
    }
    if (numReadWritePorts)
      portStr = "mrw";
    seqMemConfStr += "name " + memOp.name().str() + " depth " +
                     std::to_string(memOp.depth()) + " width " +
                     std::to_string(width) + " ports " + portStr +
                     " mask_gran " + std::to_string(maskGran) + "\n";
    // This adds a Json array element entry corresponding to this memory.
    J.object([&] {
      J.attribute("module_name", memOp.name());
      J.attribute("depth", (int64_t)memOp.depth());
      J.attribute("width", (int64_t)width);
      J.attribute("masked", "true");
      J.attribute("read", numReadPorts ? "true" : "false");
      J.attribute("write", numWritePorts ? "true" : "false");
      J.attribute("readwrite", numReadWritePorts ? "true" : "false");
      J.attribute("mask_granularity", (int64_t)maskGran);
      J.attributeArray("extra_ports", [&] {});
      // Record all the hierarchy names.
      SmallVector<std::string> hierNames;
      J.attributeArray("hierarchy", [&] {
        for (auto p : paths) {
          const InstanceOp &x = p.front();
          std::string hierName =
              x->getParentOfType<FModuleOp>().getName().str();
          for (InstanceOp inst : p) {
            hierName = hierName + "." + inst.name().str();
          }
          hierNames.push_back(hierName);
          J.value(hierName);
        }
      });
      // If verification annotation added to the memory op then print the data.
      if (verifData)
        J.attributeObject("verification_only_data", [&] {
          for (auto name : hierNames) {
            J.attributeObject(name, [&] {
              for (auto data : verifData) {
                // Id for the memory verification property.
                std::string id = data.first.strref().str();
                // Value for the property.
                auto verifValAttr = data.second;
                // Now print the value attribute based on its type.
                printAttr(J, verifValAttr, id);
              }
            });
          }
        });
    });
  };
  std::string testBenchJsonBuffer;
  llvm::raw_string_ostream testBenchOs(testBenchJsonBuffer);
  llvm::json::OStream testBenchJson(testBenchOs);
  std::string dutJsonBuffer;
  llvm::raw_string_ostream dutOs(dutJsonBuffer);
  llvm::json::OStream dutJson(dutOs);
  SmallVector<MemOp> dutMems;
  SmallVector<MemOp> tbMems;

  for (auto mod : circuitOp.getOps<FModuleOp>()) {
    bool isDut = dutModuleSet.contains(mod);
    for (auto memOp : mod.getBody().getOps<MemOp>())
      if (isDut)
        dutMems.push_back(memOp);
      else
        tbMems.push_back(memOp);
  }
  std::string seqMemConfStr, tbConfStr;
  dutJson.array([&] {
    for (auto memOp : dutMems)
      createMemMetadata(memOp, dutJson, seqMemConfStr);
  });
  testBenchJson.array([&] {
    // The tbConfStr is populated here, but unused, it will not be printed to
    // file.
    for (auto memOp : tbMems)
      createMemMetadata(memOp, testBenchJson, tbConfStr);
  });

  auto *context = &getContext();
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());
  AnnotationSet annos(circuitOp);
  auto diranno = annos.getAnnotation(metadataDirectoryAnnoClass);
  StringRef metadataDir = "metadata";
  if (diranno)
    if (auto dir = diranno.getAs<StringAttr>("dirname"))
      metadataDir = dir.getValue();

  // Use unknown loc to avoid printing the location in the metadata files.
  auto tbVerbatimOp = builder.create<sv::VerbatimOp>(builder.getUnknownLoc(),
                                                     testBenchJsonBuffer);
  auto dutVerbatimOp =
      builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), dutJsonBuffer);
  auto confVerbatimOp =
      builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), seqMemConfStr);
  auto fileAttr = hw::OutputFileAttr::getFromDirectoryAndFilename(
      context, metadataDir, "seq_mems.json", /*excludeFromFilelist=*/true);
  dutVerbatimOp->setAttr("output_file", fileAttr);
  fileAttr = hw::OutputFileAttr::getFromDirectoryAndFilename(
      context, metadataDir, "tb_seq_mems.json", /*excludeFromFilelist=*/true);
  tbVerbatimOp->setAttr("output_file", fileAttr);
  StringRef confFile = "memory";
  if (dutMod)
    confFile = dutMod.getName();
  fileAttr = hw::OutputFileAttr::getFromDirectoryAndFilename(
      context, metadataDir, confFile + ".conf", /*excludeFromFilelist=*/true);
  confVerbatimOp->setAttr("output_file", fileAttr);

  return success();
}
/// This will search for a target annotation and remove it from the operation.
/// If the annotation has a filename, it will be returned in the output
/// argument.  If the annotation is missing the filename member, or if more than
/// one matching annotation is attached, it will print an error and return
/// failure.
static LogicalResult removeAnnotationWithFilename(Operation *op,
                                                  StringRef annoClass,
                                                  StringRef &filename) {
  filename = "";
  bool error = false;
  AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
    // If there was a previous error or its not a match, continue.
    if (error || !anno.isClass(annoClass))
      return false;

    // If we have already found a matching annotation, error.
    if (!filename.empty()) {
      op->emitError("more than one ") << annoClass << " annotation attached";
      error = true;
      return false;
    }

    // Get the filename from the annotation.
    auto filenameAttr = anno.getMember<StringAttr>("filename");
    if (!filenameAttr) {
      op->emitError(annoClass) << " requires a filename";
      error = true;
      return false;
    }

    // Require a non-empty filename.
    filename = filenameAttr.getValue();
    if (filename.empty()) {
      op->emitError(annoClass) << " requires a non-empty filename";
      error = true;
      return false;
    }

    return true;
  });

  // If there was a problem above, return failure.
  return failure(error);
}

/// This function collects the name of each module annotated and prints them
/// all as a JSON array.
LogicalResult CreateSiFiveMetadataPass::emitRetimeModulesMetadata() {

  // Circuit level annotation.
  auto *retimeModulesAnnoClass =
      "sifive.enterprise.firrtl.RetimeModulesAnnotation";
  // Per module annotation.
  auto *retimeModuleAnnoClass =
      "sifive.enterprise.firrtl.RetimeModuleAnnotation";

  auto *context = &getContext();
  auto circuitOp = getOperation();

  // Get the filename, removing the annotation from the circuit.
  StringRef filename;
  if (failed(removeAnnotationWithFilename(circuitOp, retimeModulesAnnoClass,
                                          filename)))
    return failure();

  if (filename.empty())
    return success();

  // Create a string buffer for the json data.
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  llvm::json::OStream j(os);

  // The output is a json array with each element a module name.
  unsigned index = 0;
  SmallVector<Attribute> symbols;
  SmallString<3> placeholder;
  j.array([&] {
    for (auto module : circuitOp.getBody()->getOps<FModuleLike>()) {
      // The annotation has no supplemental information, just remove it.
      if (!AnnotationSet::removeAnnotations(module, retimeModuleAnnoClass))
        continue;

      // We use symbol substitution to make sure we output the correct thing
      // when the module goes through renaming.
      j.value(("{{" + Twine(index++) + "}}").str());
      symbols.push_back(SymbolRefAttr::get(context, module.moduleName()));
    }
  });

  // Put the retime information in a verbatim operation.
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());
  auto verbatimOp = builder.create<sv::VerbatimOp>(
      circuitOp.getLoc(), buffer, ValueRange(), builder.getArrayAttr(symbols));
  auto fileAttr = hw::OutputFileAttr::getFromFilename(
      context, filename, /*excludeFromFilelist=*/true);
  verbatimOp->setAttr("output_file", fileAttr);
  return success();
}

/// This function finds all external modules which will need to be generated for
/// the test harness to run.
LogicalResult CreateSiFiveMetadataPass::emitSitestBlackboxMetadata() {
  auto *dutBlackboxAnnoClass =
      "sifive.enterprise.firrtl.SitestBlackBoxAnnotation";
  auto *testBlackboxAnnoClass =
      "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation";

  // Any extmodule with these annotations or one of these ScalaClass classes
  // should be excluded from the blackbox list.
  auto *scalaClassAnnoClass = "sifive.enterprise.firrtl.ScalaClassAnnotation";
  std::array<StringRef, 3> classBlackList = {
      "freechips.rocketchip.util.BlackBoxedROM", "chisel3.shim.CloneModule",
      "sifive.enterprise.grandcentral.MemTap"};
  std::array<StringRef, 5> blackListedAnnos = {
      "firrtl.transforms.BlackBoxInlineAnno",
      "firrtl.transforms.BlackBoxResourceAnno",
      "sifive.enterprise.grandcentral.DataTapsAnnotation",
      "sifive.enterprise.grandcentral.MemTapAnnotation",
      "sifive.enterprise.grandcentral.transforms.SignalMappingAnnotation"};

  auto *context = &getContext();
  auto circuitOp = getOperation();

  // Get the filenames from the annotations.
  StringRef dutFilename, testFilename;
  if (failed(removeAnnotationWithFilename(circuitOp, dutBlackboxAnnoClass,
                                          dutFilename)) ||
      failed(removeAnnotationWithFilename(circuitOp, testBlackboxAnnoClass,
                                          testFilename)))
    return failure();

  // If we don't have either annotation, no need to run this pass.
  if (dutFilename.empty() && testFilename.empty())
    return success();

  // Find all extmodules in the circuit. Check if they are black-listed from
  // being included in the list. If they are not, separate them into two groups
  // depending on if theyre in the DUT or the test harness.
  SmallVector<StringRef> dutModules;
  SmallVector<StringRef> testModules;
  for (auto extModule : circuitOp.getBody()->getOps<FExtModuleOp>()) {
    // If the module doesn't have a defname, then we can't record it properly.
    // Just skip it.
    if (!extModule.defname())
      continue;

    // If its a generated blackbox, skip it.
    AnnotationSet annos(extModule);
    if (llvm::any_of(blackListedAnnos, [&](auto blackListedAnno) {
          return annos.hasAnnotation(blackListedAnno);
        }))
      continue;

    // If its a blacklisted scala class, skip it.
    if (auto scalaAnnoDict = annos.getAnnotation(scalaClassAnnoClass)) {
      Annotation scalaAnno(scalaAnnoDict);
      auto scalaClass = scalaAnno.getMember<StringAttr>("className");
      if (scalaClass &&
          llvm::is_contained(classBlackList, scalaClass.getValue()))
        continue;
    }

    // Record the defname of the module.
    if (dutModuleSet.contains(extModule)) {
      dutModules.push_back(*extModule.defname());
    } else {
      testModules.push_back(*extModule.defname());
    }
  }

  // This is a helper to create the verbatim output operation.
  auto createOutput = [&](SmallVectorImpl<StringRef> &names,
                          StringRef filename) {
    if (filename.empty())
      return;

    // Sort and remove duplicates.
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());

    // The output is a json array with each element a module name. The
    // defname of a module can't change so we can output them verbatim.
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    llvm::json::OStream j(os);
    j.array([&] {
      for (auto &name : names)
        j.value(name);
    });

    auto *body = circuitOp.getBody();
    // Put the information in a verbatim operation.
    auto builder = OpBuilder::atBlockEnd(body);
    auto verbatimOp =
        builder.create<sv::VerbatimOp>(circuitOp.getLoc(), buffer);
    auto fileAttr = hw::OutputFileAttr::getFromFilename(
        context, filename, /*excludeFromFilelist=*/true);
    verbatimOp->setAttr("output_file", fileAttr);
  };

  createOutput(testModules, testFilename);
  createOutput(dutModules, dutFilename);
  return success();
}

void CreateSiFiveMetadataPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  // We need this for SV verbatim and HW attributes.
  registry.insert<hw::HWDialect, sv::SVDialect>();
}

void CreateSiFiveMetadataPass::runOnOperation() {
  auto circuitOp = getOperation();
  auto *body = circuitOp.getBody();

  // Find the device under test and create a set of all modules underneath it.
  auto it = llvm::find_if(*body, [&](Operation &op) -> bool {
    return AnnotationSet(&op).hasAnnotation(dutAnnoClass);
  });
  if (it != body->end()) {
    dutMod = dyn_cast<FModuleOp>(*it);
    auto instanceGraph = getAnalysis<InstanceGraph>();
    auto *node = instanceGraph.lookup(&(*it));
    llvm::for_each(llvm::depth_first(node), [&](InstanceGraphNode *node) {
      dutModuleSet.insert(node->getModule());
    });
  }
  if (failed(emitRetimeModulesMetadata()) ||
      failed(emitSitestBlackboxMetadata()) || failed(emitMemoryMetadata()))
    return signalPassFailure();

  // This pass does not modify the hierarchy.
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createCreateSiFiveMetadataPass() {
  return std::make_unique<CreateSiFiveMetadataPass>();
}
