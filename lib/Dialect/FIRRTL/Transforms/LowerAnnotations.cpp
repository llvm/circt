//===- LowerAnnotations.cpp - Lower Annotations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerAnnotations pass.  This pass processes FIRRTL
// annotations, rewriting them, scattering them, and dealing with non-local
// annotations.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"

using namespace circt;
using namespace firrtl;

namespace {

struct aggAccess {
  bool isIndex;
  StringRef val;
};

struct AnnoPathStr {
  StringRef circuit;
  StringRef module;
  SmallVector<std::pair<StringRef, StringRef>> instances;
  StringRef name;
  SmallVector<aggAccess> agg;
  bool isLocal() const { return instances.empty(); }
  void dump() const {
    llvm::errs() << "APS{c: " << circuit << ", m: " << module << ", i: {";
    for (auto i : instances)
      llvm::errs() << "(" << i.first << ", " << i.second << "), ";
    llvm::errs() << "}, n:" << name << ", a: ";
    for (auto a : agg)
      if (a.isIndex)
        llvm::errs() << '[' << a.val << ']';
      else
        llvm::errs() << '.' << a.val;
    llvm::errs() << "}\n";
  }
};

struct BaseUnion {
  Operation *op;
  size_t portNum;
  BaseUnion(Operation *op) : op(op), portNum(~0UL) {}
  BaseUnion(Operation *mod, size_t portNum) : op(mod), portNum(portNum) {}
  BaseUnion() : op(nullptr), portNum(~0) {}
  operator bool() const { return op != nullptr; }
  bool isPort() const { return portNum != ~0UL; }
  bool isInstance() const { return op && isa<InstanceOp>(op); }
  Type getType() const {
    if (!op)
      return Type();
    if (portNum != ~0UL)
      return getModulePortType(op, portNum);
    return op->getResult(0).getType();
  }
};

struct AnnoPathValue {
  SmallVector<InstanceOp> instances;
  BaseUnion ref;
  unsigned fieldIdx;
  bool isLocal() const { return instances.empty(); }
  template <typename T>
  bool isOpOfType() const {
    if (!ref || ref.isPort())
      return false;
    return isa<T>(ref.op);
  }
};

struct AnnoRecord {
  const char *name;
  llvm::function_ref<Optional<AnnoPathStr>(DictionaryAttr, CircuitOp)>
      path_parser;
  llvm::function_ref<Optional<AnnoPathValue>(AnnoPathStr, CircuitOp,
                                             SymbolTable)>
      path_resolver;
  llvm::function_ref<LogicalResult(AnnoPathValue, DictionaryAttr)> anno_applier;
};

} // namespace

static bool hasName(StringRef name, Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<InstanceOp, MemOp, NodeOp, RegOp, RegResetOp, WireOp>(
          [&](auto nop) {
            if (nop.name() == name)
              return true;
            return false;
          })
      .Default([](auto &) { return false; });
}

static BaseUnion findNamedThing(StringRef name, Operation *op) {
  // First check ports
  auto ports = getModulePortInfo(op);
  for (size_t i = 0, e = ports.size(); i != e; ++i)
    if (ports[i].name.getValue() == name)
      return BaseUnion{op, i};

  // Second, check wires, nodes, registers, and instances.
  if (auto mod = dyn_cast<FModuleOp>(op))
    for (auto &oper : mod.getBodyBlock()->getOperations())
      if (hasName(name, &oper))
        return BaseUnion{&oper};

  return nullptr;
}

static Optional<unsigned> resolveFieldIdx(SmallVectorImpl<aggAccess> &fields,
                                          Type) {
  return 0;
}

static void addNamedAttr(Operation *op, StringRef name, StringAttr value) {
  op->setAttr(name, value);
}

static ArrayAttr getAnnotationsFrom(Operation *op) {
  if (auto annots = op->getAttrOfType<ArrayAttr>(getAnnotationAttrName()))
    return annots;
  return ArrayAttr::get(op->getContext(), {});
}

static void addAnnotation(Operation *op, Annotation anno) {
  SmallVector<Attribute> newAnnos;
  for (auto old : getAnnotationsFrom(op))
    newAnnos.push_back(old);
    newAnnos.push_back(anno.getDict());
    op->setAttr(getAnnotationAttrName(), ArrayAttr::get(op->getContext(), newAnnos));
}

////////////////////////////////////////////////////////////////////////////////
// Standard Utility resolvers and appliers
////////////////////////////////////////////////////////////////////////////////

static Optional<AnnoPathValue> noResolve(AnnoPathStr path, CircuitOp circuit,
                                         SymbolTable modules) {
  AnnoPathValue retval;
  retval.ref.op = circuit;
  return retval;
}

static LogicalResult ignoreAnno(AnnoPathValue target, DictionaryAttr anno) {
  return success();
}

static Optional<AnnoPathStr> noParse(DictionaryAttr anno, CircuitOp circuit) {
  return AnnoPathStr{};
}

static Optional<AnnoPathStr> stdParse(DictionaryAttr anno, CircuitOp circuit) {
  auto target = anno.getNamed("target");
  if (!target || !target->second.isa<StringAttr>())
    return {};

  StringRef path = target->second.cast<StringAttr>().getValue();

  AnnoPathStr retval;
  if (path.startswith("~"))
    path = path.drop_front();
  // Any non-trivial target must start with a circuit name.
  std::tie(retval.circuit, path) = path.split('|');
  if (retval.circuit.empty())
    retval.circuit = circuit.name();

  StringRef nameRef;
  std::tie(path, nameRef) = path.rsplit('>');

  SmallVector<StringRef> parts;
  path.split(parts, '/');
  retval.module = parts[0]; // There must be a module.

  for (int i = 1, e = parts.size(); i < e; ++i) {
    StringRef inst, mod;
    std::tie(inst, mod) = parts[i].split(':');
    retval.instances.emplace_back(inst, mod);
  }

  retval.name = nameRef.take_front(nameRef.find_first_of("[."));
  nameRef = nameRef.drop_front(retval.name.size());

  while (!nameRef.empty()) {
    if (nameRef[0] == '.') {
      nameRef = nameRef.drop_front();
      StringRef field = nameRef.take_front(nameRef.find_first_of("[."));
      nameRef = nameRef.drop_front(field.size());
      retval.agg.push_back({true, field});
    } else if (nameRef[0] == '[') {
      nameRef = nameRef.drop_front();
      StringRef index = nameRef.take_front(nameRef.find_first_of(']'));
      nameRef = nameRef.drop_front(index.size() + 1);
      retval.agg.push_back({false, index});
    } else {
      llvm_unreachable("invalid annotation aggregate specifier");
    }
  }
  return retval;
}

static Optional<AnnoPathValue> stdResolve(AnnoPathStr path, CircuitOp circuit,
                                          SymbolTable modules) {
  AnnoPathValue retval;
  if (path.circuit != circuit.name()) {
    circuit.emitError("circuit name mismatch in annotation");
    return {};
  }
  if (path.module.empty()) {
    retval.ref = BaseUnion{circuit};
    return retval; // Circuit only annotation.
  }
  auto curModule = modules.lookup(path.module);
  if (!curModule) {
    circuit.emitError("cannot find module '") << path.module << "'";
    return {};
  }
  if (path.instances.empty() && path.name.empty() && path.agg.empty()) {
    retval.ref = BaseUnion{curModule};
    return retval; // Module only annotation
  }
  for (auto inst : path.instances) {
    auto resolved = findNamedThing(inst.first, curModule);
    if (!resolved.isInstance()) {
      path.dump();
      circuit.emitError("cannot find instance '")
          << inst.first << "' in '" << inst.second << "'";
      return {};
    }
    auto resolvedInst = cast<InstanceOp>(resolved.op);
    retval.instances.push_back(resolvedInst);
    curModule = modules.lookup(inst.second);
    if (!curModule) {
      circuit.emitError("module '")
          << inst.second << "' in instance path doesn't exist";
      return {};
    }
    if (curModule != resolvedInst.getReferencedModule()) {
      path.dump();
      circuit.emitError("path module '")
          << getModuleName(curModule)
          << "' doesn't match instance module reference in '"
          << resolvedInst.name() << "'";
      return {};
    }
  }
  if (!path.name.empty()) {
    retval.ref = findNamedThing(path.name, curModule);
    if (!retval.ref) {
      path.dump();
      circuit.emitError("cannot find a thing named '") << path.name << "'";
      return {};
    }
    auto field = resolveFieldIdx(path.agg, retval.ref.getType());
    if (!field) {
      circuit.emitError("cannot resolve field accesses");
      return {};
    }
    retval.fieldIdx = *field;
  }
  return retval;
}

////////////////////////////////////////////////////////////////////////////////
// Specific Appliers
////////////////////////////////////////////////////////////////////////////////

LogicalResult applyExtractAssertions(AnnoPathValue target,
                                     DictionaryAttr anno) {
  auto dir = anno.getNamed("directory");
  auto file = anno.getNamed("filename");
  if (!target.isOpOfType<CircuitOp>())
    return failure();
  if (file)
    addNamedAttr(target.ref.op, "firrtl.ExtractAssertions.filename",
                 file->second.cast<StringAttr>());
  if (dir)
    addNamedAttr(target.ref.op, "firrtl.ExtractAssertions.directory",
                 dir->second.cast<StringAttr>());
  return success();
}

LogicalResult applyExtractCoverage(AnnoPathValue target, DictionaryAttr anno) {
  auto dir = anno.getNamed("directory");
  auto file = anno.getNamed("filename");
  if (!target.isOpOfType<CircuitOp>())
    return failure();
  if (file)
    addNamedAttr(target.ref.op, "firrtl.ExtractCoverageAnnotation.filename",
                 file->second.cast<StringAttr>());
  if (dir)
    addNamedAttr(target.ref.op, "firrtl.ExtractCoverageAnnotation.directory",
                 dir->second.cast<StringAttr>());
  return success();
}

LogicalResult applyExtractAssumptions(AnnoPathValue target, DictionaryAttr anno) {
  auto dir = anno.getNamed("directory");
  auto file = anno.getNamed("filename");
  if (!target.isOpOfType<CircuitOp>())
    return failure();
  if (file)
    addNamedAttr(target.ref.op, "firrtl.ExtractAssumptionsAnnotation.filename",
                 file->second.cast<StringAttr>());
  if (dir)
    addNamedAttr(target.ref.op, "firrtl.ExtractAssumptionsAnnotation.directory",
                 dir->second.cast<StringAttr>());
  return success();
}

LogicalResult applyTestBenchDir(AnnoPathValue target, DictionaryAttr anno) {
  auto dir = anno.getNamed("dirname");
  if (!target.isOpOfType<CircuitOp>())
    return failure();
  if (dir)
    addNamedAttr(target.ref.op, "firrtl.TestBench.directory",
                 dir->second.cast<StringAttr>());
  return success();
}

LogicalResult applyBlackBoxTargetDir(AnnoPathValue target, DictionaryAttr anno) {
  auto dir = anno.getNamed("targetDir");
  if (!target.isOpOfType<CircuitOp>())
    return failure();
  if (dir)
    addNamedAttr(target.ref.op, "firrtl.BlackBoxTarget.directory",
                 dir->second.cast<StringAttr>());
  return success();
}

LogicalResult applyWithoutTargetToModule(AnnoPathValue target, DictionaryAttr anno) {
  if (!target.isOpOfType<FModuleOp>() && !target.isOpOfType<FExtModuleOp>())
    return failure();
  if (!target.isLocal())
    return failure();
  SmallVector<NamedAttribute> newAnnoAttrs;
  for (auto& na : anno)
    if (na.first != "target")
      newAnnoAttrs.push_back(na);
    addAnnotation(target.ref.op, DictionaryAttr::get(target.ref.op->getContext(), newAnnoAttrs));
  return success();
}

LogicalResult applySeqMemInstance(AnnoPathValue target, DictionaryAttr anno) {
  if (!target.isLocal())
    return failure();
  return success();
}

// Fix up a path that contains missing modules and place the annotation on the
// memory op.
static Optional<AnnoPathValue> seqMemInstanceResolve(AnnoPathStr path,
                                                     CircuitOp circuit,
                                                     SymbolTable modules) {
  path.instances.pop_back();
  path.name = path.instances.back().first;
  path.instances.pop_back();
  return stdResolve(path, circuit, modules);
}

  ////////////////////////////////////////////////////////////////////////////////
  // Driving table
  ////////////////////////////////////////////////////////////////////////////////

static const AnnoRecord annotationRecords[] = {
    {"logger.LogLevelAnnotation", noParse, noResolve, ignoreAnno},
    {"firrtl.EmitCircuitAnnotation", noParse, noResolve, ignoreAnno},
    {"firrtl.transforms.DedupedResult", noParse, noResolve, ignoreAnno},
    {"sifive.enterprise.firrtl.RetimeModulesAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.firrtl.CoverPropReportAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation", noParse,
     noResolve, ignoreAnno},
    {"sifive.enterprise.firrtl.ModuleHierarchyAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.firrtl.MetadataDirAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.firrtl.ElaborationArtefactsDirectory", noParse,
     noResolve, ignoreAnno},
    {"freechips.rocketchip.linting.rule.DesiredNameAnnotation", noParse,
     noResolve, ignoreAnno},
    {"sifive.enterprise.grandcentral.phases.SimulationConfigPathPrefix",
     noParse, noResolve, ignoreAnno},
    {"sifive.enterprise.firrtl.TestHarnessPathAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
     noParse, noResolve, ignoreAnno},
    {"sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation", noParse,
     noResolve, ignoreAnno},
    {"sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation", noParse,
     noResolve, ignoreAnno},
    {"sifive.enterprise.firrtl.SitestBlackBoxAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.grandcentral.SubCircuitDirAnnotation", noParse,
     noResolve, ignoreAnno},
    {"sifive.enterprise.grandcentral.PrefixInterfacesAnnotation", noParse,
     noResolve, ignoreAnno},
    {"sifive.enterprise.grandcentral.phases.SubCircuitsTargetDirectory",
     noParse, noResolve, ignoreAnno},
    {"chisel3.aop.injecting.InjectStatement", noParse, noResolve, ignoreAnno},
    {"firrtl.transforms.BlackBoxInlineAnno", noParse, noResolve, ignoreAnno},
    {"sifive.enterprise.grandcentral.DataTapsAnnotation", noParse, noResolve,
     ignoreAnno},
    {"firrtl.transforms.DontTouchAnnotation", noParse, noResolve, ignoreAnno},
    {"sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", noParse,
     noResolve, ignoreAnno},
    {"sifive.enterprise.firrtl.MarkDUTAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.firrtl.FileListAnnotation", noParse, noResolve,
     ignoreAnno},
    {"chisel3.util.experimental.ForceNameAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.firrtl.DFTTestModeEnableAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.grandcentral.ViewAnnotation", noParse, noResolve,
     ignoreAnno},
    {"sifive.enterprise.grandcentral.MemTapAnnotation", noParse, noResolve,
     ignoreAnno},
    {"firrtl.passes.memlib.ReplSeqMemAnnotation", noParse, noResolve,
     ignoreAnno},
    {"firrtl.passes.memlib.InferReadWriteAnnotation$", noParse, noResolve,
     ignoreAnno},

    {"sifive.enterprise.firrtl.ScalaClassAnnotation", stdParse, stdResolve,
     applyWithoutTargetToModule},
    {"firrtl.transforms.NoDedupAnnotation", stdParse, stdResolve,
     applyWithoutTargetToModule},
    {"freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation",
     stdParse, stdResolve, applyWithoutTargetToModule},
    {"firrtl.passes.InlineAnnotation", stdParse, stdResolve,
     applyWithoutTargetToModule},
    {"sifive.enterprise.firrtl.DontObfuscateModuleAnnotation", stdParse,
     stdResolve, applyWithoutTargetToModule},

    {"sifive.enterprise.firrtl.ExtractCoverageAnnotation", stdParse, stdResolve,
     applyExtractCoverage},
    {"sifive.enterprise.firrtl.ExtractAssertionsAnnotation", stdParse,
     stdResolve, applyExtractAssertions},
    {"sifive.enterprise.firrtl.ExtractAssumptionsAnnotation", stdParse,
     stdResolve, applyExtractAssumptions},
    {"sifive.enterprise.firrtl.TestBenchDirAnnotation", stdParse, stdResolve,
     applyTestBenchDir},
    {"sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation", stdParse,
     seqMemInstanceResolve, applySeqMemInstance},
    {"firrtl.transforms.BlackBoxTargetDirAnno", stdParse, stdResolve, applyBlackBoxTargetDir},

    };

static const AnnoRecord *getAnnotationHandler(StringRef annoStr) {
  for (auto &a : annotationRecords)
    if (a.name == annoStr)
      return &a;
  return nullptr;
  }

  //===----------------------------------------------------------------------===//
  // Pass Infrastructure
  //===----------------------------------------------------------------------===//

  namespace {
  struct LowerAnnotationsPass
      : public LowerFIRRTLAnnotationsBase<LowerAnnotationsPass> {
    void runOnOperation() override;
    LogicalResult applyAnnotation(DictionaryAttr anno, CircuitOp circuit,
                                  SymbolTable modules);
  };
  } // end anonymous namespace

  // This is the main entrypoint for the lowering pass.
  void LowerAnnotationsPass::runOnOperation() {
    CircuitOp circuit = getOperation();
    SymbolTable modules(circuit);
    ArrayAttr attrs = circuit.annotations();
    for (auto attr : attrs)
      applyAnnotation(attr.cast<DictionaryAttr>(), circuit, modules);
  }

  LogicalResult LowerAnnotationsPass::applyAnnotation(
      DictionaryAttr anno, CircuitOp circuit, SymbolTable modules) {
    auto annoClass = anno.getNamed("class");
    if (!annoClass)
      return circuit.emitError("Annotation without a class: ") << anno;
    auto annoClassVal = annoClass->second.cast<StringAttr>().getValue();
    auto record = getAnnotationHandler(annoClassVal);
    if (!record)
      return circuit.emitError("Unhandled annotation: ") << anno;
    auto path = record->path_parser(anno, circuit);
    if (!path)
      return circuit.emitError("Unable to parse target of annotation: ") << anno;
    auto target = record->path_resolver(*path, circuit, modules);
    if (!target)
      return circuit.emitError("Unable to resolve target of annotation: ")
             << anno;
    if (record->anno_applier(*target, anno).failed())
      return circuit.emitError("Unable to apply annotation: ") << anno;
    return success();
  }

  /// This is the pass constructor.
  std::unique_ptr<mlir::Pass>
  circt::firrtl::createLowerFIRRTLAnnotationsPass() {
    return std::make_unique<LowerAnnotationsPass>();
  }
