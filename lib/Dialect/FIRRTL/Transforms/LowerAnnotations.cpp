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
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Diagnostics.h"

using namespace circt;
using namespace firrtl;

namespace {

struct BaseUnion {
  Operation *op;
  size_t portNum;
  unsigned fieldIdx;
  BaseUnion(Operation *op) : op(op), portNum(~0UL), fieldIdx(0) {}
  BaseUnion(Operation *mod, size_t portNum)
      : op(mod), portNum(portNum), fieldIdx(0) {}
  BaseUnion() : op(nullptr), portNum(~0), fieldIdx(0) {}
  operator bool() const { return op != nullptr; }

  bool isPort() const { return op && portNum != ~0UL; }
  bool isInstance() const { return op && isa<InstanceOp>(op); }
  FIRRTLType getType() const {
    if (!op)
      return FIRRTLType();
    if (portNum != ~0UL) {
      if (isa<FModuleOp, FExtModuleOp>(op))
        return getModulePortType(op, portNum).getSubTypeByFieldID(fieldIdx);
      if (isa<MemOp, InstanceOp>(op))
        return op->getResult(portNum)
            .getType()
            .cast<FIRRTLType>()
            .getSubTypeByFieldID(fieldIdx);
      llvm_unreachable("Unknown port instruction");
    }
    if (op->getNumResults() == 0)
      return FIRRTLType();
    return op->getResult(0).getType().cast<FIRRTLType>().getSubTypeByFieldID(
        fieldIdx);
  }
};

struct AnnoPathValue {
  SmallVector<InstanceOp> instances;
  BaseUnion ref;

  AnnoPathValue() = default;
  AnnoPathValue(CircuitOp op) : ref(op) {}
  AnnoPathValue(Operation *op) : ref(op) {}
  AnnoPathValue(const SmallVectorImpl<InstanceOp> &insts, BaseUnion b)
      : instances(insts.begin(), insts.end()), ref(b) {}

  bool isLocal() const { return instances.empty(); }
  template <typename T>
  bool isOpOfType() const {
    if (!ref || ref.isPort())
      return false;
    return isa<T>(ref.op);
  }
};

} // namespace

static bool hasName(StringRef name, Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<InstanceOp, MemOp, NodeOp, RegOp, RegResetOp, WireOp, CMemOp,
            SMemOp, MemoryPortOp>([&](auto nop) {
        if (nop.name() == name)
          return true;
        return false;
      })
      .Default([](auto &) { return false; });
}

static BaseUnion findNamedThing(StringRef name, Operation *op) {
  BaseUnion retval;
  auto nameChecker = [name, &retval](Operation *op) -> WalkResult {
    if (isa<FModuleOp, FExtModuleOp>(op)) {
      // Check the ports.
      auto ports = getModulePortInfo(op);
      for (size_t i = 0, e = ports.size(); i != e; ++i)
        if (ports[i].name.getValue() == name) {
          retval = BaseUnion{op, i};
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    }
    if (hasName(name, op)) {
      retval = BaseUnion{op};
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  op->walk(nameChecker);
  return retval;
}

static void addNamedAttr(Operation *op, StringRef name, StringAttr value) {
  op->setAttr(name, value);
}

static void addNamedAttr(Operation *op, StringRef name) {
  op->setAttr(name, BoolAttr::get(op->getContext(), true));
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
  op->setAttr(getAnnotationAttrName(),
              ArrayAttr::get(op->getContext(), newAnnos));
}

// Returns remainder of path if circuit is the correct circuit
LogicalResult parseAndCheckCircuit(StringRef &path, CircuitOp circuit) {
  if (path.startswith("~"))
    path = path.drop_front();

  // Any non-trivial target must start with a circuit name.
  StringRef name;
  std::tie(name, path) = path.split('|');
  // ~ or ~Foo
  if (name.empty() || name == circuit.name())
    return success();
  return circuit.emitError("circuit name '")
         << circuit.name() << "' doesn't match annotation '" << path << "'";
}

// Returns remainder of path if circuit is the correct circuit
LogicalResult parseAndCheckModule(StringRef &path, Operation *&module,
                                  CircuitOp circuit, SymbolTable &modules) {
  StringRef name;
  if (path.contains('/'))
    std::tie(name, path) = path.split('/');
  else if (path.contains('>'))
    std::tie(name, path) = path.split('>');
  else {
    name = path;
    path = "";
  }
  if (name.empty())
    return circuit.emitError("Cannot decode module in '") << path << "'";

  module = modules.lookup(name);
  if (!module || !isa<FModuleOp, FExtModuleOp>(module))
    return circuit.emitError("cannot find module '")
           << name << "' in annotation";
  return success();
}

// Returns remainder of path if circuit is the correct circuit
LogicalResult parseAndCheckInstance(StringRef &path, Operation *&module,
                                    InstanceOp &inst, CircuitOp circuit,
                                    SymbolTable &modules) {
  StringRef name;
  if (path.contains(':'))
    std::tie(name, path) = path.split(':');
  else
    return success();

  auto resolved = findNamedThing(name, module);
  if (!resolved.isInstance())
    return circuit.emitError("cannot find instance '")
           << name << "' in '" << getModuleName(module) << "'";
  if (parseAndCheckModule(path, module, circuit, modules).failed())
    return failure();
  inst = cast<InstanceOp>(resolved.op);
  if (module != inst.getReferencedModule())
    return circuit.emitError("path module '")
           << getModuleName(module)
           << "' doesn't match instance module referenced in '" << inst.name()
           << "'";
  return success();
}

// Some types have been expanded so the first layer of aggregate path is
// a return value.
LogicalResult updateExpandedPort(StringRef field, BaseUnion &entity) {
  if (auto mem = dyn_cast<MemOp>(entity.op))
    for (size_t p = 0, pe = mem.portNames().size(); p < pe; ++p)
      if (mem.getPortNameStr(p) == field) {
        entity.portNum = p;
        return success();
      }
  if (auto inst = dyn_cast<InstanceOp>(entity.op))
    for (size_t p = 0, pe = inst.getNumResults(); p < pe; ++p)
      if (inst.getPortNameStr(p) == field) {
        entity.portNum = p;
        return success();
      }
  entity.op->emitError("Cannot find port with name ") << field;
  return failure();
}

LogicalResult updateStruct(StringRef field, BaseUnion &entity) {
  // The first field for some ops refers to expanded return values.
  if (isa<MemOp, InstanceOp>(entity.op) && entity.portNum == ~0)
    return updateExpandedPort(field, entity);

  auto bundle = entity.getType().dyn_cast<BundleType>();
  if (!bundle)
    return entity.op->emitError("field access '")
           << field << "' into non-bundle type '" << bundle << "'";
  if (auto idx = bundle.getElementIndex(field)) {
    entity.fieldIdx += bundle.getFieldID(*idx);
    return success();
  }
  return entity.op->emitError("cannot resolve field '")
         << field << "' in subtype '" << bundle << "'";
}

LogicalResult updateArray(unsigned index, BaseUnion &entity) {
  auto vec = entity.getType().dyn_cast<FVectorType>();
  if (!vec)
    return entity.op->emitError("index access '")
           << index << "' into non-vector type '" << vec << "'";
  entity.fieldIdx += vec.getFieldID(index);
  return success();
}

LogicalResult parseNextField(StringRef &path, BaseUnion &entity) {
  if (path.empty())
    return entity.op->emitError("empty string for aggregates");
  if (path[0] == '.') {
    path = path.drop_front();
    StringRef field = path.take_front(path.find_first_of("[."));
    path = path.drop_front(field.size());
    return updateStruct(field, entity);
  } else if (path[0] == '[') {
    path = path.drop_front();
    StringRef index = path.take_front(path.find_first_of(']'));
    path = path.drop_front(index.size() + 1);
    unsigned idxNum;
    if (index.getAsInteger(10, idxNum))
      return entity.op->emitError("non-integer array index");
    return updateArray(idxNum, entity);
  } else {
    return entity.op->emitError("Unknown aggregate specifier in '")
           << path << "'";
  }
}

LogicalResult parseAndCheckName(StringRef &path, BaseUnion &entity,
                                Operation *module) {
  StringRef name = path.take_until([](char c) { return c == '.' || c == '['; });
  path = path.drop_front(name.size());
  if ((entity = findNamedThing(name, module)))
    return success();
  return failure();
}

////////////////////////////////////////////////////////////////////////////////
// Standard Utility resolvers and appliers
////////////////////////////////////////////////////////////////////////////////

static Optional<AnnoPathValue> noResolve(DictionaryAttr anno, CircuitOp circuit,
                                         SymbolTable &modules) {
  return AnnoPathValue(circuit);
}

static LogicalResult ignoreAnno(AnnoPathValue target, DictionaryAttr anno) {
  return success();
}

static Optional<AnnoPathValue> stdResolveImpl(StringRef path, CircuitOp circuit,
                                              SymbolTable &modules) {
  if (parseAndCheckCircuit(path, circuit).failed())
    return {};
  if (path.empty())
    return AnnoPathValue(circuit);

  Operation *rootMod;
  if (parseAndCheckModule(path, rootMod, circuit, modules).failed())
    return {};
  if (path.empty())
    return AnnoPathValue(rootMod);

  Operation *curMod = rootMod;
  SmallVector<InstanceOp> stuff;
  while (true) {
    InstanceOp inst;
    if (parseAndCheckInstance(path, curMod, inst, circuit, modules).failed())
      return {};
    if (!inst)
      break;
    stuff.push_back(inst);
  }

  BaseUnion entity;
  // The last instance in a path may be the name
  if (path.empty()) {
    entity.op = stuff.back();
    stuff.pop_back();
  } else if (parseAndCheckName(path, entity, curMod).failed())
    return {};

  while (!path.empty()) {
    if (parseNextField(path, entity).failed())
      return {};
  }

  return AnnoPathValue{stuff, entity};
}

static Optional<AnnoPathValue>
stdResolve(DictionaryAttr anno, CircuitOp circuit, SymbolTable &modules) {
  auto target = anno.getNamed("target");
  if (!target) {
    circuit.emitError("No target field in annotation") << anno;
    return {};
  }
  return stdResolveImpl(target->second.cast<StringAttr>().getValue(), circuit,
                        modules);
}

static Optional<AnnoPathValue>
tryResolve(DictionaryAttr anno, CircuitOp circuit, SymbolTable &modules) {
  anno.dump();
  auto target = anno.getNamed("target");
  if (target)
    return stdResolveImpl(target->second.cast<StringAttr>().getValue(), circuit,
                          modules);
  return AnnoPathValue(circuit);
}

////////////////////////////////////////////////////////////////////////////////
// Specific Appliers
////////////////////////////////////////////////////////////////////////////////

LogicalResult applyDirFileNormalizeToCircuit(AnnoPathValue target,
                                             DictionaryAttr anno) {
  if (!target.isOpOfType<CircuitOp>())
    return failure();
  SmallVector<NamedAttribute> newAnnoAttrs;

  // Check all values
  for (auto &na : anno) {
    if (na.first == "class" || na.first == "filename" ||
        na.first == "directory") {
      newAnnoAttrs.push_back(na);
      continue;
    }
    if (na.first == "target")
      continue;
    if (na.first == "dirname" || na.first == "dir" || na.first == "targetDir") {
      newAnnoAttrs.emplace_back(
          Identifier::get("directory", target.ref.op->getContext()), na.second);
      continue;
    }
    if (na.first == "resourceFileName") {
      newAnnoAttrs.emplace_back(
          Identifier::get("filename", target.ref.op->getContext()), na.second);
      continue;
    }
    target.ref.op->emitError("Unknown file or directory field '")
        << na.first << "'";
    return failure();
  }
  addAnnotation(target.ref.op,
                DictionaryAttr::get(target.ref.op->getContext(), newAnnoAttrs));
  return success();
}

LogicalResult applyWithoutTargetToTarget(AnnoPathValue target,
                                         DictionaryAttr anno,
                                         bool allowNonLocal) {
  if (!allowNonLocal && !target.isLocal())
    return failure();
  SmallVector<NamedAttribute> newAnnoAttrs;
  for (auto &na : anno)
    if (na.first != "target")
      newAnnoAttrs.push_back(na);
  addAnnotation(target.ref.op,
                DictionaryAttr::get(target.ref.op->getContext(), newAnnoAttrs));
  return success();
}

template <bool allowNonLocal = false>
LogicalResult applyWithoutTarget(AnnoPathValue target, DictionaryAttr anno) {
  return applyWithoutTargetToTarget(target, anno, allowNonLocal);
}

template <bool allowNonLocal = false>
LogicalResult applyWithoutTargetToModule(AnnoPathValue target,
                                         DictionaryAttr anno) {
  if (!target.isOpOfType<FModuleOp>() && !target.isOpOfType<FExtModuleOp>())
    return failure();
  return applyWithoutTargetToTarget(target, anno, allowNonLocal);
}

template <bool allowNonLocal = false>
LogicalResult applyWithoutTargetToCircuit(AnnoPathValue target,
                                          DictionaryAttr anno) {
  if (!target.isOpOfType<CircuitOp>())
    return failure();
  return applyWithoutTargetToTarget(target, anno, allowNonLocal);
}

template <bool allowNonLocal = false>
LogicalResult applyWithoutTargetToMem(AnnoPathValue target,
                                      DictionaryAttr anno) {
  if (!target.isOpOfType<MemOp>())
    return failure();
  return applyWithoutTargetToTarget(target, anno, allowNonLocal);
}

// Fix up a path that contains missing modules and place the annotation on
// the memory op.
static Optional<AnnoPathValue> seqMemInstanceResolve(DictionaryAttr anno,
                                                     CircuitOp circuit,
                                                     SymbolTable &modules) {
  auto target = anno.getNamed("target");
  if (!target) {
    circuit.emitError("No target field in annotation") << anno;
    return {};
  }
  auto path = target->second.cast<StringAttr>().getValue();
  //         path.instances.pop_back();
  //    path.name = path.instances.back().first;
  //    path.instances.pop_back();
  return stdResolveImpl(path, circuit, modules);
}

LogicalResult applyDontTouch(AnnoPathValue target, DictionaryAttr anno) {
  addNamedAttr(target.ref.op, "firrtl.DoNotTouch");
  return success();
}

LogicalResult applyGrandCentralDataTaps(AnnoPathValue target,
                                        DictionaryAttr anno) {
  addNamedAttr(target.ref.op, "firrtl.DoNotTouch");
  // TODO: port scatter logic in FIRAnnotations.cpp
  return applyWithoutTargetToCircuit(target, anno);
}

LogicalResult applyGrandCentralMemTaps(AnnoPathValue target,
                                       DictionaryAttr anno) {
  addNamedAttr(target.ref.op, "firrtl.DoNotTouch");
  // TODO: port scatter logic in FIRAnnotations.cpp
  return applyWithoutTargetToCircuit(target, anno);
}

LogicalResult applyGrandCentralView(AnnoPathValue target, DictionaryAttr anno) {
  addNamedAttr(target.ref.op, "firrtl.DoNotTouch");
  // TODO: port scatter logic in FIRAnnotations.cpp
  return applyWithoutTargetToCircuit(target, anno);
}

////////////////////////////////////////////////////////////////////////////////
// Driving table
////////////////////////////////////////////////////////////////////////////////

namespace {
struct AnnoRecord {
  llvm::function_ref<Optional<AnnoPathValue>(DictionaryAttr, CircuitOp,
                                             SymbolTable &)>
      resolver;
  llvm::function_ref<LogicalResult(AnnoPathValue, DictionaryAttr)> applier;
};
}; // namespace

static const llvm::StringMap<AnnoRecord> annotationRecords{
    /*
      {"chisel3.aop.injecting.InjectStatement", noParse, noResolve,
      ignoreAnno},
      {"chisel3.util.experimental.ForceNameAnnotation", noParse, noResolve,
       ignoreAnno},
      {"sifive.enterprise.firrtl.DFTTestModeEnableAnnotation", noParse,
      noResolve, ignoreAnno},
  */
    // Dropped annotations.
    {"firrtl.EmitCircuitAnnotation", {noResolve, ignoreAnno}},
    {"logger.LogLevelAnnotation", {noResolve, ignoreAnno}},
    {"firrtl.transforms.DedupedResult", {noResolve, ignoreAnno}},

    // Targetless Annotations.
    {"sifive.enterprise.firrtl.ElaborationArtefactsDirectory",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.MetadataDirAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"firrtl.passes.memlib.InferReadWriteAnnotation$",
     {noResolve, applyWithoutTargetToCircuit<>}},

    // Simple Annotations
    {"sifive.enterprise.firrtl.TestHarnessPathAnnotation",
     {noResolve, applyWithoutTargetToCircuit<>}},
    {"sifive.enterprise.grandcentral.phases.SimulationConfigPathPrefix",
     {noResolve, applyWithoutTargetToCircuit<>}},
    {"sifive.enterprise.firrtl.ScalaClassAnnotation",
     {stdResolve, applyWithoutTargetToModule<>}},
    {"firrtl.transforms.NoDedupAnnotation",
     {stdResolve, applyWithoutTargetToModule<>}},
    {"freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation",
     {stdResolve, applyWithoutTargetToModule<>}},
    {"firrtl.passes.InlineAnnotation",
     {stdResolve, applyWithoutTargetToModule<>}},
    {"sifive.enterprise.firrtl.DontObfuscateModuleAnnotation",
     {stdResolve, applyWithoutTargetToModule<>}},
    {"firrtl.transforms.BlackBoxInlineAnno",
     {stdResolve, applyWithoutTargetToModule<>}},
    {"freechips.rocketchip.linting.rule.DesiredNameAnnotation",
     {stdResolve, applyWithoutTargetToModule<true>}},
    {"sifive.enterprise.firrtl.MarkDUTAnnotation",
     {stdResolve, applyWithoutTargetToModule<>}},
    {"sifive.enterprise.firrtl.FileListAnnotation",
     {stdResolve, applyWithoutTargetToModule<>}},
    {"sifive.enterprise.grandcentral.PrefixInterfacesAnnotation",
     {noResolve, applyWithoutTargetToCircuit<>}},
    {"sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
     {stdResolve, applyWithoutTargetToModule<>}},
    {"firrtl.passes.memlib.ReplSeqMemAnnotation",
     {noResolve, applyWithoutTargetToCircuit<>}},

    // Directory or Filename Annotations
    {"sifive.enterprise.firrtl.ExtractCoverageAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.ExtractAssertionsAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.ExtractAssumptionsAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.TestBenchDirAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"firrtl.transforms.BlackBoxTargetDirAnno",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"firrtl.transforms.BlackBoxResourceFileNameAnno",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.RetimeModulesAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.CoverPropReportAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.firrtl.ModuleHierarchyAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.grandcentral.SubCircuitDirAnnotation",
     {noResolve, applyDirFileNormalizeToCircuit}},
    {"sifive.enterprise.grandcentral.phases.SubCircuitsTargetDirectory",
     {noResolve, applyDirFileNormalizeToCircuit}},

    // Complex Annotations
    {"sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation",
     {/*stdParse,*/ seqMemInstanceResolve, applyWithoutTargetToMem<>}},
    {"firrtl.transforms.DontTouchAnnotation", {stdResolve, applyDontTouch}},
    {"sifive.enterprise.grandcentral.DataTapsAnnotation",
     {noResolve, applyGrandCentralDataTaps}},
    {"sifive.enterprise.grandcentral.MemTapAnnotation",
     {noResolve, applyGrandCentralMemTaps}},
    {"sifive.enterprise.grandcentral.ViewAnnotation",
     {noResolve, applyGrandCentralView}},

    // Testing Annotation
    {"circt.test", {stdResolve, applyWithoutTarget<>}},
    {"circt.testNT", {noResolve, applyWithoutTarget<>}},
    {"circt.missing", {tryResolve, applyWithoutTarget<>}},

};

static const AnnoRecord *getAnnotationHandler(StringRef annoStr) {
  auto ii = annotationRecords.find(annoStr);
  if (ii != annotationRecords.end())
    return &ii->second;
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
                                SymbolTable &modules);

  bool ignoreUnhandledAnno = false;
  bool ignoreClasslessAnno = false;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void LowerAnnotationsPass::runOnOperation() {
  CircuitOp circuit = getOperation();
  SymbolTable modules(circuit);
  ArrayAttr attrs = circuit.annotations();
  circuit.annotationsAttr(ArrayAttr::get(circuit.getContext(), {}));
  size_t numFailures = 0;
  for (auto attr : attrs)
    if (applyAnnotation(attr.cast<DictionaryAttr>(), circuit, modules).failed())
      ++numFailures;
}

LogicalResult LowerAnnotationsPass::applyAnnotation(DictionaryAttr anno,
                                                    CircuitOp circuit,
                                                    SymbolTable &modules) {
  StringRef annoClassVal;
  if (auto annoClass = anno.getNamed("class"))
    annoClassVal = annoClass->second.cast<StringAttr>().getValue();
  else if (ignoreClasslessAnno)
    annoClassVal = "circt.missing";
  else
    return circuit.emitError("Annotation without a class: ") << anno;

  auto record = getAnnotationHandler(annoClassVal);
  if (!record) {
    if (ignoreUnhandledAnno)
      record = getAnnotationHandler("circt.missing");
    else
      return circuit.emitWarning("Unhandled annotation: ") << anno;
  }

  auto target = record->resolver(anno, circuit, modules);
  if (!target)
    return circuit.emitError("Unable to resolve target of annotation: ")
           << anno;
  if (record->applier(*target, anno).failed())
    return circuit.emitError("Unable to apply annotation: ") << anno;
  return success();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLAnnotationsPass(
    bool ignoreUnhandledAnnotations, bool ignoreClasslessAnnotations) {
  auto pass = std::make_unique<LowerAnnotationsPass>();
  pass->ignoreUnhandledAnno = ignoreUnhandledAnnotations;
  pass->ignoreClasslessAnno = ignoreClasslessAnnotations;
  return pass;
}
