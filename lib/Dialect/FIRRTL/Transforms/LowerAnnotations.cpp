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
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringExtras.h"

using namespace circt;
using namespace firrtl;

namespace {

struct fieldOrIndex {
  StringRef name;
  bool isIndex;
};

struct tokenAnnoTarget {
  StringRef circuit;
  SmallVector<std::pair<StringRef, StringRef>> instances;
  StringRef module;
  StringRef name;
  SmallVector<fieldOrIndex> agg;
};

struct AnnoTarget {
  Operation *op;
  size_t portNum;
  unsigned fieldIdx;
  AnnoTarget(Operation *op) : op(op), portNum(~0UL), fieldIdx(0) {}
  AnnoTarget(Operation *mod, size_t portNum)
      : op(mod), portNum(portNum), fieldIdx(0) {}
  AnnoTarget() : op(nullptr), portNum(~0), fieldIdx(0) {}
  operator bool() const { return op != nullptr; }

  bool isPort() const { return op && portNum != ~0UL; }
  bool isInstance() const { return op && isa<InstanceOp>(op); }
  FModuleOp getModule() const {
    if (auto mod = dyn_cast<FModuleOp>(op))
      return mod;
    return op->getParentOfType<FModuleOp>();
  }
  FIRRTLType getType() const {
    if (!op)
      return FIRRTLType();
    if (portNum != ~0UL) {
      if (auto mod = dyn_cast<FModuleLike>(op))
        return mod.portType(portNum).getSubTypeByFieldID(fieldIdx);
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
  AnnoTarget ref;

  AnnoPathValue() = default;
  AnnoPathValue(CircuitOp op) : ref(op) {}
  AnnoPathValue(Operation *op) : ref(op) {}
  AnnoPathValue(const SmallVectorImpl<InstanceOp> &insts, AnnoTarget b)
      : instances(insts.begin(), insts.end()), ref(b) {}

  bool isLocal() const { return instances.empty(); }

  template <typename... T>
  bool isOpOfType() const {
    if (!ref || ref.isPort())
      return false;
    return isa<T...>(ref.op);
  }
};

struct applyState {
  CircuitOp circuit;
  SymbolTable &symTbl;
  llvm::function_ref<void(DictionaryAttr)> addToWorklistFn;
};

} // namespace

static StringRef getName(Operation *op) {
  return TypeSwitch<Operation *, StringRef>(op)
      .Case<InstanceOp, MemOp, NodeOp, RegOp, RegResetOp, WireOp, CombMemOp,
            SeqMemOp, MemoryPortOp>([&](auto nop) { return nop.name(); })
      .Default([](auto &) { return ""; });
}

static bool hasName(StringRef name, Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<InstanceOp, MemOp, NodeOp, RegOp, RegResetOp, WireOp, CombMemOp,
            SeqMemOp, MemoryPortOp>([&](auto nop) {
        if (nop.name() == name)
          return true;
        return false;
      })
      .Default([](auto &) { return false; });
}

static AnnoTarget findNamedThing(StringRef name, Operation *op) {
  AnnoTarget retval;
  auto nameChecker = [name, &retval](Operation *op) -> WalkResult {
    if (auto mod = dyn_cast<FModuleLike>(op)) {
      // Check the ports.
      auto ports = mod.getPorts();
      for (size_t i = 0, e = ports.size(); i != e; ++i)
        if (ports[i].name.getValue() == name) {
          retval = AnnoTarget{op, i};
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    }
    if (hasName(name, op)) {
      retval = AnnoTarget{op};
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  op->walk(nameChecker);
  return retval;
}

static ArrayAttr getAnnotationsFrom(Operation *op) {
  if (auto annots = op->getAttrOfType<ArrayAttr>(getAnnotationAttrName()))
    return annots;
  return ArrayAttr::get(op->getContext(), {});
}

static ArrayAttr appendArrayAttr(ArrayAttr array, Attribute a) {
  if (!array)
    return ArrayAttr::get(a.getContext(), ArrayRef<Attribute>{a});
  SmallVector<Attribute> old(array.begin(), array.end());
  old.push_back(a);
  return ArrayAttr::get(a.getContext(), old);
}

static ArrayAttr replaceArrayAttrElement(ArrayAttr array, size_t elem,
                                         Attribute newVal) {
  SmallVector<Attribute> old(array.begin(), array.end());
  old[elem] = newVal;
  return ArrayAttr::get(array.getContext(), old);
}

static void addAnnotation(AnnoTarget ref, ArrayRef<NamedAttribute> anno) {
  DictionaryAttr annotation;
  if (ref.fieldIdx) {
    SmallVector<NamedAttribute> annoField(anno.begin(), anno.end());
    annoField.emplace_back(
        Identifier::get("circt.fieldID", ref.op->getContext()),
        IntegerAttr::get(
            IntegerType::get(ref.op->getContext(), 32, IntegerType::Signless),
            ref.fieldIdx));
    annotation = DictionaryAttr::get(ref.op->getContext(), annoField);
  } else {
    annotation = DictionaryAttr::get(ref.op->getContext(), anno);
  }

  if (!ref.isPort()) {
    auto newAnno = appendArrayAttr(getAnnotationsFrom(ref.op), annotation);
    ref.op->setAttr(getAnnotationAttrName(), newAnno);
    return;
  }

  auto portAnnoRaw = ref.op->getAttr("portAnnotations");
  ArrayAttr portAnno = portAnnoRaw.dyn_cast_or_null<ArrayAttr>();
  if (!portAnno || portAnno.size() != getNumPorts(ref.op)) {
    SmallVector<Attribute> emptyPortAttr(
        getNumPorts(ref.op), ArrayAttr::get(ref.op->getContext(), {}));
    portAnno = ArrayAttr::get(ref.op->getContext(), emptyPortAttr);
  }
  portAnno = replaceArrayAttrElement(
      portAnno, ref.portNum,
      appendArrayAttr(portAnno[ref.portNum].dyn_cast<ArrayAttr>(), annotation));
  ref.op->setAttr("portAnnotations", portAnno);
}

// Some types have been expanded so the first layer of aggregate path is
// a return value.
static LogicalResult updateExpandedPort(StringRef field, AnnoTarget &entity) {
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

static LogicalResult updateStruct(StringRef field, AnnoTarget &entity) {
  // The first field for some ops refers to expanded return values.
  if (isa<MemOp, InstanceOp>(entity.op) && entity.portNum == ~0UL)
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

static LogicalResult updateArray(StringRef indexStr, AnnoTarget &entity) {
  size_t index;
  if (indexStr.getAsInteger(10, index)) {
    entity.op->emitError("Cannot convert '") << indexStr << "' to an integer";
    return failure();
  }

  auto vec = entity.getType().dyn_cast<FVectorType>();
  if (!vec)
    return entity.op->emitError("index access '")
           << index << "' into non-vector type '" << vec << "'";
  entity.fieldIdx += vec.getFieldID(index);
  return success();
}

Optional<AnnoPathValue> resolveEntities(tokenAnnoTarget path,
                                        applyState state) {
  if (!path.circuit.empty() && state.circuit.name() != path.circuit) {
    state.circuit->emitError("circuit name doesn't match annotation '")
        << path.circuit << '\'';
    return {};
  }
  if (path.module.empty()) {
    assert(path.name.empty() && path.instances.empty() && path.agg.empty());
    return AnnoPathValue(state.circuit);
  }

  SmallVector<InstanceOp> instances;
  for (auto p : path.instances) {
    auto mod = state.symTbl.lookup<FModuleOp>(p.first);
    if (!mod) {
      state.circuit->emitError("module doesn't exist '") << p.first << '\'';
      return {};
    }
    auto resolved = findNamedThing(p.second, mod);
    if (!resolved.isInstance()) {
      state.circuit.emitError("cannot find instance '")
          << p.second << "' in '" << mod.getName() << "'";
      return {};
    }
    instances.push_back(cast<InstanceOp>(resolved.op));
  }
  auto mod = state.symTbl.lookup<FModuleOp>(path.module);
  if (!mod) {
    state.circuit->emitError("module doesn't exist '") << path.module << '\'';
    return {};
  }
  AnnoTarget ref;
  if (path.name.empty()) {
    assert(path.agg.empty());
    ref = AnnoTarget(mod);
  } else {
    ref = findNamedThing(path.name, mod);
    if (!ref) {
      state.circuit->emitError("cannot find name '")
          << path.name << "' in " << mod.getName();
      return {};
    }
  }
  for (auto agg : path.agg) {
    if (agg.isIndex) {
      if (failed(updateArray(agg.name, ref)))
        return {};
    } else {
      if (failed(updateStruct(agg.name, ref)))
        return {};
    }
  }
  return AnnoPathValue(instances, ref);
}

/// Return an input \p target string in canonical form.  This converts a Legacy
/// Annotation (e.g., A.B.C) into a modern annotation (e.g., ~A|B>C).  Trailing
/// subfield/subindex references are preserved.
static std::string canonicalizeTarget(StringRef target) {

  if (target.empty())
    return target.str();

  // If this is a normal Target (not a Named), erase that field in the JSON
  // object and return that Target.
  if (target[0] == '~')
    return target.str();

  // This is a legacy target using the firrtl.annotations.Named type.  This
  // can be trivially canonicalized to a non-legacy target, so we do it with
  // the following three mappings:
  //   1. CircuitName => CircuitTarget, e.g., A -> ~A
  //   2. ModuleName => ModuleTarget, e.g., A.B -> ~A|B
  //   3. ComponentName => ReferenceTarget, e.g., A.B.C -> ~A|B>C
  std::string newTarget = ("~" + target).str();
  auto n = newTarget.find('.');
  if (n != std::string::npos)
    newTarget[n] = '|';
  n = newTarget.find('.');
  if (n != std::string::npos)
    newTarget[n] = '>';
  return newTarget;
}

/// split a target string into it constituent parts.  This is the primary parser
/// for targets.
static Optional<tokenAnnoTarget> tokenizePath(StringRef origTarget) {
  StringRef target = origTarget;
  tokenAnnoTarget retval;
  std::tie(retval.circuit, target) = target.split('|');
  if (!retval.circuit.empty() && retval.circuit[0] == '~')
    retval.circuit = retval.circuit.drop_front();
  while (target.count(':')) {
    StringRef nla;
    std::tie(nla, target) = target.split(':');
    StringRef inst, mod;
    std::tie(mod, inst) = nla.split('/');
    retval.instances.emplace_back(mod, inst);
  }
  // remove aggregate
  auto targetBase =
      target.take_until([](char c) { return c == '.' || c == '['; });
  auto aggBase = target.drop_front(targetBase.size());
  std::tie(retval.module, retval.name) = targetBase.split('>');
  while (!aggBase.empty()) {
    if (aggBase[0] == '.') {
      aggBase = aggBase.drop_front();
      StringRef field = aggBase.take_front(aggBase.find_first_of("[."));
      aggBase = aggBase.drop_front(field.size());
      retval.agg.push_back({field, false});
    } else if (aggBase[0] == '[') {
      aggBase = aggBase.drop_front();
      StringRef index = aggBase.take_front(aggBase.find_first_of(']'));
      aggBase = aggBase.drop_front(index.size() + 1);
      retval.agg.push_back({index, true});
    } else {
      return {};
    }
  }

  return retval;
}

/// Make an anchor for a non-local annotation.  Use the expanded path to build
/// the module and name list in the anchor.
static FlatSymbolRefAttr buildNLA(AnnoPathValue target, applyState state) {
  OpBuilder b(state.circuit.getBodyRegion());
  SmallVector<Attribute> mods;
  SmallVector<Attribute> insts;
  for (auto inst : target.instances) {
    mods.push_back(FlatSymbolRefAttr::get(inst->getParentOfType<FModuleOp>()));
    insts.push_back(StringAttr::get(state.circuit.getContext(), inst.name()));
  }
  mods.push_back(FlatSymbolRefAttr::get(target.ref.getModule()));
  insts.push_back(
      StringAttr::get(state.circuit.getContext(), getName(target.ref.op)));
  auto modAttr = ArrayAttr::get(state.circuit.getContext(), mods);
  auto instAttr = ArrayAttr::get(state.circuit.getContext(), insts);
  auto nla = b.create<NonLocalAnchor>(state.circuit.getLoc(), "nla", modAttr,
                                      instAttr);
  state.symTbl.insert(nla);
  return FlatSymbolRefAttr::get(nla);
}

/// Scatter breadcrumb annotations corresponding to non-local annotations
/// along the instance path.  Returns symbol name used to anchor annotations to
/// path.
// FIXME: uniq annotation chain links
static FlatSymbolRefAttr scatterNonLocalPath(AnnoPathValue target,
                                             applyState state) {

  FlatSymbolRefAttr sym = buildNLA(target, state);

  NamedAttrList pathmetadata;
  pathmetadata.append("circt.nonlocal", sym);
  pathmetadata.append(
      "class", StringAttr::get(state.circuit.getContext(), "circt.nonlocal"));
  for (auto item : target.instances)
    addAnnotation(AnnoTarget(item), pathmetadata);

  return sym;
}

////////////////////////////////////////////////////////////////////////////////
// Standard Utility Resolvers
////////////////////////////////////////////////////////////////////////////////

static Optional<AnnoPathValue> noResolve(DictionaryAttr anno,
                                         applyState state) {
  return AnnoPathValue(state.circuit);
}

static Optional<AnnoPathValue> stdResolveImpl(StringRef rawPath,
                                              applyState state) {
  auto pathStr = canonicalizeTarget(rawPath);
  StringRef path{pathStr};

  auto tokens = tokenizePath(path);
  if (!tokens) {
    state.circuit->emitError("Cannot tokenize annotation path ") << rawPath;
    return {};
  }

  return resolveEntities(*tokens, state);
}

static Optional<AnnoPathValue> stdResolve(DictionaryAttr anno,
                                          applyState state) {
  auto target = anno.getNamed("target");
  if (!target) {
    state.circuit.emitError("No target field in annotation ") << anno;
    return {};
  }
  if (!target->second.isa<StringAttr>()) {
    state.circuit.emitError(
        "Target field in annotation doesn't contain string ")
        << anno;
    return {};
  }
  return stdResolveImpl(target->second.cast<StringAttr>().getValue(), state);
}

static Optional<AnnoPathValue> tryResolve(DictionaryAttr anno,
                                          applyState state) {
  auto target = anno.getNamed("target");
  if (target)
    return stdResolveImpl(target->second.cast<StringAttr>().getValue(), state);
  return AnnoPathValue(state.circuit);
}

////////////////////////////////////////////////////////////////////////////////
// Standard Utility Appliers
////////////////////////////////////////////////////////////////////////////////

static LogicalResult
ignoreAnno(AnnoPathValue target, DictionaryAttr anno,
           llvm::function_ref<void(ArrayAttr)> addToWorklist) {
  return success();
}

static LogicalResult applyWithoutTargetImpl(AnnoPathValue target,
                                            DictionaryAttr anno,
                                            applyState state,
                                            bool allowNonLocal) {
  if (!allowNonLocal && !target.isLocal())
    return failure();
  SmallVector<NamedAttribute> newAnnoAttrs;
  for (auto &na : anno) {
    if (na.first != "target") {
      newAnnoAttrs.push_back(na);
    } else if (!target.isLocal()) {
      auto sym = scatterNonLocalPath(target, state);
      newAnnoAttrs.push_back(
          {Identifier::get("circt.nonlocal", anno.getContext()), sym});
    }
  }
  addAnnotation(target.ref, newAnnoAttrs);
  return success();
}

template <bool allowNonLocal, typename T, typename... Tr>
static LogicalResult applyWithoutTarget(AnnoPathValue target,
                                        DictionaryAttr anno, applyState state) {
  if (!target.isOpOfType<T, Tr...>())
    return failure();
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

template <bool allowNonLocal = false>
static LogicalResult applyWithoutTarget(AnnoPathValue target,
                                        DictionaryAttr anno, applyState state) {
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

////////////////////////////////////////////////////////////////////////////////
// Driving table
////////////////////////////////////////////////////////////////////////////////

namespace {
struct AnnoRecord {
  llvm::function_ref<Optional<AnnoPathValue>(DictionaryAttr, applyState)>
      resolver;
  llvm::function_ref<LogicalResult(AnnoPathValue, DictionaryAttr, applyState)>
      applier;
};
}; // namespace

static const llvm::StringMap<AnnoRecord> annotationRecords{{

    // Testing Annotation
    {"circt.test", {stdResolve, applyWithoutTarget<true>}},
    {"circt.testNT", {noResolve, applyWithoutTarget<>}},
    {"circt.missing", {tryResolve, applyWithoutTarget<>}}

}};

static const AnnoRecord *getAnnotationHandler(StringRef annoStr,
                                              bool ignoreUnhandledAnno) {
  auto ii = annotationRecords.find(annoStr);
  if (ii != annotationRecords.end())
    return &ii->second;
  if (ignoreUnhandledAnno)
    return &annotationRecords.find("circt.missing")->second;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerAnnotationsPass
    : public LowerFIRRTLAnnotationsBase<LowerAnnotationsPass> {
  void runOnOperation() override;
  LogicalResult applyAnnotation(DictionaryAttr anno, applyState state);

  bool ignoreUnhandledAnno = false;
  bool ignoreClasslessAnno = false;
  SmallVector<DictionaryAttr> worklistAttrs;
};
} // end anonymous namespace

LogicalResult LowerAnnotationsPass::applyAnnotation(DictionaryAttr anno,
                                                    applyState state) {
  // Lookup the class
  StringRef annoClassVal;
  if (auto annoClass = anno.getNamed("class"))
    annoClassVal = annoClass->second.cast<StringAttr>().getValue();
  else if (ignoreClasslessAnno)
    annoClassVal = "circt.missing";
  else
    return state.circuit.emitError("Annotation without a class: ") << anno;

  // See if we handle the class
  auto record = getAnnotationHandler(annoClassVal, ignoreUnhandledAnno);
  if (!record)
    return state.circuit.emitWarning("Unhandled annotation: ") << anno;

  // Try to apply the annotation
  auto target = record->resolver(anno, state);
  if (!target)
    return state.circuit.emitError("Unable to resolve target of annotation: ")
           << anno;
  if (record->applier(*target, anno, state).failed())
    return state.circuit.emitError("Unable to apply annotation: ") << anno;
  return success();
}

// This is the main entrypoint for the lowering pass.
void LowerAnnotationsPass::runOnOperation() {
  CircuitOp circuit = getOperation();
  SymbolTable modules(circuit);
  // Grab the annotations.
  for (auto anno : circuit.annotations())
    worklistAttrs.push_back(anno.cast<DictionaryAttr>());
  // Clear the annotations.
  circuit.annotationsAttr(ArrayAttr::get(circuit.getContext(), {}));
  size_t numFailures = 0;
  applyState state{circuit, modules,
                   [&](DictionaryAttr ann) { worklistAttrs.push_back(ann); }

  };
  while (!worklistAttrs.empty()) {
    auto attr = worklistAttrs.pop_back_val();
    if (applyAnnotation(attr, state).failed())
      ++numFailures;
  }
  if (numFailures)
    signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLAnnotationsPass(
    bool ignoreUnhandledAnnotations, bool ignoreClasslessAnnotations) {
  auto pass = std::make_unique<LowerAnnotationsPass>();
  pass->ignoreUnhandledAnno = ignoreUnhandledAnnotations;
  pass->ignoreClasslessAnno = ignoreClasslessAnnotations;
  return pass;
}
