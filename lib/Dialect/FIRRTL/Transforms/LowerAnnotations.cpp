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
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringExtras.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

namespace {

/// Stores an index into an aggregate.
struct TargetToken {
  StringRef name;
  bool isIndex;
};

/// The parsed annotation path.
struct TokenAnnoTarget {
  StringRef circuit;
  SmallVector<std::pair<StringRef, StringRef>> instances;
  StringRef module;
  // The final name of the target
  StringRef name;
  // Any aggregates indexed.
  SmallVector<TargetToken> component;
};

// The potentially non-local resolved annotation.
struct AnnoPathValue {
  SmallVector<InstanceOp> instances;
  AnnoTarget ref;
  unsigned fieldIdx = 0;

  AnnoPathValue() = default;
  AnnoPathValue(CircuitOp op) : ref(OpAnnoTarget(op)) {}
  AnnoPathValue(Operation *op) : ref(OpAnnoTarget(op)) {}
  AnnoPathValue(const SmallVectorImpl<InstanceOp> &insts, AnnoTarget b,
                unsigned fieldIdx)
      : instances(insts.begin(), insts.end()), ref(b), fieldIdx(fieldIdx) {}

  bool isLocal() const { return instances.empty(); }

  template <typename... T>
  bool isOpOfType() const {
    if (auto opRef = ref.dyn_cast<OpAnnoTarget>())
      return isa<T...>(opRef.getOp());
    return false;
  }
};

/// State threaded through functions for resolving and applying annotations.
struct ApplyState {
  using AddToWorklistFn = llvm::function_ref<void(DictionaryAttr)>;
  ApplyState(CircuitOp circuit, SymbolTable &symTbl,
             AddToWorklistFn addToWorklistFn)
      : circuit(circuit), symTbl(symTbl), addToWorklistFn(addToWorklistFn) {}

  CircuitOp circuit;
  SymbolTable &symTbl;
  AddToWorklistFn addToWorklistFn;

  ModuleNamespace &getNamespace(FModuleLike module) {
    auto &ptr = namespaces[module];
    if (!ptr)
      ptr = std::make_unique<ModuleNamespace>(module);
    return *ptr;
  }

private:
  DenseMap<Operation *, std::unique_ptr<ModuleNamespace>> namespaces;
};

} // namespace

/// Abstraction over namable things.  Do they have names?
static bool hasName(StringRef name, Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<InstanceOp, MemOp, NodeOp, RegOp, RegResetOp, WireOp, CombMemOp,
            SeqMemOp, MemoryPortOp>(
          [&](auto nop) { return nop.name() == name; })
      .Default([](auto &) { return false; });
}

/// Find a matching name in an operation (usually FModuleOp).  This walk could
/// be cached in the future.  This finds a port or operation for a given name.
static AnnoTarget findNamedThing(StringRef name, Operation *op) {
  AnnoTarget retval;
  auto nameChecker = [name, &retval](Operation *op) -> WalkResult {
    if (auto mod = dyn_cast<FModuleLike>(op)) {
      // Check the ports.
      auto ports = mod.getPorts();
      for (size_t i = 0, e = ports.size(); i != e; ++i)
        if (ports[i].name.getValue() == name) {
          retval = PortAnnoTarget(op, i);
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    }
    if (hasName(name, op)) {
      retval = OpAnnoTarget(op);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  op->walk(nameChecker);
  return retval;
}

/// Get annotations or an empty set of annotations.
static ArrayAttr getAnnotationsFrom(Operation *op) {
  if (auto annots = op->getAttrOfType<ArrayAttr>(getAnnotationAttrName()))
    return annots;
  return ArrayAttr::get(op->getContext(), {});
}

/// Construct the annotation array with a new thing appended.
static ArrayAttr appendArrayAttr(ArrayAttr array, Attribute a) {
  if (!array)
    return ArrayAttr::get(a.getContext(), ArrayRef<Attribute>{a});
  SmallVector<Attribute> old(array.begin(), array.end());
  old.push_back(a);
  return ArrayAttr::get(a.getContext(), old);
}

/// Update an ArrayAttribute by replacing one entry.
static ArrayAttr replaceArrayAttrElement(ArrayAttr array, size_t elem,
                                         Attribute newVal) {
  SmallVector<Attribute> old(array.begin(), array.end());
  old[elem] = newVal;
  return ArrayAttr::get(array.getContext(), old);
}

/// Apply a new annotation to a resolved target.  This handles ports,
/// aggregates, modules, wires, etc.
static void addAnnotation(AnnoTarget ref, unsigned fieldIdx,
                          ArrayRef<NamedAttribute> anno) {
  auto *context = ref.getOp()->getContext();
  DictionaryAttr annotation;
  if (fieldIdx) {
    SmallVector<NamedAttribute> annoField(anno.begin(), anno.end());
    annoField.emplace_back(
        StringAttr::get(context, "circt.fieldID"),
        IntegerAttr::get(IntegerType::get(context, 32, IntegerType::Signless),
                         fieldIdx));
    annotation = DictionaryAttr::get(context, annoField);
  } else {
    annotation = DictionaryAttr::get(context, anno);
  }

  if (ref.isa<OpAnnoTarget>()) {
    auto newAnno = appendArrayAttr(getAnnotationsFrom(ref.getOp()), annotation);
    ref.getOp()->setAttr(getAnnotationAttrName(), newAnno);
    return;
  }

  auto portRef = ref.cast<PortAnnoTarget>();
  auto portAnnoRaw = ref.getOp()->getAttr(getPortAnnotationAttrName());
  ArrayAttr portAnno = portAnnoRaw.dyn_cast_or_null<ArrayAttr>();
  if (!portAnno || portAnno.size() != getNumPorts(ref.getOp())) {
    SmallVector<Attribute> emptyPortAttr(
        getNumPorts(ref.getOp()),
        ArrayAttr::get(ref.getOp()->getContext(), {}));
    portAnno = ArrayAttr::get(ref.getOp()->getContext(), emptyPortAttr);
  }
  portAnno = replaceArrayAttrElement(
      portAnno, portRef.getPortNo(),
      appendArrayAttr(portAnno[portRef.getPortNo()].dyn_cast<ArrayAttr>(),
                      annotation));
  ref.getOp()->setAttr("portAnnotations", portAnno);
}

// Some types have been expanded so the first layer of aggregate path is
// a return value.
static LogicalResult updateExpandedPort(StringRef field, AnnoTarget &ref) {
  if (auto mem = dyn_cast<MemOp>(ref.getOp()))
    for (size_t p = 0, pe = mem.portNames().size(); p < pe; ++p)
      if (mem.getPortNameStr(p) == field) {
        ref = PortAnnoTarget(ref.getOp(), p);
        return success();
      }
  ref.getOp()->emitError("Cannot find port with name ") << field;
  return failure();
}

/// Try to resolve an non-array aggregate name from a target given the type and
/// operation of the resolved target.  This needs to deal with places where we
/// represent bundle returns as split into constituent parts.
static FailureOr<unsigned> findBundleElement(Operation *op, Type type,
                                             StringRef field) {
  auto bundle = type.dyn_cast<BundleType>();
  if (!bundle) {
    op->emitError("field access '")
        << field << "' into non-bundle type '" << bundle << "'";
    return failure();
  }
  auto idx = bundle.getElementIndex(field);
  if (!idx) {
    op->emitError("cannot resolve field '")
        << field << "' in subtype '" << bundle << "'";
    return failure();
  }
  return *idx;
}

/// Try to resolve an array index from a target given the type of the resolved
/// target.
static FailureOr<unsigned> findVectorElement(Operation *op, Type type,
                                             StringRef indexStr) {
  size_t index;
  if (indexStr.getAsInteger(10, index)) {
    op->emitError("Cannot convert '") << indexStr << "' to an integer";
    return failure();
  }
  auto vec = type.dyn_cast<FVectorType>();
  if (!vec) {
    op->emitError("index access '")
        << index << "' into non-vector type '" << vec << "'";
    return failure();
  }
  return index;
}

static FailureOr<unsigned> findFieldID(AnnoTarget ref,
                                       ArrayRef<TargetToken> tokens) {
  if (tokens.empty())
    return 0;

  auto *op = ref.getOp();
  auto type = ref.getType();
  auto fieldIdx = 0;
  // The first field for some ops refers to expanded return values.
  if (isa<MemOp>(ref.getOp())) {
    if (failed(updateExpandedPort(tokens.front().name, ref)))
      return {};
    tokens = tokens.drop_front();
  }

  for (auto token : tokens) {
    if (token.isIndex) {
      auto result = findVectorElement(op, type, token.name);
      if (failed(result))
        return failure();
      auto vector = type.cast<FVectorType>();
      type = vector.getElementType();
      fieldIdx += vector.getFieldID(*result);
    } else {
      auto result = findBundleElement(op, type, token.name);
      if (failed(result))
        return failure();
      auto bundle = type.cast<BundleType>();
      type = bundle.getElementType(*result);
      fieldIdx += bundle.getFieldID(*result);
    }
  }
  return fieldIdx;
}

/// Convert a parsed target string to a resolved target structure.  This
/// resolves all names and aggregates from a parsed target.
Optional<AnnoPathValue> resolveEntities(TokenAnnoTarget path,
                                        ApplyState &state) {
  // Validate circuit name.
  if (!path.circuit.empty() && state.circuit.name() != path.circuit) {
    state.circuit->emitError("circuit name doesn't match annotation '")
        << path.circuit << '\'';
    return {};
  }
  // Circuit only target.
  if (path.module.empty()) {
    assert(path.name.empty() && path.instances.empty() &&
           path.component.empty());
    return AnnoPathValue(state.circuit);
  }

  // Resolve all instances for non-local paths.
  SmallVector<InstanceOp> instances;
  for (auto p : path.instances) {
    auto mod = state.symTbl.lookup<FModuleOp>(p.first);
    if (!mod) {
      state.circuit->emitError("module doesn't exist '") << p.first << '\'';
      return {};
    }
    auto resolved = findNamedThing(p.second, mod);
    if (!resolved || !isa<InstanceOp>(resolved.getOp())) {
      state.circuit.emitError("cannot find instance '")
          << p.second << "' in '" << mod.getName() << "'";
      return {};
    }
    instances.push_back(cast<InstanceOp>(resolved.getOp()));
  }
  // The final module is where the named target is (or is the named target).
  auto mod = state.symTbl.lookup<FModuleOp>(path.module);
  if (!mod) {
    state.circuit->emitError("module doesn't exist '") << path.module << '\'';
    return {};
  }
  AnnoTarget ref;
  if (path.name.empty()) {
    assert(path.component.empty());
    ref = OpAnnoTarget(mod);
  } else {
    ref = findNamedThing(path.name, mod);
    if (!ref) {
      state.circuit->emitError("cannot find name '")
          << path.name << "' in " << mod.getName();
      return {};
    }
  }

  // If the reference is pointing to an instance op, we have to move the target
  // to the module.
  ArrayRef<TargetToken> component(path.component);
  if (auto instance = dyn_cast<InstanceOp>(ref.getOp())) {
    instances.push_back(instance);
    auto target = instance.getReferencedModule(state.symTbl);
    if (component.empty()) {
      ref = OpAnnoTarget(instance.getReferencedModule(state.symTbl));
    } else {
      auto field = component.front().name;
      ref = AnnoTarget();
      for (size_t p = 0, pe = target.getNumPorts(); p < pe; ++p)
        if (target.getPortName(p) == field) {
          ref = PortAnnoTarget(target, p);
          break;
        }
      if (!ref) {
        state.circuit->emitError("!cannot find port '")
            << field << "' in module " << target.moduleName();
        return {};
      }
      component = component.drop_front();
    }
  }

  // If we have aggregate specifiers, resolve those now.
  auto result = findFieldID(ref, component);
  if (failed(result))
    return {};
  auto fieldIdx = *result;

  return AnnoPathValue(instances, ref, fieldIdx);
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
static Optional<TokenAnnoTarget> tokenizePath(StringRef origTarget) {
  StringRef target = origTarget;
  TokenAnnoTarget retval;
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
      retval.component.push_back({field, false});
    } else if (aggBase[0] == '[') {
      aggBase = aggBase.drop_front();
      StringRef index = aggBase.take_front(aggBase.find_first_of(']'));
      aggBase = aggBase.drop_front(index.size() + 1);
      retval.component.push_back({index, true});
    } else {
      return {};
    }
  }

  return retval;
}

/// Make an anchor for a non-local annotation.  Use the expanded path to build
/// the module and name list in the anchor.
static FlatSymbolRefAttr buildNLA(AnnoPathValue target, ApplyState &state) {
  OpBuilder b(state.circuit.getBodyRegion());
  SmallVector<Attribute> insts;
  for (auto inst : target.instances)
    insts.push_back(OpAnnoTarget(inst).getNLAReference(
        state.getNamespace(inst->getParentOfType<FModuleLike>())));

  auto module = dyn_cast<FModuleLike>(target.ref.getOp());
  if (!module)
    module = target.ref.getOp()->getParentOfType<FModuleLike>();
  insts.push_back(target.ref.getNLAReference(state.getNamespace(module)));
  auto instAttr = ArrayAttr::get(state.circuit.getContext(), insts);
  auto nla = b.create<NonLocalAnchor>(state.circuit.getLoc(), "nla", instAttr);
  state.symTbl.insert(nla);
  return FlatSymbolRefAttr::get(nla);
}

/// Scatter breadcrumb annotations corresponding to non-local annotations
/// along the instance path.  Returns symbol name used to anchor annotations to
/// path.
// FIXME: uniq annotation chain links
static FlatSymbolRefAttr scatterNonLocalPath(AnnoPathValue target,
                                             ApplyState &state) {

  FlatSymbolRefAttr sym = buildNLA(target, state);

  NamedAttrList pathmetadata;
  pathmetadata.append("circt.nonlocal", sym);
  pathmetadata.append(
      "class", StringAttr::get(state.circuit.getContext(), "circt.nonlocal"));
  for (auto item : target.instances)
    addAnnotation(OpAnnoTarget(item), 0, pathmetadata);

  return sym;
}

//===----------------------------------------------------------------------===//
// Standard Utility Resolvers
//===----------------------------------------------------------------------===//

/// Always resolve to the circuit, ignoring the annotation.
static Optional<AnnoPathValue> noResolve(DictionaryAttr anno,
                                         ApplyState &state) {
  return AnnoPathValue(state.circuit);
}

/// Implementation of standard resolution.  First parses the target path, then
/// resolves it.
static Optional<AnnoPathValue> stdResolveImpl(StringRef rawPath,
                                              ApplyState &state) {
  auto pathStr = canonicalizeTarget(rawPath);
  StringRef path{pathStr};

  auto tokens = tokenizePath(path);
  if (!tokens) {
    state.circuit->emitError("Cannot tokenize annotation path ") << rawPath;
    return {};
  }

  return resolveEntities(*tokens, state);
}

/// (SFC) FIRRTL SingleTargetAnnotation resolver.  Uses the 'target' field of
/// the annotation with standard parsing to resolve the path.  This requires
/// 'target' to exist and be normalized (per docs/FIRRTLAnnotations.md).
static Optional<AnnoPathValue> stdResolve(DictionaryAttr anno,
                                          ApplyState &state) {
  auto target = anno.getNamed("target");
  if (!target) {
    state.circuit.emitError("No target field in annotation ") << anno;
    return {};
  }
  if (!target->getValue().isa<StringAttr>()) {
    state.circuit.emitError(
        "Target field in annotation doesn't contain string ")
        << anno;
    return {};
  }
  return stdResolveImpl(target->getValue().cast<StringAttr>().getValue(),
                        state);
}

/// Resolves with target, if it exists.  If not, resolves to the circuit.
static Optional<AnnoPathValue> tryResolve(DictionaryAttr anno,
                                          ApplyState &state) {
  auto target = anno.getNamed("target");
  if (target)
    return stdResolveImpl(target->getValue().cast<StringAttr>().getValue(),
                          state);
  return AnnoPathValue(state.circuit);
}

//===----------------------------------------------------------------------===//
// Standard Utility Appliers
//===----------------------------------------------------------------------===//

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
static LogicalResult applyWithoutTargetImpl(AnnoPathValue target,
                                            DictionaryAttr anno,
                                            ApplyState &state,
                                            bool allowNonLocal) {
  if (!allowNonLocal && !target.isLocal())
    return failure();
  SmallVector<NamedAttribute> newAnnoAttrs;
  for (auto &na : anno) {
    if (na.getName().getValue() != "target") {
      newAnnoAttrs.push_back(na);
    } else if (!target.isLocal()) {
      auto sym = scatterNonLocalPath(target, state);
      newAnnoAttrs.push_back(
          {StringAttr::get(anno.getContext(), "circt.nonlocal"), sym});
    }
  }
  addAnnotation(target.ref, target.fieldIdx, newAnnoAttrs);
  return success();
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
/// Ensures the target resolves to an expected type of operation.
template <bool allowNonLocal, typename T, typename... Tr>
static LogicalResult applyWithoutTarget(AnnoPathValue target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  if (!target.isOpOfType<T, Tr...>())
    return failure();
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
template <bool allowNonLocal = false>
static LogicalResult applyWithoutTarget(AnnoPathValue target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

//===----------------------------------------------------------------------===//
// Driving table
//===----------------------------------------------------------------------===//

namespace {
struct AnnoRecord {
  llvm::function_ref<Optional<AnnoPathValue>(DictionaryAttr, ApplyState &)>
      resolver;
  llvm::function_ref<LogicalResult(AnnoPathValue, DictionaryAttr, ApplyState &)>
      applier;
};
} // end anonymous namespace

static const llvm::StringMap<AnnoRecord> annotationRecords{{

    // Testing Annotation
    {"circt.test", {stdResolve, applyWithoutTarget<true>}},
    {"circt.testNT", {noResolve, applyWithoutTarget<>}},
    {"circt.missing", {tryResolve, applyWithoutTarget<>}}

}};

/// Lookup a record for a given annotation class.  Optionally, returns the
/// record for "circuit.missing" if the record doesn't exist.
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
  LogicalResult applyAnnotation(DictionaryAttr anno, ApplyState &state);

  bool ignoreUnhandledAnno = false;
  bool ignoreClasslessAnno = false;
  SmallVector<DictionaryAttr> worklistAttrs;
};
} // end anonymous namespace

LogicalResult LowerAnnotationsPass::applyAnnotation(DictionaryAttr anno,
                                                    ApplyState &state) {
  // Lookup the class
  StringRef annoClassVal;
  if (auto annoClass = anno.getNamed("class"))
    annoClassVal = annoClass->getValue().cast<StringAttr>().getValue();
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
  ApplyState state{circuit, modules,
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
