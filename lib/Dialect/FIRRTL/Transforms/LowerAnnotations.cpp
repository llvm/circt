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
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationLowering.h"
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

} // namespace

//===----------------------------------------------------------------------===//
// Local Helpers
//===----------------------------------------------------------------------===//

/// Append the argument `target` to the `annotation` using the key "target".
static inline void appendTarget(NamedAttrList &annotation, ArrayAttr target) {
  annotation.append("target", target);
}

//===----------------------------------------------------------------------===//
// Global Helpers 
//===----------------------------------------------------------------------===//

/// Mutably update a prototype Annotation (stored as a `NamedAttrList`) with
/// subfield/subindex information from a Target string.  Subfield/subindex
/// information will be placed in the key "target" at the back of the
/// Annotation.  If no subfield/subindex information, the Annotation is
/// unmodified.  Return the split input target as a base target (include a
/// reference if one exists) and an optional array containing subfield/subindex
/// tokens.
std::pair<StringRef, llvm::Optional<ArrayAttr>>
circt::firrtl::splitAndAppendTarget(NamedAttrList &annotation, StringRef target,
                                    MLIRContext *context) {
  auto targetPair = splitTarget(target, context);
  if (targetPair.second.hasValue())
    appendTarget(annotation, targetPair.second.getValue());

  return targetPair;
}

/// Split out non-local paths.  This will return a set of target strings for
/// each named entity along the path.
/// c|c:ai/Am:bi/Bm>d.agg[3] ->
/// c|c>ai, c|Am>bi, c|Bm>d.agg[2]
SmallVector<std::tuple<std::string, StringRef, StringRef>>
circt::firrtl::expandNonLocal(StringRef target) {
  SmallVector<std::tuple<std::string, StringRef, StringRef>> retval;
  StringRef circuit;
  std::tie(circuit, target) = target.split('|');
  while (target.count(':')) {
    StringRef nla;
    std::tie(nla, target) = target.split(':');
    StringRef inst, mod;
    std::tie(mod, inst) = nla.split('/');
    retval.emplace_back((circuit + "|" + mod + ">" + inst).str(), mod.str(),
                        inst.str());
  }
  if (target.empty()) {
    retval.emplace_back(circuit.str(), "", "");
  } else {

    StringRef mod, name;
    // remove aggregate
    auto targetBase =
        target.take_until([](char c) { return c == '.' || c == '['; });
    std::tie(mod, name) = targetBase.split('>');
    if (name.empty())
      name = mod;
    retval.emplace_back((circuit + "|" + target).str(), mod, name);
  }
  return retval;
}

/// Make an anchor for a non-local annotation.  Use the expanded path to build
/// the module and name list in the anchor.
FlatSymbolRefAttr circt::firrtl::buildNLA(AnnoPathValue target,
                                          AnnoApplyState state) {
  OpBuilder b(state.circuit.getBodyRegion());
  SmallVector<Attribute> mods;
  SmallVector<Attribute> insts;
  for (auto inst : target.getInstances()) {
    mods.push_back(FlatSymbolRefAttr::get(inst->getParentOfType<FModuleOp>()));
    insts.push_back(StringAttr::get(state.circuit.getContext(), inst.name()));
  }
  mods.push_back(FlatSymbolRefAttr::get(target.getModule()));
  insts.push_back(target.getOp()->getAttrOfType<StringAttr>("name"));
  auto modAttr = ArrayAttr::get(state.circuit.getContext(), mods);
  auto instAttr = ArrayAttr::get(state.circuit.getContext(), insts);
  auto nla = b.create<NonLocalAnchor>(state.circuit.getLoc(), "nla", modAttr,
                                      instAttr);
  state.symTbl.insert(nla);
  return FlatSymbolRefAttr::get(nla);
}

//===----------------------------------------------------------------------===//
// Data to/from the lowerer
//===----------------------------------------------------------------------===//

/// Implements the same behavior as DictionaryAttr::getAs<A> to return the value
/// of a specific type associated with a key in a dictionary.  However, this is
/// specialized to print a useful error message, specific to custom annotation
/// process, on failure.
template <typename A>
static A tryGetAs(DictionaryAttr &dict, const Attribute &root, StringRef key,
                  Location loc, Twine className, Twine path = Twine()) {
  // Check that the key exists.
  auto value = dict.get(key);
  if (!value) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className + "' did not contain required key '" +
             key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain required key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  // Check that the value has the correct type.
  auto valueA = value.dyn_cast_or_null<A>();
  if (!valueA) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  return valueA;
}

/// Abstraction over namable things.  Get a name in a generic way.
static StringRef getName(Operation *op) {
  return TypeSwitch<Operation *, StringRef>(op)
      .Case<InstanceOp, MemOp, NodeOp, RegOp, RegResetOp, WireOp, CombMemOp,
            SeqMemOp, MemoryPortOp>([&](auto nop) { return nop.name(); })
      .Default([](auto &) {
        llvm_unreachable("unnamable op");
        return "";
      });
}

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
static AnnoPathValue findNamedThing(StringRef name, Operation *op) {
  AnnoPathValue retval{op};
  auto nameChecker = [name, &retval](Operation *op) -> WalkResult {
    if (auto mod = dyn_cast<FModuleLike>(op)) {
      // Check the ports.
      auto ports = mod.getPorts();
      for (size_t i = 0, e = ports.size(); i != e; ++i)
        if (ports[i].name.getValue() == name) {
          retval = AnnoPathValue{mod, 0, i};
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    }
    if (hasName(name, op)) {
      retval = AnnoPathValue{op};
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
static void addAnnotation(AnnoPathValue ref, ArrayRef<NamedAttribute> anno) {
  DictionaryAttr annotation;
  if (ref.getField()) {
    SmallVector<NamedAttribute> annoField(anno.begin(), anno.end());
    annoField.emplace_back(
<<<<<<< HEAD
        Identifier::get("circt.fieldID", ref.getOp()->getContext()),
=======
        StringAttr::get("circt.fieldID", ref.op->getContext()),
>>>>>>> origin/main
        IntegerAttr::get(
            IntegerType::get(ref.getOp()->getContext(), 32, IntegerType::Signless),
            ref.getField()));
    annotation = DictionaryAttr::get(ref.getOp()->getContext(), annoField);
  } else {
    annotation = DictionaryAttr::get(ref.getOp()->getContext(), anno);
  }

  if (!ref.isPort()) {
    auto newAnno = appendArrayAttr(getAnnotationsFrom(ref.getOp()), annotation);
    ref.getOp()->setAttr(getAnnotationAttrName(), newAnno);
    return;
  }

  auto portAnnoRaw = ref.getOp()->getAttr(getPortAnnotationAttrName());
  ArrayAttr portAnno = portAnnoRaw.dyn_cast_or_null<ArrayAttr>();
  if (!portAnno || portAnno.size() != getNumPorts(ref.getOp())) {
    SmallVector<Attribute> emptyPortAttr(
        getNumPorts(ref.getOp()), ArrayAttr::get(ref.getOp()->getContext(), {}));
    portAnno = ArrayAttr::get(ref.getOp()->getContext(), emptyPortAttr);
  }
  portAnno = replaceArrayAttrElement(
      portAnno, ref.getPort(),
      appendArrayAttr(portAnno[ref.getPort()].dyn_cast<ArrayAttr>(), annotation));
  ref.getOp()->setAttr("portAnnotations", portAnno);
}

// Some types have been expanded so the first layer of aggregate path is
// a return value.
static LogicalResult updateExpandedPort(StringRef field,
                                        AnnoPathValue &entity) {
  if (auto mem = dyn_cast<MemOp>(entity.getOp()))
    for (size_t p = 0, pe = mem.portNames().size(); p < pe; ++p)
      if (mem.getPortNameStr(p) == field) {
        entity.setPort(p);
        return success();
      }
  if (auto inst = dyn_cast<InstanceOp>(entity.getOp()))
    for (size_t p = 0, pe = inst.getNumResults(); p < pe; ++p)
      if (inst.getPortNameStr(p) == field) {
        entity.setPort(p);
        return success();
      }
  entity.getOp()->emitError("Cannot find port with name ") << field;
  return failure();
}

/// Try to resolve an non-array aggregate name from a target given the type and
/// operation of the resolved target.  This needs to deal with places where we
/// represent bundle returns as split into constituent parts.
static LogicalResult updateStruct(StringRef field, AnnoPathValue &entity) {
  // The first field for some ops refers to expanded return values.
  if (isa<MemOp, InstanceOp>(entity.getOp()) && !entity.isPort())
    return updateExpandedPort(field, entity);

  auto bundle = entity.getType().dyn_cast<BundleType>();
  if (!bundle)
    return entity.getOp()->emitError("field access '")
           << field << "' into non-bundle type '" << bundle << "'";
  if (auto idx = bundle.getElementIndex(field)) {
    entity.setField(entity.getField() + bundle.getFieldID(*idx));
    return success();
  }
  return entity.getOp()->emitError("cannot resolve field '")
         << field << "' in subtype '" << bundle << "'";
}

/// Try to resolve an array index from a target given the type of the resolved
/// target.
static LogicalResult updateArray(StringRef indexStr, AnnoPathValue &entity) {
  size_t index;
  if (indexStr.getAsInteger(10, index)) {
    entity.getOp()->emitError("Cannot convert '") << indexStr << "' to an integer";
    return failure();
  }

  auto vec = entity.getType().dyn_cast<FVectorType>();
  if (!vec)
    return entity.getOp()->emitError("index access '")
           << index << "' into non-vector type '" << vec << "'";
  entity.setField(entity.getField() + vec.getFieldID(index));
  return success();
}

/// Convert a parsed target string to a resolved target structure.  This
/// resolves all names and aggregates from a parsed target.
Optional<AnnoPathValue> resolveEntities(TokenAnnoTarget path,
                                        AnnoApplyState state) {
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
    if (!resolved.isInstance()) {
      state.circuit.emitError("cannot find instance '")
          << p.second << "' in '" << mod.getName() << "'";
      return {};
    }
    instances.push_back(cast<InstanceOp>(resolved.getOp()));
  }
  // The final module is where the named target is (or is the named target).
  auto mod = state.symTbl.lookup<FModuleLike>(path.module);
  if (!mod) {
    state.circuit->emitError("module doesn't exist '") << path.module << '\'';
    return {};
  }
  AnnoPathValue ref;
  if (path.name.empty()) {
    assert(path.component.empty());
    ref = AnnoPathValue(mod);
  } else {
    ref = findNamedThing(path.name, mod);
    if (!ref) {
      state.circuit->emitError("cannot find name '")
          << path.name << "' in " << mod.moduleName();
      return {};
    }
  }
  // If we have aggregate specifiers, resolve those now.
  for (auto agg : path.component) {
    if (agg.isIndex) {
      if (failed(updateArray(agg.name, ref)))
        return {};
    } else {
      if (failed(updateStruct(agg.name, ref)))
        return {};
    }
  }
  ref.setPath(instances);
  return ref;
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

/// Split a target into a base target (including a reference if one exists) and
/// an optional array of subfield/subindex tokens.
std::pair<StringRef, llvm::Optional<ArrayAttr>>
circt::firrtl::splitTarget(StringRef target, MLIRContext *context) {
  if (target.empty())
    return {target, None};

  // Find the string index where the target can be partitioned into the "base
  // target" and the "target".  The "base target" is the module or instance and
  // the "target" is everything else.  This has two variants that need to be
  // considered:
  //
  //   1) A Local target, e.g., ~Foo|Foo>bar.baz
  //   2) An instance target, e.g., ~Foo|Foo/bar:Bar>baz.qux
  //
  // In (1), this should be partitioned into ["~Foo|Foo>bar", ".baz"].  In (2),
  // this should be partitioned into ["~Foo|Foo/bar:Bar", ">baz.qux"].
  bool isInstance = false;
  size_t fieldBegin = target.find_if_not([&isInstance](char c) {
    switch (c) {
    case '/':
      return isInstance = true;
    case '>':
      return !isInstance;
    case '[':
    case '.':
      return false;
    default:
      return true;
    };
  });

  // Exit if the target does not contain a subfield or subindex.
  if (fieldBegin == StringRef::npos)
    return {target, None};

  auto targetBase = target.take_front(fieldBegin);
  target = target.substr(fieldBegin);
  SmallVector<Attribute> annotationVec;
  SmallString<16> temp;
  for (auto c : target.drop_front()) {
    if (c == ']') {
      // Create a IntegerAttr with the previous sub-index token.
      APInt subIndex;
      if (!temp.str().getAsInteger(10, subIndex))
        annotationVec.push_back(IntegerAttr::get(IntegerType::get(context, 64),
                                                 subIndex.getZExtValue()));
      else
        // We don't have a good way to emit error here. This will be reported as
        // an error in the FIRRTL parser.
        annotationVec.push_back(StringAttr::get(context, temp));
      temp.clear();
    } else if (c == '[' || c == '.') {
      // Create a StringAttr with the previous token.
      if (!temp.empty())
        annotationVec.push_back(StringAttr::get(context, temp));
      temp.clear();
    } else
      temp.push_back(c);
  }
  // Save the last token.
  if (!temp.empty())
    annotationVec.push_back(StringAttr::get(context, temp));

  return {targetBase, ArrayAttr::get(context, annotationVec)};
}

/// Make an anchor for a non-local annotation.  Use the expanded path to build
/// the module and name list in the anchor.
FlatSymbolRefAttr circt::firrtl::buildNLA(
    CircuitOp circuit, size_t nlaSuffix,
    SmallVectorImpl<std::tuple<std::string, StringRef, StringRef>> &nlas) {
  OpBuilder b(circuit.getBodyRegion());
  SmallVector<Attribute> mods;
  SmallVector<Attribute> insts;
  for (auto &nla : nlas) {
    mods.push_back(
        FlatSymbolRefAttr::get(circuit.getContext(), std::get<1>(nla)));
    insts.push_back(StringAttr::get(circuit.getContext(), std::get<2>(nla)));
  }
  auto modAttr = ArrayAttr::get(circuit.getContext(), mods);
  auto instAttr = ArrayAttr::get(circuit.getContext(), insts);
  auto nla = b.create<NonLocalAnchor>(
      circuit.getLoc(), "nla_" + std::to_string(nlaSuffix), modAttr, instAttr);
  return FlatSymbolRefAttr::get(nla);
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

/// Scatter breadcrumb annotations corresponding to non-local annotations
/// along the instance path.  Returns symbol name used to anchor annotations to
/// path.
// FIXME: uniq annotation chain links
static FlatSymbolRefAttr scatterNonLocalPath(AnnoPathValue target,
                                             AnnoApplyState state) {

  FlatSymbolRefAttr sym = buildNLA(target, state);

  NamedAttrList pathmetadata;
  pathmetadata.append("circt.nonlocal", sym);
  pathmetadata.append(
      "class", StringAttr::get(state.circuit.getContext(), "circt.nonlocal"));
  for (auto item : target.getInstances())
    addAnnotation(AnnoPathValue(item), pathmetadata);

  return sym;
}

//===----------------------------------------------------------------------===//
// Standard Utility Resolvers
//===----------------------------------------------------------------------===//

/// Always resolve to the circuit, ignoring the annotation.
static Optional<AnnoPathValue> noResolve(DictionaryAttr anno,
                                         AnnoApplyState state) {
  return AnnoPathValue(state.circuit);
}

/// Implementation of standard resolution.  First parses the target path, then
/// resolves it.
static Optional<AnnoPathValue> stdResolveImpl(StringRef rawPath,
                                              AnnoApplyState state) {
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
                                          AnnoApplyState state) {
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
                                          AnnoApplyState state) {
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
                                            AnnoApplyState state,
                                            bool allowNonLocal) {
  if (!allowNonLocal && !target.isLocal())
    return failure();
  SmallVector<NamedAttribute> newAnnoAttrs;
  for (auto &na : anno) {
    if (na.getName() != "target") {
      newAnnoAttrs.push_back(na);
    } else if (!target.isLocal()) {
      auto sym = scatterNonLocalPath(target, state);
      newAnnoAttrs.push_back(
          {StringAttr::get("circt.nonlocal", anno.getContext()), sym});
    }
  }
  addAnnotation(target, newAnnoAttrs);
  return success();
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
/// Ensures the target resolves to an expected type of operation.
template <bool allowNonLocal, typename T, typename... Tr>
static LogicalResult applyWithoutTarget(AnnoPathValue target,
                                        DictionaryAttr anno,
                                        AnnoApplyState state) {
  if (!target.isOpOfType<T, Tr...>())
    return failure();
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
template <bool allowNonLocal = false>
static LogicalResult applyWithoutTarget(AnnoPathValue target,
                                        DictionaryAttr anno,
                                        AnnoApplyState state) {
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

//===----------------------------------------------------------------------===//
// Driving table
//===----------------------------------------------------------------------===//

namespace {
struct AnnoRecord {
  llvm::function_ref<Optional<AnnoPathValue>(DictionaryAttr, AnnoApplyState)>
      resolver;
  llvm::function_ref<LogicalResult(AnnoPathValue, DictionaryAttr,
                                   AnnoApplyState)>
      applier;
};
} // end anonymous namespace

static const llvm::StringMap<AnnoRecord> annotationRecords{
    {"sifive.enterprise.firrtl.MarkDUTAnnotation",
     {stdResolve, applyWithoutTarget<>}},
//    {"sifive.enterprise.grandcentral.DataTapsAnnotation",
//     {stdResolve, applyGCDataTap}},
    {"sifive.enterprise.grandcentral.MemTapAnnotation",
     {stdResolve, applyGCMemTap}},
//    {"sifive.enterprise.grandcentral.SignalDriverAnnotation",
//     {stdResolve, applyGCSigDriver}},
//    {"sifive.enterprise.grandcentral.GrandCentralView$",
//     {stdResolve, applyGCView}},
//    {"SerializedViewAnnotation", {stdResolve, applyGCView}},
//    {
//        "sifive.enterprise.grandcentral.ViewAnnotation",
//        {stdResolve, applyGCView},
//    },
//    {
//        "sifive.enterprise.grandcentral.ModuleReplacementAnnotation",
//        {stdResolve, applyModRep},
//    },
     //    {"firrtl.transforms.DontTouchAnnotation",
      //    {stdResolve, applyDontTouch}},

    // Testing Annotation
    {"circt.test", {stdResolve, applyWithoutTarget<true>}},
    {"circt.testNT", {noResolve, applyWithoutTarget<>}},
    {"circt.missing", {tryResolve, applyWithoutTarget<>}}};

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
  LogicalResult applyAnnotation(DictionaryAttr anno, AnnoApplyState state);

  bool ignoreUnhandledAnno = false;
  bool ignoreClasslessAnno = false;
  SmallVector<DictionaryAttr> worklistAttrs;
};
} // end anonymous namespace

LogicalResult LowerAnnotationsPass::applyAnnotation(DictionaryAttr anno,
                                                    AnnoApplyState state) {
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
    return state.circuit.emitWarning("Unhandled annotation: ") << annoClassVal;

  // Try to apply the annotation
  auto target = record->resolver(anno, state);
  if (!target)
    return state.circuit.emitError("Unable to resolve target of annotation: ")
           << annoClassVal;
  if (record->applier(*target, anno, state).failed())
    return state.circuit.emitError("Unable to apply annotation: ")
           << annoClassVal;
  return success();
}

// This is the main entrypoint for the lowering pass.
void LowerAnnotationsPass::runOnOperation() {
  CircuitOp circuit = getOperation();
  // Grab the annotations.
  if (auto raw = circuit->getAttrOfType<ArrayAttr>("raw_annotations")) {
    SymbolTable modules(circuit);
    for (auto anno : raw)
      worklistAttrs.push_back(anno.cast<DictionaryAttr>());
    // Clear the raw annotations.
    circuit->removeAttr("raw_annotations");
    AnnoApplyState state{
        circuit, modules,
        [&](DictionaryAttr ann) { worklistAttrs.push_back(ann); }

    };
    size_t numFailures = 0;
    while (!worklistAttrs.empty()) {
      auto attr = worklistAttrs.pop_back_val();
      if (applyAnnotation(attr, state).failed())
        ++numFailures;
    }
    if (numFailures)
      signalPassFailure();
  }
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLAnnotationsPass(
    bool ignoreUnhandledAnnotations, bool ignoreClasslessAnnotations) {
  auto pass = std::make_unique<LowerAnnotationsPass>();
  pass->ignoreUnhandledAnno = ignoreUnhandledAnnotations;
  pass->ignoreClasslessAnno = ignoreClasslessAnnotations;
  return pass;
}
