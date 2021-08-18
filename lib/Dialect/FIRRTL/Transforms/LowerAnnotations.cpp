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

// Add the annotations to worklist.
static void addToWorklist(ArrayAttr &annos);
// Generate a unique ID.
IntegerAttr newID(MLIRContext *context);

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

static void addAnnotation(BaseUnion ref, ArrayRef<NamedAttribute> anno) {
  SmallVector<Attribute> newAnnos;
  for (auto old : getAnnotationsFrom(ref.op))
    newAnnos.push_back(old);
  if (ref.fieldIdx) {
    SmallVector<NamedAttribute> anno2(anno.begin(), anno.end());
    anno2.emplace_back(
        Identifier::get("circt.fieldID", ref.op->getContext()),
        IntegerAttr::get(
            IntegerType::get(ref.op->getContext(), 32, IntegerType::Signless),
            ref.fieldIdx));
    newAnnos.push_back(DictionaryAttr::get(ref.op->getContext(), anno2));
  } else {
    newAnnos.push_back(DictionaryAttr::get(ref.op->getContext(), anno));
  }
  ref.op->setAttr(getAnnotationAttrName(),
                  ArrayAttr::get(ref.op->getContext(), newAnnos));
}

// Returns remainder of path if circuit is the correct circuit
static LogicalResult parseAndCheckCircuit(StringRef &path, CircuitOp circuit) {
  if (path.empty())
    return circuit.emitError("empty target string");

  if (path.startswith("~"))
    path = path.drop_front();

  // Any non-trivial target must start with a circuit name.
  StringRef name;
  std::tie(name, path) = path.split('|');
  // ~ or ~Foo
  if (name.empty() || name == circuit.name())
    return success();
  return circuit.emitError("circuit name '")
         << circuit.name() << "' doesn't match annotation '" << name << "'";
}

// Returns remainder of path if circuit is the correct circuit
static LogicalResult parseAndCheckModule(StringRef &path, Operation *&module,
                                         CircuitOp circuit,
                                         SymbolTable &modules) {
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
static LogicalResult parseAndCheckInstance(StringRef &path, Operation *&module,
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
static LogicalResult updateExpandedPort(StringRef field, BaseUnion &entity) {
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

static LogicalResult updateStruct(StringRef field, BaseUnion &entity) {
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

static LogicalResult updateArray(unsigned index, BaseUnion &entity) {
  auto vec = entity.getType().dyn_cast<FVectorType>();
  if (!vec)
    return entity.op->emitError("index access '")
           << index << "' into non-vector type '" << vec << "'";
  entity.fieldIdx += vec.getFieldID(index);
  return success();
}

static LogicalResult parseNextField(StringRef &path, BaseUnion &entity) {
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

static LogicalResult parseAndCheckName(StringRef &path, BaseUnion &entity,
                                       Operation *module) {
  StringRef name = path.take_until([](char c) { return c == '.' || c == '['; });
  path = path.drop_front(name.size());
  if ((entity = findNamedThing(name, module)))
    return success();
  return failure();
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
  std::string newTarget = "~";
  llvm::raw_string_ostream s(newTarget);
  unsigned tokenIdx = 0;
  for (auto a : target) {
    if (a == '.') {
      switch (tokenIdx) {
      case 0:
        s << "|";
        break;
      case 1:
        s << ">";
        break;
      default:
        s << ".";
        break;
      }
      ++tokenIdx;
    } else
      s << a;
  }
  return newTarget;
}

/// Scatter breadcrumb annotations corrosponding to non-local annotations along
/// the instance path.
// FIXME: uniq annotation chain links
static void scatterNonLocalPath(AnnoPathValue target, Attribute key) {
  for (auto inst : llvm::enumerate(target.instances)) {
    SmallVector<NamedAttribute> newAnnoAttrs;
    newAnnoAttrs.push_back(
        {Identifier::get("circt.nonlocal.key", key.getContext()), key});
    if (inst.index() != 0)
      newAnnoAttrs.push_back(
          {Identifier::get("circt.nonlocal.parent", key.getContext()),
           target.instances[inst.index() - 1].nameAttr()});
    if (inst.index() != target.instances.size() - 1)
      newAnnoAttrs.push_back(
          {Identifier::get("circt.nonlocal.child", key.getContext()),
           target.instances[inst.index() + 1].nameAttr()});
    newAnnoAttrs.push_back(
        {Identifier::get("class", key.getContext()),
         StringAttr::get(key.getContext(), "circt.nonlocal")});
    addAnnotation(BaseUnion(inst.value()), newAnnoAttrs);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Standard Utility resolvers and appliers
////////////////////////////////////////////////////////////////////////////////

static Optional<AnnoPathValue> noResolve(DictionaryAttr anno, CircuitOp circuit,
                                         SymbolTable &modules) {
  return AnnoPathValue(circuit);
}

static LogicalResult
ignoreAnno(AnnoPathValue target, DictionaryAttr anno,
           llvm::function_ref<void(ArrayAttr &)> addToWorklist) {
  return success();
}

static Optional<AnnoPathValue>
stdResolveImpl(StringRef rawPath, CircuitOp circuit, SymbolTable &modules) {
  auto pathStr = canonicalizeTarget(rawPath);
  StringRef path{pathStr};

  if (parseAndCheckCircuit(path, circuit).failed())
    return {};
  if (path.empty())
    return AnnoPathValue(circuit);

  Operation *rootMod = nullptr;
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
    circuit.emitError("No target field in annotation ") << anno;
    return {};
  }
  if (!target->second.isa<StringAttr>()) {
    circuit.emitError("Target field in annotation doesn't contain string ")
        << anno;
    return {};
  }
  return stdResolveImpl(target->second.cast<StringAttr>().getValue(), circuit,
                        modules);
}

static Optional<AnnoPathValue>
tryResolve(DictionaryAttr anno, CircuitOp circuit, SymbolTable &modules) {
  auto target = anno.getNamed("target");
  if (target)
    return stdResolveImpl(target->second.cast<StringAttr>().getValue(), circuit,
                          modules);
  return AnnoPathValue(circuit);
}

////////////////////////////////////////////////////////////////////////////////
// Specific Appliers
////////////////////////////////////////////////////////////////////////////////

static LogicalResult applyDirFileNormalizeToCircuit(
    AnnoPathValue target, DictionaryAttr anno,
    llvm::function_ref<void(ArrayAttr &)> addToWorklist) {
  if (!target.isOpOfType<CircuitOp>())
    return failure();
  if (!target.isLocal())
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
  addAnnotation(target.ref, newAnnoAttrs);
  return success();
}

static LogicalResult applyWithoutTargetToTarget(AnnoPathValue target,
                                                DictionaryAttr anno,
                                                bool allowNonLocal) {
  if (!allowNonLocal && !target.isLocal())
    return failure();
  SmallVector<NamedAttribute> newAnnoAttrs;
  for (auto &na : anno)
    if (na.first != "target") {
      newAnnoAttrs.push_back(na);
    } else if (!target.isLocal()) {
      newAnnoAttrs.push_back(
          {Identifier::get("circt.nonlocal.key", anno.getContext()),
           na.second});
      newAnnoAttrs.push_back(
          {Identifier::get("circt.nonlocal.parent", anno.getContext()),
           target.instances.back().nameAttr()});
      scatterNonLocalPath(target, na.second);
    }
  addAnnotation(target.ref, newAnnoAttrs);
  return success();
}

template <bool allowNonLocal = false>
static LogicalResult
applyWithoutTarget(AnnoPathValue target, DictionaryAttr anno,
                   llvm::function_ref<void(ArrayAttr &)> addToWorklist) {
  return applyWithoutTargetToTarget(target, anno, allowNonLocal);
}

template <bool allowNonLocal = false>
static LogicalResult applyWithoutTargetToModule(
    AnnoPathValue target, DictionaryAttr anno,
    llvm::function_ref<void(ArrayAttr &)> addToWorklist) {
  if (!target.isOpOfType<FModuleOp>() && !target.isOpOfType<FExtModuleOp>())
    return failure();
  return applyWithoutTargetToTarget(target, anno, allowNonLocal);
}

template <bool allowNonLocal = false>
static LogicalResult applyWithoutTargetToCircuit(
    AnnoPathValue target, DictionaryAttr anno,
    llvm::function_ref<void(ArrayAttr &)> addToWorklist) {
  if (!target.isOpOfType<CircuitOp>())
    return failure();
  return applyWithoutTargetToTarget(target, anno, allowNonLocal);
}

template <bool allowNonLocal = false>
static LogicalResult
applyWithoutTargetToMem(AnnoPathValue target, DictionaryAttr anno,
                        llvm::function_ref<void(ArrayAttr &)> addToWorklist) {
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

/// Implements the same behavior as DictionaryAttr::getAs<A> to return the value
/// of a specific type associated with a key in a dictionary.  However, this is
/// specialized to print a useful error message, specific to custom annotation
/// process, on failure.
template <typename A>
static A tryGetAs(DictionaryAttr &dict, DictionaryAttr &root, StringRef key,
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
static LogicalResult
applyDontTouch(AnnoPathValue target, DictionaryAttr anno,
               llvm::function_ref<void(ArrayAttr &)> addToWorklist) {
  addNamedAttr(target.ref.op, "firrtl.DoNotTouch");
  return success();
}

// Get a DonotTouch annotation for the given target.
static DictionaryAttr getDontTouchAnno(MLIRContext *context, StringRef target) {
  NamedAttrList dontTouchAnn;
  dontTouchAnn.append(
      "class",
      StringAttr::get(context, "firrtl.transforms.DontTouchAnnotation"));
  if (!target.empty())
    dontTouchAnn.append("target", StringAttr::get(context, target));
  return DictionaryAttr::getWithSorted(context, dontTouchAnn);
}

// Get a new annotation with the given target.
static DictionaryAttr getAnnoWithTarget(MLIRContext *context,
                                        NamedAttrList attr, StringRef target) {
  attr.append("target", StringAttr::get(context, target));
  return DictionaryAttr::get(context, attr);
}

static LogicalResult
applyGrandCentralDataTaps(AnnoPathValue target, DictionaryAttr anno,
                          llvm::function_ref<void(ArrayAttr &)> addToWorklist) {

  addNamedAttr(target.ref.op, "firrtl.DoNotTouch");
  auto classAttr = anno.getAs<StringAttr>("class");
  auto clazz = classAttr.getValue();
  auto loc = target.ref.op->getLoc();
  auto context = target.ref.op->getContext();
  auto id = newID(context);
  SmallVector<Attribute> newAnnos;
  NamedAttrList attrs;
  attrs.append("class", classAttr);
  auto blackBoxAttr = tryGetAs<StringAttr>(anno, anno, "blackBox", loc, clazz);
  if (!blackBoxAttr)
    return failure();
  std::string bbTarget = canonicalizeTarget(blackBoxAttr.getValue());
  if (bbTarget.empty())
    return failure();
  newAnnos.push_back(getDontTouchAnno(context, bbTarget));

  // Process all the taps.
  auto keyAttr = tryGetAs<ArrayAttr>(anno, anno, "keys", loc, clazz);
  if (!keyAttr)
    return failure();
  for (size_t i = 0, e = keyAttr.size(); i != e; ++i) {
    auto b = keyAttr[i];
    auto path = ("keys[" + Twine(i) + "]").str();
    auto bDict = b.cast<DictionaryAttr>();
    auto classAttr =
        tryGetAs<StringAttr>(bDict, anno, "class", loc, clazz, path);
    if (!classAttr)
      return failure();

    // The "portName" field is common across all sub-types of DataTapKey.
    NamedAttrList port;
    auto portNameAttr =
        tryGetAs<StringAttr>(bDict, anno, "portName", loc, clazz, path);
    if (!portNameAttr)
      return failure();
    auto maybePortTarget = canonicalizeTarget(portNameAttr.getValue());
    if (maybePortTarget.empty())
      return failure();
    port.append("class", classAttr);
    port.append("id", id);
    newAnnos.push_back(getDontTouchAnno(context, maybePortTarget));

    if (classAttr.getValue() ==
        "sifive.enterprise.grandcentral.ReferenceDataTapKey") {
      NamedAttrList source;
      auto portID = newID(context);
      source.append("class", bDict.get("class"));
      source.append("id", id);
      source.append("portID", portID);
      auto sourceAttr =
          tryGetAs<StringAttr>(bDict, anno, "source", loc, clazz, path);
      if (!sourceAttr)
        return failure();
      auto maybeSourceTarget = canonicalizeTarget(sourceAttr.getValue());
      if (maybeSourceTarget.empty())
        return failure();
      source.append("type", StringAttr::get(context, "source"));
      newAnnos.push_back(getAnnoWithTarget(context, source, maybeSourceTarget));
      newAnnos.push_back(getDontTouchAnno(context, maybeSourceTarget));

      // Port Annotations generation.
      port.append("portID", portID);
      port.append("type", StringAttr::get(context, "portName"));
      newAnnos.push_back(getAnnoWithTarget(context, port, maybePortTarget));
      continue;
    }

    if (classAttr.getValue() ==
        "sifive.enterprise.grandcentral.DataTapModuleSignalKey") {
      NamedAttrList module;
      auto portID = newID(context);
      module.append("class", classAttr);
      module.append("id", id);
      auto internalPathAttr =
          tryGetAs<StringAttr>(bDict, anno, "internalPath", loc, clazz, path);
      auto moduleAttr =
          tryGetAs<StringAttr>(bDict, anno, "module", loc, clazz, path);
      if (!internalPathAttr || !moduleAttr)
        return failure();
      module.append("internalPath", internalPathAttr);
      module.append("portID", portID);
      auto moduleTarget = canonicalizeTarget(moduleAttr.getValue());
      if (moduleTarget.empty())
        return failure();
      newAnnos.push_back(getAnnoWithTarget(context, module, moduleTarget));
      newAnnos.push_back(getDontTouchAnno(context, moduleTarget));

      // Port Annotations generation.
      port.append("portID", portID);
      newAnnos.push_back(getAnnoWithTarget(context, port, maybePortTarget));
      continue;
    }

    if (classAttr.getValue() ==
        "sifive.enterprise.grandcentral.DeletedDataTapKey") {
      // Port Annotations generation.
      newAnnos.push_back(getAnnoWithTarget(context, port, maybePortTarget));
      continue;
    }

    if (classAttr.getValue() ==
        "sifive.enterprise.grandcentral.LiteralDataTapKey") {
      NamedAttrList literal;
      literal.append("class", classAttr);
      auto literalAttr =
          tryGetAs<StringAttr>(bDict, anno, "literal", loc, clazz, path);
      if (!literalAttr)
        return failure();
      literal.append("literal", literalAttr);

      // Port Annotaiton generation.
      newAnnos.push_back(getAnnoWithTarget(context, literal, maybePortTarget));
      continue;
    }

    mlir::emitError(
        loc, "Annotation '" + Twine(clazz) + "' with path '" + path + ".class" +
                 +"' contained an unknown/unimplemented DataTapKey class '" +
                 classAttr.getValue() + "'.")
            .attachNote()
        << "The full Annotation is reprodcued here: " << anno << "\n";
    return failure();
  }
  ArrayAttr attr = ArrayAttr::get(context, newAnnos);
  addToWorklist(attr);

  // TODO: port scatter logic in FIRAnnotations.cpp
  return applyWithoutTargetToCircuit(target, anno, addToWorklist);
}

static LogicalResult
applyGrandCentralMemTaps(AnnoPathValue target, DictionaryAttr anno,
                         llvm::function_ref<void(ArrayAttr &)> addToWorklist) {
  addNamedAttr(target.ref.op, "firrtl.DoNotTouch");
  // TODO: port scatter logic in FIRAnnotations.cpp
  return applyWithoutTargetToCircuit(target, anno, addToWorklist);
}

static LogicalResult
applyGrandCentralView(AnnoPathValue target, DictionaryAttr anno,
                      llvm::function_ref<void(ArrayAttr &)> addToWorklist) {
  addNamedAttr(target.ref.op, "firrtl.DoNotTouch");
  // TODO: port scatter logic in FIRAnnotations.cpp
  return applyWithoutTargetToCircuit(target, anno, addToWorklist);
}

////////////////////////////////////////////////////////////////////////////////
// Driving table
////////////////////////////////////////////////////////////////////////////////

namespace {
struct AnnoRecord {
  llvm::function_ref<Optional<AnnoPathValue>(DictionaryAttr, CircuitOp,
                                             SymbolTable &)>
      resolver;
  llvm::function_ref<LogicalResult(AnnoPathValue, DictionaryAttr,
                                   llvm::function_ref<void(ArrayAttr &)>)>
      applier;
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
    {"sifive.enterprise.grandcentral.ReferenceDataTapKey",
     {stdResolve, applyWithoutTarget<>}},
    {"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
     {stdResolve, applyWithoutTarget<>}},
    {"sifive.enterprise.grandcentral.DeletedDataTapKey",
     {stdResolve, applyWithoutTarget<>}},
    {"sifive.enterprise.grandcentral.LiteralDataTapKey",
     {stdResolve, applyWithoutTarget<>}},
    {"sifive.enterprise.grandcentral.DeletedDataTapKey",
     {stdResolve, applyWithoutTarget<>}},

    // Testing Annotation
    {"circt.test", {stdResolve, applyWithoutTarget<true>}},
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
  void addToWorklist(ArrayAttr &annos);

  bool ignoreUnhandledAnno = false;
  bool ignoreClasslessAnno = false;
  static size_t annotationID;
  SmallVector<ArrayAttr> worklistAttrs;
};
} // end anonymous namespace
size_t LowerAnnotationsPass::annotationID = 0;

// Add the annotations to worklist.
void LowerAnnotationsPass::addToWorklist(ArrayAttr &annos) {
  LowerAnnotationsPass::worklistAttrs.push_back(annos);
}

// Generate a unique ID.
IntegerAttr newID(MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(context, 64),
                          LowerAnnotationsPass::annotationID++);
}
// This is the main entrypoint for the lowering pass.
void LowerAnnotationsPass::runOnOperation() {
  CircuitOp circuit = getOperation();
  SymbolTable modules(circuit);
  worklistAttrs.push_back(circuit.annotations());
  circuit.annotationsAttr(ArrayAttr::get(circuit.getContext(), {}));
  size_t numFailures = 0;
  while (!worklistAttrs.empty()) {
    auto attrs = worklistAttrs.pop_back_val();
    for (auto attr : attrs)
      if (applyAnnotation(attr.cast<DictionaryAttr>(), circuit, modules)
              .failed())
        ++numFailures;
  }
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

  auto addToWorklist = [&](ArrayAttr &ann) { worklistAttrs.push_back(ann); };
  auto target = record->resolver(anno, circuit, modules);
  if (!target)
    return circuit.emitError("Unable to resolve target of annotation: ")
           << anno;
  if (record->applier(*target, anno, addToWorklist).failed())
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
