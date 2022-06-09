//===- FIRRTLAnnotationHelper.cpp - FIRRTL Annotation Lookup ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helpers mapping annotations to operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using llvm::StringRef;

// Some types have been expanded so the first layer of aggregate path is
// a return value.
static LogicalResult updateExpandedPort(StringRef field, AnnoTarget &ref) {
  if (auto mem = dyn_cast<MemOp>(ref.getOp()))
    for (size_t p = 0, pe = mem.portNames().size(); p < pe; ++p)
      if (mem.getPortNameStr(p) == field) {
        ref = PortAnnoTarget(mem, p);
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

static FailureOr<unsigned> findFieldID(AnnoTarget &ref,
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

void TokenAnnoTarget::toVector(SmallVectorImpl<char> &out) const {
  out.push_back('~');
  out.append(circuit.begin(), circuit.end());
  out.push_back('|');
  for (auto modInstPair : instances) {
    out.append(modInstPair.first.begin(), modInstPair.first.end());
    out.push_back('/');
    out.append(modInstPair.second.begin(), modInstPair.second.end());
    out.push_back(':');
  }
  out.append(module.begin(), module.end());
  if (name.empty())
    return;
  out.push_back('>');
  out.append(name.begin(), name.end());
  for (auto comp : component) {
    out.push_back(comp.isIndex ? '[' : '.');
    out.append(comp.name.begin(), comp.name.end());
    if (comp.isIndex)
      out.push_back(']');
  }
}

std::string firrtl::canonicalizeTarget(StringRef target) {

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

Optional<AnnoPathValue> firrtl::resolveEntities(TokenAnnoTarget path,
                                                CircuitOp circuit,
                                                SymbolTable &symTbl,
                                                CircuitTargetCache &cache) {
  // Validate circuit name.
  if (!path.circuit.empty() && circuit.name() != path.circuit) {
    circuit->emitError("circuit name doesn't match annotation '")
        << path.circuit << '\'';
    return {};
  }
  // Circuit only target.
  if (path.module.empty()) {
    assert(path.name.empty() && path.instances.empty() &&
           path.component.empty());
    return AnnoPathValue(circuit);
  }

  // Resolve all instances for non-local paths.
  SmallVector<InstanceOp> instances;
  for (auto p : path.instances) {
    auto mod = symTbl.lookup<FModuleOp>(p.first);
    if (!mod) {
      circuit->emitError("module doesn't exist '") << p.first << '\'';
      return {};
    }
    auto resolved = cache.lookup(mod, p.second);
    if (!resolved || !isa<InstanceOp>(resolved.getOp())) {
      circuit.emitError("cannot find instance '")
          << p.second << "' in '" << mod.getName() << "'";
      return {};
    }
    instances.push_back(cast<InstanceOp>(resolved.getOp()));
  }
  // The final module is where the named target is (or is the named target).
  auto mod = symTbl.lookup<FModuleLike>(path.module);
  if (!mod) {
    circuit->emitError("module doesn't exist '") << path.module << '\'';
    return {};
  }
  AnnoTarget ref;
  if (path.name.empty()) {
    assert(path.component.empty());
    ref = OpAnnoTarget(mod);
  } else {
    ref = cache.lookup(mod, path.name);
    if (!ref) {
      circuit->emitError("cannot find name '")
          << path.name << "' in " << mod.moduleName();
      return {};
    }
  }

  // If the reference is pointing to an instance op, we have to move the target
  // to the module.  This is done both because it is logical to have one
  // representation (this effectively canonicalizes a reference target on an
  // instance into an instance target) and because the SFC has a pass that does
  // this conversion.  E.g., this is converting (where "bar" is an instance):
  //   ~Foo|Foo>bar
  // Into:
  //   ~Foo|Foo/bar:Bar
  ArrayRef<TargetToken> component(path.component);
  if (auto instance = dyn_cast<InstanceOp>(ref.getOp())) {
    instances.push_back(instance);
    auto target = instance.getReferencedModule(symTbl);
    if (component.empty()) {
      ref = OpAnnoTarget(instance.getReferencedModule(symTbl));
    } else {
      auto field = component.front().name;
      ref = AnnoTarget();
      for (size_t p = 0, pe = target.getNumPorts(); p < pe; ++p)
        if (target.getPortName(p) == field) {
          ref = PortAnnoTarget(target, p);
          break;
        }
      if (!ref) {
        circuit->emitError("!cannot find port '")
            << field << "' in module " << target.moduleName();
        return {};
      }
      component = component.drop_front();
    }
  }

  // If we have aggregate specifiers, resolve those now. This call can update
  // the ref to target a port of a memory.
  auto result = findFieldID(ref, component);
  if (failed(result))
    return {};
  auto fieldIdx = *result;

  return AnnoPathValue(instances, ref, fieldIdx);
}

/// split a target string into it constituent parts.  This is the primary parser
/// for targets.
Optional<TokenAnnoTarget> firrtl::tokenizePath(StringRef origTarget) {
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

Optional<AnnoPathValue> firrtl::resolvePath(StringRef rawPath,
                                            CircuitOp circuit,
                                            SymbolTable &symTbl,
                                            CircuitTargetCache &cache) {
  auto pathStr = canonicalizeTarget(rawPath);
  StringRef path{pathStr};

  auto tokens = tokenizePath(path);
  if (!tokens) {
    circuit->emitError("Cannot tokenize annotation path ") << rawPath;
    return {};
  }

  return resolveEntities(*tokens, circuit, symTbl, cache);
}

//===----------------------------------------------------------------------===//
// AnnoTargetCache
//===----------------------------------------------------------------------===//

void AnnoTargetCache::gatherTargets(FModuleLike mod) {
  // Add ports
  for (auto p : llvm::enumerate(mod.getPorts()))
    targets.insert({p.value().name, PortAnnoTarget(mod, p.index())});

  // And named things
  mod.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<InstanceOp, MemOp, NodeOp, RegOp, RegResetOp, WireOp, CombMemOp,
              SeqMemOp, MemoryPortOp>([&](auto op) {
          // To be safe, check attribute and non-empty name before adding.
          if (auto name = op.nameAttr(); name && !name.getValue().empty())
            targets.insert({name, OpAnnoTarget(op)});
        });
  });
}
