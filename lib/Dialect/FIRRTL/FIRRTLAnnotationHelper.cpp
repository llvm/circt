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
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using llvm::StringRef;

// Some types have been expanded so the first layer of aggregate path is
// a return value.
static LogicalResult updateExpandedPort(StringRef field, AnnoTarget &ref) {
  if (auto mem = dyn_cast<MemOp>(ref.getOp()))
    for (size_t p = 0, pe = mem.getPortNames().size(); p < pe; ++p)
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
        << index << "' into non-vector type '" << type << "'";
    return failure();
  }
  return index;
}

static FailureOr<unsigned> findFieldID(AnnoTarget &ref,
                                       ArrayRef<TargetToken> tokens) {
  if (tokens.empty())
    return 0;

  auto *op = ref.getOp();
  auto fieldIdx = 0;
  // The first field for some ops refers to expanded return values.
  if (isa<MemOp>(ref.getOp())) {
    if (failed(updateExpandedPort(tokens.front().name, ref)))
      return {};
    tokens = tokens.drop_front();
  }

  auto type = ref.getType();
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
  if (!path.circuit.empty() && circuit.getName() != path.circuit) {
    mlir::emitError(circuit.getLoc())
        << "circuit name doesn't match annotation '" << path.circuit << '\'';
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
      mlir::emitError(circuit.getLoc())
          << "module doesn't exist '" << p.first << '\'';
      return {};
    }
    auto resolved = cache.lookup(mod, p.second);
    if (!resolved || !isa<InstanceOp>(resolved.getOp())) {
      mlir::emitError(circuit.getLoc()) << "cannot find instance '" << p.second
                                        << "' in '" << mod.getName() << "'";
      return {};
    }
    instances.push_back(cast<InstanceOp>(resolved.getOp()));
  }
  // The final module is where the named target is (or is the named target).
  auto mod = symTbl.lookup<FModuleLike>(path.module);
  if (!mod) {
    mlir::emitError(circuit.getLoc())
        << "module doesn't exist '" << path.module << '\'';
    return {};
  }
  AnnoTarget ref;
  if (path.name.empty()) {
    assert(path.component.empty());
    ref = OpAnnoTarget(mod);
  } else {
    ref = cache.lookup(mod, path.name);
    if (!ref) {
      mlir::emitError(circuit.getLoc())
          << "cannot find name '" << path.name << "' in " << mod.moduleName();
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
    } else if (component.front().isIndex) {
      mlir::emitError(circuit.getLoc())
          << "illegal target '" << path.str() << "' indexes into an instance";
      return {};
    } else {
      auto field = component.front().name;
      ref = AnnoTarget();
      for (size_t p = 0, pe = target.getNumPorts(); p < pe; ++p)
        if (target.getPortName(p) == field) {
          ref = PortAnnoTarget(target, p);
          break;
        }
      if (!ref) {
        mlir::emitError(circuit.getLoc())
            << "!cannot find port '" << field << "' in module "
            << target.moduleName();
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
  // An empty string is not a legal target.
  if (origTarget.empty())
    return {};
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
    mlir::emitError(circuit.getLoc())
        << "Cannot tokenize annotation path " << rawPath;
    return {};
  }

  return resolveEntities(*tokens, circuit, symTbl, cache);
}

InstanceOp firrtl::addPortsToModule(
    FModuleLike mod, InstanceOp instOnPath, FIRRTLType portType, Direction dir,
    StringRef newName, InstancePathCache &instancePathcache,
    llvm::function_ref<ModuleNamespace &(FModuleLike)> getNamespace,
    CircuitTargetCache *targetCaches) {
  // To store the cloned version of `instOnPath`.
  InstanceOp clonedInstOnPath;
  // Get a new port name from the Namespace.
  auto portName = [&](FModuleLike nameForMod) {
    return StringAttr::get(nameForMod.getContext(),
                           getNamespace(nameForMod).newName("_gen_" + newName));
  };
  // The port number for the new port.
  unsigned portNo = mod.getNumPorts();
  PortInfo portInfo = {portName(mod), portType, dir, {}, mod.getLoc()};
  mod.insertPorts({{portNo, portInfo}});
  if (targetCaches)
    targetCaches->insertPort(mod, portNo);
  // Now update all the instances of `mod`.
  for (auto *use : instancePathcache.instanceGraph
                       .lookup(cast<hw::HWModuleLike>((Operation *)mod))
                       ->uses()) {
    InstanceOp useInst = cast<InstanceOp>(use->getInstance());
    auto clonedInst = useInst.cloneAndInsertPorts({{portNo, portInfo}});
    if (useInst == instOnPath)
      clonedInstOnPath = clonedInst;
    // Update all occurences of old instance.
    instancePathcache.replaceInstance(useInst, clonedInst);
    if (targetCaches)
      targetCaches->replaceOp(useInst, clonedInst);
    useInst->replaceAllUsesWith(clonedInst.getResults().drop_back());
    useInst->erase();
  }
  return clonedInstOnPath;
}

Value firrtl::borePortsOnPath(
    SmallVector<InstanceOp> &instancePath, FModuleOp lcaModule, Value fromVal,
    StringRef newNameHint, InstancePathCache &instancePathcache,
    llvm::function_ref<ModuleNamespace &(FModuleLike)> getNamespace,
    CircuitTargetCache *targetCachesInstancePathCache) {
  // Create connections of the `fromVal` by adding ports through all the
  // instances of `instancePath`. `instancePath` specifies a path from a source
  // module through the lcaModule till a target module. source module is the
  // parent module of `fromval` and target module is the referenced target
  // module of the last instance in the instancePath.

  // If instancePath empty then just return the `fromVal`.
  if (instancePath.empty())
    return fromVal;

  FModuleOp srcModule;
  if (auto block = fromVal.dyn_cast<BlockArgument>())
    srcModule = cast<FModuleOp>(block.getOwner()->getParentOp());
  else
    srcModule = fromVal.getDefiningOp()->getParentOfType<FModuleOp>();

  // If the `srcModule` is the `lcaModule`, then we only need to drill input
  // ports starting from the `lcaModule` till the end of `instancePath`. For all
  // other cases, we start to drill output ports from the `srcModule` till the
  // `lcaModule`, and then input ports from `lcaModule` till the end of
  // `instancePath`.
  Direction dir = Direction::Out;
  if (srcModule == lcaModule)
    dir = Direction::In;

  auto valType = fromVal.getType().cast<FIRRTLType>();
  auto forwardVal = fromVal;
  for (auto instOnPath : instancePath) {
    auto referencedModule = cast<FModuleOp>(
        instancePathcache.instanceGraph.getReferencedModule(instOnPath));
    unsigned portNo = referencedModule.getNumPorts();
    // Add a new port to referencedModule, and all its uses/instances.
    InstanceOp clonedInst = addPortsToModule(
        referencedModule, instOnPath, valType, dir, newNameHint,
        instancePathcache, getNamespace, targetCachesInstancePathCache);
    //  `instOnPath` will be erased and replaced with `clonedInst`. So, there
    //  should be no more use of `instOnPath`.
    auto clonedInstRes = clonedInst.getResult(portNo);
    auto referencedModNewPort = referencedModule.getArgument(portNo);
    // If out direction, then connect `forwardVal` to the `referencedModNewPort`
    // (the new port of referenced module) else set the new input port of the
    // cloned instance with `forwardVal`. If out direction, then set the
    // `forwardVal` to the result of the cloned instance, else set the
    // `forwardVal` to the new port of the referenced module.
    if (dir == Direction::Out) {
      auto builder = ImplicitLocOpBuilder::atBlockEnd(
          forwardVal.getLoc(), forwardVal.isa<BlockArgument>()
                                   ? referencedModule.getBodyBlock()
                                   : forwardVal.getDefiningOp()->getBlock());
      builder.create<ConnectOp>(referencedModNewPort, forwardVal);
      forwardVal = clonedInstRes;
    } else {
      auto builder = ImplicitLocOpBuilder::atBlockEnd(clonedInst.getLoc(),
                                                      clonedInst->getBlock());
      builder.create<ConnectOp>(clonedInstRes, forwardVal);
      forwardVal = referencedModNewPort;
    }

    // Switch direction of reached the `lcaModule`.
    if (clonedInst->getParentOfType<FModuleOp>() == lcaModule)
      dir = Direction::In;
  }
  return forwardVal;
}

//===----------------------------------------------------------------------===//
// AnnoTargetCache
//===----------------------------------------------------------------------===//

void AnnoTargetCache::gatherTargets(FModuleLike mod) {
  // Add ports
  for (const auto &p : llvm::enumerate(mod.getPorts()))
    targets.insert({p.value().name, PortAnnoTarget(mod, p.index())});

  // And named things
  mod.walk([&](Operation *op) { insertOp(op); });
}

LogicalResult
firrtl::findLCAandSetPath(AnnoPathValue &srcTarget, AnnoPathValue &dstTarget,
                          SmallVector<InstanceOp> &pathFromSrcToDst,
                          FModuleOp &lcaModule, ApplyState &state) {

  auto srcModule = cast<FModuleOp>(srcTarget.ref.getModule().getOperation());
  auto dstModule = cast<FModuleOp>(dstTarget.ref.getModule().getOperation());
  // Get the path from top to `srcModule` and `dstModule`, and then compare the
  // path to find the lca. Then use that to get the path from `srcModule` to the
  // `dstModule`.
  pathFromSrcToDst.clear();
  if (srcModule == dstModule) {
    lcaModule = srcModule;
    return success();
  }
  auto *top = state.instancePathCache.instanceGraph.getTopLevelNode();
  lcaModule = cast<FModuleOp>(top->getModule().getOperation());
  auto initializeInstances = [&](SmallVector<InstanceOp> &instances,
                                 FModuleLike targetModule) -> LogicalResult {
    if (!instances.empty())
      return success();
    auto instancePathsFromTop = state.instancePathCache.getAbsolutePaths(
        cast<hw::HWModuleLike>(*targetModule));
    if (instancePathsFromTop.size() > 1)
      return targetModule->emitError("cannot handle multiple paths to target");

    // Get the path from top to dst
    for (auto inst : instancePathsFromTop.back())
      instances.push_back(cast<InstanceOp>(inst));
    return success();
  };
  if (initializeInstances(dstTarget.instances, dstModule).failed() ||
      initializeInstances(srcTarget.instances, srcModule).failed())
    return failure();

  auto &dstPathFromTop = dstTarget.instances;
  // A map of the modules in the path from top to dst to its index into the
  // `dstPathFromTop`.
  DenseMap<FModuleOp, size_t> dstPathMap;
  // Initialize the leaf module, to end of path.
  dstPathMap[dstModule] = dstPathFromTop.size();
  for (auto &dstInstance : llvm::enumerate(dstPathFromTop))
    dstPathMap[dstInstance.value()->getParentOfType<FModuleOp>()] =
        dstInstance.index();
  auto *dstPathIterator = dstPathFromTop.begin();
  // Now, reverse iterate over the path of the source, from the source module to
  // the Top.
  for (auto srcInstPath : llvm::reverse(srcTarget.instances)) {
    auto refModule = cast<FModuleOp>(
        state.instancePathCache.instanceGraph.getReferencedModule(srcInstPath)
            .getOperation());
    auto mapIter = dstPathMap.find(refModule);
    // If `refModule` exists on the dst path, then this is the Lowest Common
    // Ancestor between the source and dst module.
    if (mapIter != dstPathMap.end()) {
      lcaModule = refModule;
      dstPathIterator += mapIter->getSecond();
      break;
    }
    pathFromSrcToDst.push_back(srcInstPath);
  }
  pathFromSrcToDst.insert(pathFromSrcToDst.end(), dstPathIterator,
                          dstPathFromTop.end());
  return success();
}
