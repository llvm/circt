//===- ResolveXmrRef.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_RESOLVEXMRREF
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

static void appendEscaped(std::string &out, StringRef text) {
  for (char c : text) {
    if (c == '\\' || c == '|' || c == ':')
      out.push_back('\\');
    out.push_back(c);
  }
}

static std::string buildPathSuffixKey(ArrayAttr pathArray, unsigned index) {
  std::string key;
  for (unsigned i = index, e = pathArray.size(); i < e; ++i) {
    auto ref = cast<hw::InnerRefAttr>(pathArray[i]);
    appendEscaped(key, ref.getModule().getValue());
    key.push_back(':');
    appendEscaped(key, ref.getName().getValue());
    key.push_back('|');
  }
  return key;
}

static std::string buildBoreKey(hw::HWModuleOp childMod, ArrayAttr pathArray,
                                unsigned index, Type targetType) {
  std::string key;
  appendEscaped(key, childMod.getModuleName());
  key.push_back('|');
  key.append(buildPathSuffixKey(pathArray, index));
  key.push_back('|');
  llvm::raw_string_ostream os(key);
  targetType.print(os);
  os.flush();
  return key;
}

static bool isConditionallyGuarded(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<sv::IfDefOp>(parent))
      return true;
  return false;
}

static std::string buildCaptureKey(hw::HWModuleOp payloadMod,
                                   ArrayAttr pathArray, Type targetType) {
  std::string key;
  appendEscaped(key, payloadMod.getModuleName());
  key.push_back('|');
  key.append(buildPathSuffixKey(pathArray, 0));
  key.push_back('|');
  llvm::raw_string_ostream os(key);
  targetType.print(os);
  os.flush();
  return key;
}

namespace {
struct ResolveXMRRefPass
    : public arc::impl::ResolveXMRRefBase<ResolveXMRRefPass> {
  using Base = arc::impl::ResolveXMRRefBase<ResolveXMRRefPass>;
  using Base::Base;

  llvm::StringMap<unsigned> borePortByKey;
  llvm::StringMap<unsigned> captureInputPortByKey;
  llvm::StringMap<unsigned> nextBoreOrdinalByModule;
  bool bindContextDiagnosedFailure = false;

  void runOnOperation() override;
  Value boreRecursively(ArrayAttr pathArray, unsigned index,
                        hw::HWModuleOp currentMod, SymbolTable &symTable,
                        StringAttr targetSym, Type targetType);
  Value getInstancePortValue(hw::InstanceOp inst, unsigned portIdx,
                             mlir::Operation *moduleOp);
  std::string createUniqueBoredPortName(hw::HWModuleOp mod,
                                        StringAttr targetSym);
  std::string createUniqueCapturePortName(hw::HWModuleOp mod,
                                          StringAttr targetSym);
  enum class XMRUseMode { ReadOnly, Unsupported };
  XMRUseMode classifyXMRUse(sv::XMRRefOp xmrRefOp);
  Value resolveViaBindContext(sv::XMRRefOp xmrRefOp, hw::HierPathOp pathOp,
                              hw::HWModuleOp payloadMod, SymbolTable &symTable,
                              Type targetType);
};
} // namespace

ResolveXMRRefPass::XMRUseMode
ResolveXMRRefPass::classifyXMRUse(sv::XMRRefOp xmrRefOp) {
  bool hasRead = false;
  for (Operation *user : xmrRefOp->getUsers()) {
    if (isa<sv::ReadInOutOp>(user)) {
      hasRead = true;
      continue;
    }

    xmrRefOp.emitError()
        << "unsupported sv.xmr.ref use by '" << user->getName()
        << "'; only read-only uses (sv.read_inout) are supported";
    return XMRUseMode::Unsupported;
  }

  if (!hasRead) {
    xmrRefOp.emitError("sv.xmr.ref has no read uses; write/other uses are not "
                       "supported yet");
    return XMRUseMode::Unsupported;
  }

  return XMRUseMode::ReadOnly;
}

std::string ResolveXMRRefPass::createUniqueBoredPortName(hw::HWModuleOp mod,
                                                         StringAttr targetSym) {
  auto moduleName = mod.getModuleName();
  unsigned &ordinal = nextBoreOrdinalByModule[moduleName];

  while (true) {
    std::string candidate = "xmr_bored_" + targetSym.getValue().str() + "_" +
                            std::to_string(ordinal++);

    bool exists = false;
    for (unsigned i = 0, e = mod.getNumPorts(); i < e; ++i) {
      if (mod.getPort(i).name.getValue() == candidate) {
        exists = true;
        break;
      }
    }
    if (!exists)
      return candidate;
  }
}

std::string
ResolveXMRRefPass::createUniqueCapturePortName(hw::HWModuleOp mod,
                                               StringAttr targetSym) {
  auto moduleName = mod.getModuleName();
  unsigned &ordinal = nextBoreOrdinalByModule[moduleName];

  while (true) {
    std::string candidate = "xmr_capture_" + targetSym.getValue().str() + "_" +
                            std::to_string(ordinal++);

    bool exists = false;
    for (unsigned i = 0, e = mod.getNumPorts(); i < e; ++i) {
      if (mod.getPort(i).name.getValue() == candidate) {
        exists = true;
        break;
      }
    }
    if (!exists)
      return candidate;
  }
}

Value ResolveXMRRefPass::resolveViaBindContext(sv::XMRRefOp xmrRefOp,
                                               hw::HierPathOp pathOp,
                                               hw::HWModuleOp payloadMod,
                                               SymbolTable &symTable,
                                               Type targetType) {
  bindContextDiagnosedFailure = false;
  auto pathArray = pathOp.getNamepath();
  auto root = dyn_cast<hw::InnerRefAttr>(pathArray[0]);
  if (!root)
    return nullptr;

  SmallVector<hw::InstanceOp> bindInstances;
  bool hasConditionalBind = false;
  auto module = getOperation();
  module.walk([&](sv::BindOp bindOp) {
    auto boundInst = bindOp.getReferencedInstance(nullptr);
    if (!boundInst)
      return;

    if (boundInst.getModuleName() != payloadMod.getModuleName())
      return;

    if (bindOp.getInstance().getModule() != root.getModule())
      return;

    if (isConditionallyGuarded(bindOp)) {
      hasConditionalBind = true;
      return;
    }

    bindInstances.push_back(boundInst);
  });

  if (hasConditionalBind) {
    xmrRefOp.emitError("cannot resolve XMR through conditionally guarded "
                       "sv.bind; only unconditional sv.bind is supported");
    bindContextDiagnosedFailure = true;
    return nullptr;
  }

  if (bindInstances.empty())
    return nullptr;

  if (bindInstances.size() != 1) {
    xmrRefOp.emitError("bind-context XMR requires a unique unconditional "
                       "sv.bind instance");
    bindContextDiagnosedFailure = true;
    return nullptr;
  }

  auto hostModule = bindInstances.front()->getParentOfType<hw::HWModuleOp>();
  if (!hostModule || hostModule.getModuleName() != root.getModule()) {
    xmrRefOp.emitError("bind host module mismatch while resolving XMR path");
    bindContextDiagnosedFailure = true;
    return nullptr;
  }

  Value hostValue = boreRecursively(pathArray, 0, hostModule, symTable,
                                    pathOp.ref(), targetType);
  if (!hostValue)
    return nullptr;

  llvm::SmallPtrSet<Operation *, 8> bindInstSet;
  for (auto inst : bindInstances)
    bindInstSet.insert(inst.getOperation());

  auto uses = SymbolTable::getSymbolUses(payloadMod, module);
  if (!uses)
    return nullptr;

  for (auto use : *uses) {
    auto userInst = dyn_cast<hw::InstanceOp>(use.getUser());
    if (!userInst)
      continue;
    if (!bindInstSet.contains(userInst.getOperation())) {
      xmrRefOp.emitError("payload module has non-bind instances; refusing to "
                         "resolve bind-context XMR ambiguously");
      bindContextDiagnosedFailure = true;
      return nullptr;
    }
  }

  std::string captureKey = buildCaptureKey(payloadMod, pathArray, targetType);
  unsigned inputPortIdx;
  auto captureIt = captureInputPortByKey.find(captureKey);
  if (captureIt == captureInputPortByKey.end()) {
    std::string portName =
        createUniqueCapturePortName(payloadMod, pathOp.ref());

    OpBuilder b(payloadMod.getContext());
    auto appended =
        payloadMod.appendInput(b.getStringAttr(portName), targetType);
    inputPortIdx = appended.second.getArgNumber();
    captureInputPortByKey[captureKey] = inputPortIdx;

    SmallVector<hw::InstanceOp> instances;
    instances.reserve(bindInstSet.size());
    for (auto use : *uses)
      if (auto userInst = dyn_cast<hw::InstanceOp>(use.getUser()))
        instances.push_back(userInst);

    for (auto userInst : instances) {
      SmallVector<Value> operands(userInst.getOperands());
      if (!bindInstSet.contains(userInst.getOperation())) {
        xmrRefOp.emitError("payload module has non-bind instances; refusing to "
                           "resolve bind-context XMR ambiguously");
        bindContextDiagnosedFailure = true;
        return nullptr;
      }
      operands.push_back(hostValue);

      OpBuilder ib(userInst);
      auto newInst = hw::InstanceOp::create(
          ib, userInst.getLoc(), payloadMod, userInst.getInstanceNameAttr(),
          operands, userInst.getParameters(), userInst.getInnerSymAttr());

      for (unsigned i = 0; i < userInst.getNumResults(); ++i)
        userInst.getResult(i).replaceAllUsesWith(newInst.getResult(i));

      if (userInst.getDoNotPrint())
        newInst.setDoNotPrintAttr(UnitAttr::get(newInst.getContext()));

      userInst.erase();
    }

    return payloadMod.getBodyBlock()->getArgument(inputPortIdx);
  } else {
    inputPortIdx = captureIt->second;

    auto bindInst = bindInstances.front();
    if (inputPortIdx >= bindInst.getNumOperands()) {
      xmrRefOp.emitError("bind instance is missing required capture input "
                         "operand for previously materialized XMR");
      bindContextDiagnosedFailure = true;
      return nullptr;
    }

    return bindInst.getOperand(inputPortIdx);
  }
}

void ResolveXMRRefPass::runOnOperation() {
  auto module = getOperation();
  SymbolTable symTable(module);
  borePortByKey.clear();
  captureInputPortByKey.clear();
  nextBoreOrdinalByModule.clear();
  bool failed = false;

  SmallVector<Operation *> opsToErase;

  module.walk([&](sv::XMRRefOp xmrRefOp) {
    if (failed)
      return;

    if (classifyXMRUse(xmrRefOp) != XMRUseMode::ReadOnly) {
      failed = true;
      return;
    }

    auto pathOp = xmrRefOp.getReferencedPath(nullptr);
    if (!pathOp) {
      xmrRefOp.emitError("unable to resolve path for XMR reference");
      failed = true;
      return;
    }

    StringAttr leafModName = pathOp.leafMod();
    Operation *leafMod = symTable.lookup(leafModName);
    if (!leafMod) {
      xmrRefOp.emitError("leaf module not found in symbol table");
      failed = true;
      return;
    }

    OpBuilder builder(xmrRefOp);
    Value resolvedValue = nullptr;

    Type targetType = cast<hw::InOutType>(xmrRefOp.getType()).getElementType();

    ArrayAttr pathArray = pathOp.getNamepath();
    auto currentModule = xmrRefOp->getParentOfType<hw::HWModuleOp>();

    resolvedValue = boreRecursively(pathArray, 0, currentModule, symTable,
                                    pathOp.ref(), targetType);

    if (!resolvedValue && pathOp.isComponent()) {
      resolvedValue = resolveViaBindContext(xmrRefOp, pathOp, currentModule,
                                            symTable, targetType);
      if (!resolvedValue && bindContextDiagnosedFailure) {
        failed = true;
        return;
      }
    }

    if (!resolvedValue) {
      if (isa<hw::HWModuleExternOp>(leafMod)) {
        xmrRefOp.emitError("unable to resolve XMR into internal blackbox "
                           "symbol; rerun with "
                           "--arc-resolve-xmr=lower-blackbox-internal-to-zero "
                           "to force zero-lowering");
        failed = true;
        return;
      }
      if (pathOp.isComponent()) {
        xmrRefOp.emitError("unable to resolve component path");
        failed = true;
        return;
      }
      xmrRefOp.emitError("unsupported XMR reference type");
      failed = true;
      return;
    }

    if (resolvedValue) {
      for (OpOperand &use :
           llvm::make_early_inc_range(xmrRefOp.getResult().getUses())) {
        if (auto readOp = dyn_cast<sv::ReadInOutOp>(use.getOwner())) {
          readOp.getResult().replaceAllUsesWith(resolvedValue);
          opsToErase.push_back(readOp);
        }
      }
      opsToErase.push_back(xmrRefOp);
    }
  });

  if (failed)
    return signalPassFailure();

  module.walk([&](hw::HierPathOp pathOp) {
    if (pathOp->use_empty()) {
      opsToErase.push_back(pathOp);
    }
  });

  for (Operation *op : opsToErase)
    op->erase();
}

Value ResolveXMRRefPass::boreRecursively(ArrayAttr pathArray, unsigned index,
                                         hw::HWModuleOp currentMod,
                                         SymbolTable &symTable,
                                         StringAttr targetSym,
                                         Type targetType) {
  auto innerRef = cast<hw::InnerRefAttr>(pathArray[index]);
  StringAttr symName = innerRef.getName();

  if (index == pathArray.size() - 1) {
    hw::InnerSymbolTable ist(currentMod);
    auto target = ist.lookup(symName);
    if (!target)
      return nullptr;

    if (target.isPort()) {
      unsigned pIdx = target.getPort();
      if (currentMod.getPort(pIdx).isOutput()) {
        auto outOp =
            cast<hw::OutputOp>(currentMod.getBodyBlock()->getTerminator());
        unsigned oIdx = 0;
        for (unsigned i = 0; i < pIdx; ++i)
          if (currentMod.getPort(i).isOutput())
            oIdx++;
        return outOp.getOperand(oIdx);
      }
      unsigned aIdx = 0;
      for (unsigned i = 0; i < pIdx; ++i)
        if (!currentMod.getPort(i).isOutput())
          aIdx++;
      return currentMod.getBodyBlock()->getArgument(aIdx);
    }
    return target.getOp()->getNumResults() > 0 ? target.getOp()->getResult(0)
                                               : nullptr;
  }

  hw::InnerSymbolTable currentIST(currentMod);
  auto instOp =
      dyn_cast_or_null<hw::InstanceOp>(currentIST.lookup(symName).getOp());
  if (!instOp)
    return nullptr;

  Operation *childModOp = symTable.lookup(instOp.getModuleNameAttr().getAttr());
  if (!childModOp)
    return nullptr;

  auto nextRef = cast<hw::InnerRefAttr>(pathArray[index + 1]);
  StringAttr nextSym = nextRef.getName();

  hw::InnerSymbolTable childIST(childModOp);
  auto nextTarget = childIST.lookup(nextSym);

  if (nextTarget && nextTarget.isPort()) {
    return getInstancePortValue(instOp, nextTarget.getPort(), childModOp);
  }

  if (isa<hw::HWModuleExternOp>(childModOp)) {
    if (lowerBlackBoxInternalToZero) {
      instOp.emitWarning() << "XMR target '" << nextSym
                           << "' is internal to blackbox. Lowering to 0.";
      OpBuilder b(instOp);
      return hw::ConstantOp::create(b, instOp.getLoc(), targetType, 0);
    }
    return nullptr;
  }

  auto childMod = cast<hw::HWModuleOp>(childModOp);
  std::string boreKey =
      buildBoreKey(childMod, pathArray, index + 1, targetType);
  if (auto it = borePortByKey.find(boreKey); it != borePortByKey.end()) {
    if (it->second >= instOp.getNumResults())
      return nullptr;
    return instOp.getResult(it->second);
  }

  Value childVal = boreRecursively(pathArray, index + 1, childMod, symTable,
                                   targetSym, targetType);
  if (!childVal)
    return nullptr;

  std::string portName = createUniqueBoredPortName(childMod, targetSym);

  OpBuilder mb(childMod.getContext());
  hw::PortInfo newPort;
  newPort.name = mb.getStringAttr(portName);
  newPort.dir = hw::ModulePort::Direction::Output;
  newPort.type = targetType;

  SmallVector<std::pair<unsigned, hw::PortInfo>> newOutputs;
  newOutputs.push_back({childMod.getNumOutputPorts(), newPort});
  childMod.modifyPorts({}, newOutputs, {}, {});

  auto outOp = cast<hw::OutputOp>(childMod.getBodyBlock()->getTerminator());
  outOp->insertOperands(outOp->getNumOperands(), childVal);
  unsigned resIdx = childMod.getNumOutputPorts() - 1;
  borePortByKey[boreKey] = resIdx;

  auto top = getOperation();
  auto uses = SymbolTable::getSymbolUses(childMod, top);
  if (!uses)
    return nullptr;

  SmallVector<hw::InstanceOp> instances;
  for (auto use : *uses)
    if (auto userInst = dyn_cast<hw::InstanceOp>(use.getUser()))
      instances.push_back(userInst);

  Value boredValue;
  for (auto userInst : instances) {
    SmallVector<Value> operands(userInst.getOperands());
    OpBuilder b(userInst);
    auto newInst = hw::InstanceOp::create(
        b, userInst.getLoc(), childMod, userInst.getInstanceNameAttr(),
        operands, userInst.getParameters(), userInst.getInnerSymAttr());

    for (unsigned i = 0; i < userInst.getNumResults(); ++i)
      userInst.getResult(i).replaceAllUsesWith(newInst.getResult(i));

    if (userInst.getDoNotPrint())
      newInst.setDoNotPrintAttr(UnitAttr::get(newInst.getContext()));

    if (userInst == instOp)
      boredValue = newInst.getResult(resIdx);

    userInst.erase();
  }

  return boredValue;
}

Value ResolveXMRRefPass::getInstancePortValue(hw::InstanceOp inst,
                                              unsigned portIdx,
                                              mlir::Operation *moduleOp) {
  unsigned outIdx = 0;
  unsigned inOrInoutIdx = 0;

  for (unsigned i = 0; i < portIdx; ++i) {
    hw::PortInfo port;
    if (auto mod = dyn_cast<hw::HWModuleOp>(moduleOp))
      port = mod.getPort(i);
    else
      port = cast<hw::HWModuleExternOp>(moduleOp).getPort(i);

    if (port.isOutput())
      outIdx++;
    else
      inOrInoutIdx++;
  }

  hw::PortInfo targetPort;
  if (auto mod = dyn_cast<hw::HWModuleOp>(moduleOp))
    targetPort = mod.getPort(portIdx);
  else
    targetPort = cast<hw::HWModuleExternOp>(moduleOp).getPort(portIdx);

  if (targetPort.isOutput()) {
    if (outIdx < inst.getNumResults())
      return inst.getResult(outIdx);
  } else {
    if (inOrInoutIdx < inst.getNumOperands())
      return inst.getOperand(inOrInoutIdx);
  }

  return nullptr;
}
