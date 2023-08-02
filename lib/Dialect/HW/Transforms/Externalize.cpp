//===- Externalize.cpp - Externalize operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include "llvm/Support/Debug.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_EXTERNALIZE
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct ExternalizePass : public impl::ExternalizeBase<ExternalizePass> {
  using ExternalizeBase<ExternalizePass>::ExternalizeBase;
  void runOnOperation() override;
};

// A mapping between interface-published port names which are required to be
// present in the provided external module, and the actual names that the
// external module uses for said ports.
using PortNameMap = llvm::DenseMap<mlir::StringAttr, mlir::StringAttr>;

static LogicalResult parsePortNameMap(Location loc,
                                      Pass::ListOption<std::string> &args,
                                      PortNameMap &nameMap) {
  std::string portId, portName;
  auto *ctx = loc.getContext();
  for (auto arg : args) {
    auto argRef = llvm::StringRef(arg);
    auto [id, name] = argRef.split('=');
    if (name.empty())
      return emitError(loc)
             << "Error parsing port name mapping - '=' not found";

    if (!nameMap
             .try_emplace(StringAttr::get(ctx, id), StringAttr::get(ctx, name))
             .second)
      return emitError(loc) << "Duplicate entries of '" << id
                            << "' in the provided port mapping";
  }

  return success();
}
} // anonymous namespace

void ExternalizePass::runOnOperation() {
  mlir::ModuleOp mod = getOperation();
  auto *ctx = mod.getContext();
  SymbolTable &symtbl = getAnalysis<SymbolTable>();

  // Collect all target ops.
  llvm::SmallVector<hw::ExternalizeableOpInterface> opsToReplace;
  auto opNameAttr = StringAttr::get(ctx, opName);
  auto res =
      mod.walk([&](Operation *op) {
        if (op->getName().getIdentifier() == opNameAttr) {
          auto extIf = dyn_cast<hw::ExternalizeableOpInterface>(op);
          if (!extIf) {
            op->emitOpError()
                << "Trying to run hw-externalizeable on operation '" << opName
                << "' which does not implement the ExternalizeableOpInterface";
            return WalkResult::interrupt();
          }
          opsToReplace.push_back(extIf);
        }
        return WalkResult::advance();
      });

  if (opsToReplace.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  if (res.wasInterrupted())
    return signalPassFailure();

  // Parse and validate the user-provided port name list based on what's
  // reported by the interface.
  auto someOpHandle = opsToReplace.front();
  llvm::SmallVector<hw::PortInfo> requiredPorts =
      someOpHandle.getRequiredPorts();
  PortNameMap providedPortNames;
  if (failed(parsePortNameMap(mod.getLoc(), portNames, providedPortNames)))
    return signalPassFailure();

  for (auto &requiredPort : requiredPorts) {
    auto it = providedPortNames.find(requiredPort.name);
    if (it == providedPortNames.end()) {
      mod.emitError() << "No port name mapping provided for required port "
                      << requiredPort.name << " of the '" << opName
                      << "' ExternalizeableOpInterface";
      return signalPassFailure();
    }

    // Replace the port name with the user-provided mapping
    requiredPort.name = it->second;
  }

  llvm::SmallVector<hw::PortInfo> optionalPorts =
      someOpHandle.getOptionalPorts();
  for (auto &optionalPort : optionalPorts) {
    auto it = providedPortNames.find(optionalPort.name);
    if (it == providedPortNames.end())
      continue; // not provided.

    // Replace the port name with the user-provided mapping and move it to
    // the set of required ports.
    optionalPort.name = it->second;
    requiredPorts.push_back(optionalPort);
  }

  // All port names have been parsed.
  auto builder = OpBuilder::atBlockBegin(getOperation().getBody());
  auto moduleOp = builder.create<HWModuleExternOp>(
      getOperation().getLoc(), builder.getStringAttr(moduleName), requiredPorts,
      moduleName);
  symtbl.insert(moduleOp);

  // Fetch the port mapping of the module to allow the interface implementations
  // a method of easily looking up the user-named ports.
  hw::ModulePortLookupInfo modPortLookup = moduleOp.getPortLookupInfo();

  // Replace all matched operations with an instance of the external module.
  SmallVector<Value, 4> instPorts;
  for (auto replOp : opsToReplace) {
    ImplicitLocOpBuilder builder(replOp.getLoc(), replOp);
    llvm::SmallVector<Value> instOperandMap, instResultMap;
    replOp.doOperandAndResultMapping(builder, modPortLookup, providedPortNames,
                                     instOperandMap, instResultMap);

    auto instOp = builder.create<InstanceOp>(
        moduleOp, builder.getStringAttr(instName), instOperandMap);
    for (auto [opRes, instRes] : llvm::zip(instResultMap, instOp.getResults()))
      opRes.replaceAllUsesWith(instRes);
    replOp.erase();
    ++numOperationsCnverted;
  }
}

std::unique_ptr<Pass>
circt::hw::createExternalizePass(const circt::hw::ExternalizeOptions &options) {
  return std::make_unique<ExternalizePass>(options);
}
