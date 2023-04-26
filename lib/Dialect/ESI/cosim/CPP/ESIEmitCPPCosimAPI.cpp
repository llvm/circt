//===- ESIEmitCPPCosimAPI.cpp - ESI C++ cosim API emission ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit the C++ ESI Capnp cosim API.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

#include "CPPCosimAPI.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace circt;
using namespace circt::esi;
using namespace cppcosimapi;

namespace {
struct CosimCPPAPI {
  CosimCPPAPI(ModuleOp module, llvm::raw_ostream &os)
      : module(module), ios(os), diag(module.getContext()->getDiagEngine()),
        unknown(UnknownLoc::get(module.getContext())) {
    diag.registerHandler([this](Diagnostic &diag) -> LogicalResult {
      if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
        ++errorCount;
      return failure();
    });
  }

  /// Emit the whole API.
  LogicalResult emit();

  /// Collect the types for which we need to emit the API. Output some metadata
  /// comments.
  LogicalResult visitEndpoint(CosimEndpointOp);

private:
  LogicalResult gatherTypes(Location loc);
  LogicalResult emitTypes();
  LogicalResult emitServiceDeclarations();
  LogicalResult emitDesignModules();
  LogicalResult emitGlobalNamespace();

  ModuleOp module;
  mlir::raw_indented_ostream ios;
  mlir::DiagnosticEngine &diag;
  const Location unknown;
  size_t errorCount = 0;

  // All the `esi.cosim` input and output types encountered during the IR walk.
  // This is NOT in a deterministic order!
  llvm::MapVector<mlir::Type, CPPType> types;
  llvm::SmallVector<CPPService> cppServices;
};
} // anonymous namespace

LogicalResult CosimCPPAPI::visitEndpoint(CosimEndpointOp ep) {
  ios << "// Endpoint ";
  StringAttr epName = ep->getAttrOfType<StringAttr>("name");
  if (epName)
    ios << epName << " endpoint at " << ep.getLoc() << ":\n";

  auto emitDirection = [&](bool isSend, mlir::Type type) -> LogicalResult {
    auto dirType = esi::innerType(type);
    auto dirTypeSchemaIt = types.find(dirType);
    if (dirTypeSchemaIt == types.end()) {
      CPPType dirTypeSchema(dirType);
      if (!dirTypeSchema.isSupported())
        return ep.emitOpError("Type ") << dirType << " not supported.";
      dirTypeSchemaIt = types.insert({dirType, dirTypeSchema}).first;
    }

    ios << "//   " << (isSend ? "Send" : "Recv")  << "\n";
    return success();
  };

  if (failed(emitDirection(true, ep.getSend().getType())) ||
      failed(emitDirection(false, ep.getRecv().getType())))
    return failure();
  return success();
}

LogicalResult CosimCPPAPI::gatherTypes(Location loc) {
  auto storeType = [&](mlir::Type type) -> LogicalResult {
    auto dirType = esi::innerType(type);
    auto dirTypeSchemaIt = types.find(dirType);
    if (dirTypeSchemaIt == types.end()) {
      CPPType dirTypeSchema(dirType);
      if (!dirTypeSchema.isSupported())
        return emitError(loc) << "Type " << dirType << " not supported.";
      dirTypeSchemaIt = types.insert({dirType, dirTypeSchema}).first;
    }
    return success();
  };

  for (auto serviceDeclOp : module.getOps<ServiceDeclOpInterface>()) {
    llvm::SmallVector<ServicePortInfo> ports;
    serviceDeclOp.getPortList(ports);
    for (auto portInfo : ports) {
      if (portInfo.toClientType)
        if (failed(storeType(portInfo.toClientType)))
          return failure();
      if (portInfo.toServerType)
        if (failed(storeType(portInfo.toServerType)))
          return failure();
    }
  }
  return success();
}

LogicalResult CosimCPPAPI::emit() {
  // Walk and collect the type data.
  if (failed(gatherTypes(module.getLoc())))
    return failure();

  ios << "#pragma once\n\n";

  ios << "// The ESI C++ API relies on the refl-cpp library for type "
         "introspection. "
         "This must be provided by the user.\n";
  ios << "// See https://github.com/veselink1/refl-cpp \n";
  ios << "#include \"refl.hpp\"\n\n";

  ios << "#include <cstdint>\n";
  ios << "#include \"esi/backends/cosim/capnp.h\"\n";
  ios << "\n// Include the generated Cap'nProto schema header. This must "
         "defined "
         "by the build system.\n";
  ios << "#include ESI_COSIM_CAPNP_H\n";
  ios << "\n\n";

  ios << "namespace esi {\n";
  ios << "namespace runtime {\n\n";

  if (failed(emitTypes()) || failed(emitServiceDeclarations()) ||
      failed(emitDesignModules()))
    return failure();

  ios << "} // namespace runtime\n";
  ios << "} // namespace esi\n\n";

  ios << "// ESI dynamic reflection support\n";
  if (failed(emitGlobalNamespace()))
    return failure();

  return success();
}

LogicalResult CosimCPPAPI::emitServiceDeclarations() {
  // Locate all of the service declarations which are needed by the
  // cosim-implemented services in the service hierarchy.
  for (auto serviceDeclOp : module.getOps<ServiceDeclOpInterface>()) {
    auto cppService = CPPService(serviceDeclOp, types);
    if (failed(cppService.write(ios)))
      return failure();
    cppServices.push_back(cppService);
  }

  return success();
}

LogicalResult CosimCPPAPI::emitGlobalNamespace() {
  // Emit ESI type reflection classes.
  llvm::SmallVector<std::string> namespaces = {"esi", "runtime", "ESITypes"};
  for (auto &cppType : types)
    cppType.second.writeReflection(ios, namespaces);

  return success();
}

LogicalResult CosimCPPAPI::emitTypes() {
  ios << "class ESITypes {\n";
  ios << "public:\n";
  ios.indent();

  // Iterate through the various types and emit their CPP APIs.
  for (auto cppType : types) {
    if (failed(cppType.second.write(ios)))
      // If we fail during an emission, dump out early since the output may
      // be corrupted.
      return failure();
  }

  ios.unindent();
  ios << "};\n\n";

  return success();
}

LogicalResult CosimCPPAPI::emitDesignModules() {
  // Get a list of metadata ops which originated in modules (path is empty).
  SmallVector<
      std::pair<hw::HWModuleLike, SmallVector<ServiceHierarchyMetadataOp, 0>>>
      modsWithLocalServices;
  for (auto hwmod : module.getOps<hw::HWModuleLike>()) {
    SmallVector<ServiceHierarchyMetadataOp, 0> metadataOps;
    hwmod.walk([&metadataOps](ServiceHierarchyMetadataOp md) {
      if (md.getServerNamePath().empty() && md.getImplType() == "cosim")
        metadataOps.push_back(md);
    });
    if (!metadataOps.empty())
      modsWithLocalServices.push_back(std::make_pair(hwmod, metadataOps));
  }

  SmallVector<CPPDesignModule> designMods;
  for (auto &mod : modsWithLocalServices)
    designMods.push_back(
        CPPDesignModule(mod.first, mod.second, cppServices));

  // Write modules
  for (auto &designMod : designMods) {
    if (failed(designMod.write(ios)))
      return failure();
    ios << "\n";
  }
  return success();
}

LogicalResult circt::esi::cppcosimapi::exportCosimCPPAPI(ModuleOp module,
                                            llvm::raw_ostream &os) {
  CosimCPPAPI api(module, os);
  return api.emit();
}
