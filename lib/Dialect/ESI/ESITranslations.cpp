//===- ESITranslations.cpp - ESI translations -------------------*- C++ -*-===//
//
// ESI translations:
// - Cap'nProto schema generation
// - C++ API generation
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/IndentingOStream.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Format.h"

#include <algorithm>

#ifdef CAPNP
#include "capnp/ESICapnp.h"
#include "circt/Dialect/ESI/CosimSchema.h"
#endif

using namespace circt;
using namespace circt::esi;

//===----------------------------------------------------------------------===//
// ESI Cosim Cap'nProto schema generation.
//
// Cosimulation in ESI is done over capnp. This translation walks the IR, finds
// all the `esi.cosim` ops, and creates a schema for all the types. It requires
// CAPNP to be enabled.
//===----------------------------------------------------------------------===//

#ifdef CAPNP

namespace {
struct ExportCosimSchema {
  ExportCosimSchema(ModuleOp module, llvm::raw_ostream &os)
      : module(module), os(os), diag(module.getContext()->getDiagEngine()),
        unknown(UnknownLoc::get(module.getContext())) {
    diag.registerHandler([this](Diagnostic &diag) -> LogicalResult {
      if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
        ++errorCount;
      return failure();
    });
  }

  /// Emit an ID in capnp format.
  llvm::raw_ostream &emitId(uint64_t id) {
    return os << "@" << llvm::format_hex(id, /*width=*/16 + 2);
  }

  /// Emit the whole schema.
  LogicalResult emit();

  /// Collect the types for which we need to emit a schema. Output some metadata
  /// comments.
  LogicalResult visitEndpoint(CosimEndpointOp);

private:
  ModuleOp module;
  llvm::raw_ostream &os;
  mlir::DiagnosticEngine &diag;
  const Location unknown;
  size_t errorCount = 0;

  // All the `esi.cosim` input and output types encountered during the IR walk.
  // This is NOT in a deterministic order!
  llvm::SmallVector<std::shared_ptr<capnp::CapnpTypeSchema>> types;
};
} // anonymous namespace

LogicalResult ExportCosimSchema::visitEndpoint(CosimEndpointOp ep) {
  auto sendTypeSchema =
      std::make_shared<capnp::CapnpTypeSchema>(ep.getSend().getType());
  if (!sendTypeSchema->isSupported())
    return ep.emitOpError("Type ")
           << ep.getSend().getType() << " not supported.";
  types.push_back(sendTypeSchema);

  auto recvTypeSchema =
      std::make_shared<capnp::CapnpTypeSchema>(ep.getRecv().getType());
  if (!recvTypeSchema->isSupported())
    return ep.emitOpError("Type '")
           << ep.getRecv().getType() << "' not supported.";
  types.push_back(recvTypeSchema);

  os << "# Endpoint ";
  StringAttr epName = ep->getAttrOfType<StringAttr>("name");
  if (epName)
    os << epName << " endpoint at " << ep.getLoc() << ":\n";
  os << "#   Send type: ";
  sendTypeSchema->writeMetadata(os);
  os << "\n";

  os << "#   Recv type: ";
  recvTypeSchema->writeMetadata(os);
  os << "\n";

  return success();
}

static void emitCosimSchemaBody(llvm::raw_ostream &os) {
  StringRef entireSchemaFile = circt::esi::cosim::CosimSchema;
  size_t idLocation = entireSchemaFile.find("@0x");
  size_t newlineAfter = entireSchemaFile.find('\n', idLocation);

  os << "\n\n"
     << "#########################################################\n"
     << "## Standard RPC interfaces.\n"
     << "#########################################################\n";
  os << entireSchemaFile.substr(newlineAfter) << "\n";
}

LogicalResult ExportCosimSchema::emit() {
  os << "#########################################################\n"
     << "## ESI generated schema.\n"
     << "#########################################################\n";

  // Walk and collect the type data.
  auto walkResult = module.walk([this](CosimEndpointOp ep) {
    if (failed(visitEndpoint(ep)))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  os << "#########################################################\n";

  // We need a sorted list to ensure determinism.
  llvm::sort(types.begin(), types.end(), [](auto &a, auto &b) {
    return a->capnpTypeID() > b->capnpTypeID();
  });

  // Compute and emit the capnp file id.
  uint64_t fileHash = 2544816649379317016; // Some random number.
  for (auto &schema : types)
    fileHash =
        llvm::hashing::detail::hash_16_bytes(fileHash, schema->capnpTypeID());
  // Capnp IDs always have a '1' high bit.
  fileHash |= 0x8000000000000000;
  emitId(fileHash) << ";\n\n";

  os << "#########################################################\n"
     << "## Types for your design.\n"
     << "#########################################################\n\n";
  // Iterate through the various types and emit their schemas.
  auto end = std::unique(
      types.begin(), types.end(),
      [&](const auto &lhs, const auto &rhs) { return *lhs == *rhs; });
  for (auto typeIter = types.begin(); typeIter != end; ++typeIter) {
    if (failed((*typeIter)->write(os)))
      // If we fail during an emission, dump out early since the output may be
      // corrupted.
      return failure();
  }

  // Include the RPC schema in each generated file.
  emitCosimSchemaBody(os);

  return errorCount == 0 ? success() : failure();
}

LogicalResult circt::esi::exportCosimSchema(ModuleOp module,
                                            llvm::raw_ostream &os) {
  ExportCosimSchema schema(module, os);
  return schema.emit();
}

#else // Not CAPNP

LogicalResult circt::esi::exportCosimSchema(ModuleOp module,
                                            llvm::raw_ostream &os) {
  return mlir::emitError(UnknownLoc::get(module.getContext()),
                         "Not compiled with CAPNP support");
}

#endif

//===----------------------------------------------------------------------===//
// ESI C++ Cap'nProto cosim API generation.
//===----------------------------------------------------------------------===//

#ifdef CAPNP

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
  support::indenting_ostream ios;
  mlir::DiagnosticEngine &diag;
  const Location unknown;
  size_t errorCount = 0;

  // All the `esi.cosim` input and output types encountered during the IR walk.
  // This is NOT in a deterministic order!
  llvm::MapVector<mlir::Type, capnp::CPPType> types;
  llvm::SmallVector<capnp::CPPService> cppServices;
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
      capnp::CPPType dirTypeSchema(dirType);
      if (!dirTypeSchema.isSupported())
        return ep.emitOpError("Type ") << dirType << " not supported.";
      dirTypeSchemaIt = types.insert({dirType, dirTypeSchema}).first;
    }

    ios << "//   " << (isSend ? "Send" : "Recv") << " type: ";
    dirTypeSchemaIt->second.writeMetadata(ios.getStream());
    ios << "\n";
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
      capnp::CPPType dirTypeSchema(dirType);
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
    auto cppService = capnp::CPPService(serviceDeclOp, types);
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
  ios.addIndent();

  // Iterate through the various types and emit their CPP APIs.
  for (auto cppType : types) {
    if (failed(cppType.second.write(ios)))
      // If we fail during an emission, dump out early since the output may
      // be corrupted.
      return failure();
  }

  ios.reduceIndent();
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

  SmallVector<capnp::CPPDesignModule> designMods;
  for (auto &mod : modsWithLocalServices)
    designMods.push_back(
        capnp::CPPDesignModule(mod.first, mod.second, cppServices));

  // Write modules
  for (auto &designMod : designMods) {
    if (failed(designMod.write(ios)))
      return failure();
    ios << "\n";
  }
  return success();
}

LogicalResult circt::esi::exportCosimCPPAPI(ModuleOp module,
                                            llvm::raw_ostream &os) {
  CosimCPPAPI api(module, os);
  return api.emit();
}

#else

LogicalResult circt::esi::exportCosimCPPAPI(ModuleOp module,
                                            llvm::raw_ostream &os) {
  return mlir::emitError(UnknownLoc::get(module.getContext()),
                         "Not compiled with CAPNP support");
}

#endif

//===----------------------------------------------------------------------===//
// Register all ESI translations.
//===----------------------------------------------------------------------===//

void circt::esi::registerESITranslations() {
#ifdef CAPNP
  mlir::TranslateFromMLIRRegistration cosimToCapnp(
      "export-esi-capnp", "ESI Cosim Cap'nProto schema generation",
      exportCosimSchema, [](mlir::DialectRegistry &registry) {
        registry.insert<ESIDialect, circt::hw::HWDialect, circt::sv::SVDialect,
                        mlir::func::FuncDialect, mlir::BuiltinDialect>();
      });
  mlir::TranslateFromMLIRRegistration cosimToCPP(
      "export-esi-capnp-cpp", "ESI Cosim Cap'nProto cpp API generation",
      exportCosimCPPAPI, [](mlir::DialectRegistry &registry) {
        registry.insert<ESIDialect, circt::hw::HWDialect, circt::sv::SVDialect,
                        mlir::func::FuncDialect, mlir::BuiltinDialect>();
      });
#endif
}
