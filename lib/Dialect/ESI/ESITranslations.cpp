//===- ESITranslations.cpp - ESI translations -------------------*- C++ -*-===//
//
// ESI translations:
// - Cap'nProto schema generation
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"

#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/SV/SVDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Translation.h"

#include <algorithm>

#ifdef CAPNP
#include "capnp/ESICapnp.h"
#endif

using namespace mlir;
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
      if (diag.getSeverity() == DiagnosticSeverity::Error)
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
  LogicalResult visitEndpoint(CosimEndpoint);

private:
  ModuleOp module;
  llvm::raw_ostream &os;
  DiagnosticEngine &diag;
  const Location unknown;
  size_t errorCount = 0;

  // All the `esi.cosim` input and output types encountered during the IR walk.
  // This is NOT in a deterministic order!
  llvm::SmallVector<capnp::TypeSchema> types;
};
} // anonymous namespace

LogicalResult ExportCosimSchema::visitEndpoint(CosimEndpoint ep) {
  capnp::TypeSchema sendTypeSchema(ep.send().getType());
  if (!sendTypeSchema.isSupported())
    return ep.emitOpError("Type ") << ep.send().getType() << " not supported.";
  types.push_back(sendTypeSchema);

  capnp::TypeSchema recvTypeSchema(ep.recv().getType());
  if (!recvTypeSchema.isSupported())
    return ep.emitOpError("Type '")
           << ep.recv().getType() << "' not supported.";
  types.push_back(recvTypeSchema);

  os << "# Endpoint ";
  StringAttr epName = ep->getAttrOfType<StringAttr>("name");
  if (epName)
    os << epName << " is endpoint ";
  os << "#" << ep.endpointID() << " at " << ep.getLoc() << ":\n";
  os << "#   Send type: ";
  sendTypeSchema.writeMetadata(os);
  os << "\n";

  os << "#   Recv type: ";
  recvTypeSchema.writeMetadata(os);
  os << "\n";

  return success();
}

LogicalResult ExportCosimSchema::emit() {
  os << "#########################################################\n"
     << "## ESI generated schema. For use with CosimDpi.capnp\n"
     << "#########################################################\n";

  // Walk and collect the type data.
  module.walk([this](CosimEndpoint ep) { visitEndpoint(ep); });
  os << "#########################################################\n";

  // We need a sorted list to ensure determinism.
  llvm::sort(types.begin(), types.end(),
             [](capnp::TypeSchema &a, capnp::TypeSchema &b) {
               return a.capnpTypeID() > b.capnpTypeID();
             });

  // Compute and emit the capnp file id.
  uint64_t fileHash = 2544816649379317016; // Some random number.
  for (capnp::TypeSchema &schema : types)
    fileHash =
        llvm::hashing::detail::hash_16_bytes(fileHash, schema.capnpTypeID());
  // Capnp IDs always have a '1' high bit.
  fileHash |= 0x8000000000000000;
  emitId(fileHash) << ";\n\n";

  // Iterate through the various types and emit their schemas.
  auto end = std::unique(types.begin(), types.end());
  for (auto typeIter = types.begin(); typeIter < end; ++typeIter) {
    if (failed(typeIter->write(os)))
      // If we fail during an emission, dump out early since the output may be
      // corrupted.
      return failure();
  }

  return errorCount == 0 ? success() : failure();
}

static LogicalResult exportCosimSchema(ModuleOp module, llvm::raw_ostream &os) {
  ExportCosimSchema schema(module, os);
  return schema.emit();
}

#endif

//===----------------------------------------------------------------------===//
// Register all ESI translations.
//===----------------------------------------------------------------------===//

void circt::esi::registerESITranslations() {
#ifdef CAPNP
  TranslateFromMLIRRegistration cosimToCapnp(
      "emit-esi-capnp", exportCosimSchema, [](DialectRegistry &registry) {
        registry
            .insert<ESIDialect, circt::rtl::RTLDialect, circt::sv::SVDialect,
                    mlir::StandardOpsDialect, mlir::BuiltinDialect>();
      });
#endif
}
