//===- ESITranslations.cpp - ESI translations -------------------*- C++ -*-===//
//
// ESI translations:
// - Cap'nProto schema generation
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"

#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/SV/Dialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>

using namespace mlir;
using namespace circt::esi;

//===----------------------------------------------------------------------===//
// ESI Cosim Cap'nProto schema generation.
//
// Cosimulation in ESI is done over capnp. This translation walks the IR, finds
// all the `esi.cosim` ops, and creates a schema for all the types.
//===----------------------------------------------------------------------===//

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
  llvm::raw_ostream &emitId(llvm::hash_code id) {
    return os << "@" << llvm::format_hex(id, /*width=*/16 + 2);
  }

  /// Emit the whole schema.
  LogicalResult emit();

  /// Collect the types for which we need to emit a schema. Output some metadata
  /// comments.
  LogicalResult visitEndpoint(CosimEndpoint);

  /// Do we support emitting a schema for 'type'?
  static bool isTypeSupported(Type type);

  /// Emit a schema for a single int.
  LogicalResult emitSchemaFor(IntegerType type);

  /// Emit a struct name.
  llvm::raw_ostream &emitName(Type type) { return os << "ESI_" << type; }

private:
  /// Intentation utils.
  raw_ostream &indent() { return os.indent(currentIndent); }
  void addIndent() { currentIndent += 2; }
  void reduceIndent() { currentIndent -= 2; }
  size_t currentIndent = 0;

  ModuleOp module;
  llvm::raw_ostream &os;
  DiagnosticEngine &diag;
  const Location unknown;
  size_t errorCount = 0;

  // All the `esi.cosim` input and output types encountered during the IR walk.
  // This is NOT in a deterministic order!
  llvm::DenseSet<Type> types;
};
} // anonymous namespace

bool ExportCosimSchema::isTypeSupported(Type type) {
  auto chanPort = type.dyn_cast<ChannelPort>();
  if (chanPort) {
    uint64_t innerHash = isTypeSupported(chanPort.getInner());
    return llvm::hash_combine(chanPort.getTypeID(), innerHash);
  }

  auto i = type.dyn_cast<IntegerType>();
  if (i)
    return i.getWidth() <= 64;
  return false;
}

LogicalResult ExportCosimSchema::emitSchemaFor(IntegerType type) {
  SmallString<16> typeStr;
  if (type.isSigned())
    typeStr = "Int";
  else
    typeStr = "UInt";

  // Round up.
  auto w = type.getWidth();
  if (w == 1)
    typeStr = "Bool";
  else if (w <= 8)
    typeStr += "8";
  else if (w <= 16)
    typeStr += "16";
  else if (w <= 32)
    typeStr += "32";
  else if (w <= 64)
    typeStr += "64";
  else
    return diag.emit(unknown, DiagnosticSeverity::Error)
           << "Capnp does not support ints larger than 64-bit";

  // Since capnp requires messages to be structs, emit a wrapper struct.
  indent() << "struct ";
  emitName(type) << " ";
  emitId(getCapnpTypeID(type)) << " {\n";
  addIndent();

  // Specify the actual type, followed by the capnp field.
  indent() << "// Actual type is " << type << ".\n";
  indent() << "i @0 :" << typeStr << ";\n";

  reduceIndent();
  indent() << "}\n\n";
  return success();
}

LogicalResult ExportCosimSchema::visitEndpoint(CosimEndpoint ep) {
  ChannelPort inputPort = ep.input().getType().dyn_cast<ChannelPort>();
  if (!inputPort)
    return ep.emitOpError("Expected ChannelPort type for input. Got ")
           << inputPort;
  if (!isTypeSupported(inputPort))
    return ep.emitOpError("Type '") << inputPort << "' not supported.";
  types.insert(inputPort);

  ChannelPort outputPort = ep.output().getType().dyn_cast<ChannelPort>();
  if (!outputPort)
    ep.emitOpError("Expected ChannelPort type for output. Got ") << outputPort;
  if (!isTypeSupported(outputPort))
    return ep.emitOpError("Type '") << outputPort << "' not supported.";
  types.insert(outputPort);

  os << "# Endpoint #" << ep.endpointID() << " at " << ep.getLoc() << ":\n";
  os << "#   Input type: ";
  emitName(inputPort.getInner()) << " ";
  emitId(ep.getInputTypeID()) << "\n";
  os << "#   Output type: ";
  emitName(outputPort.getInner()) << " ";
  emitId(ep.getOutputTypeID()) << "\n";

  return success();
}

LogicalResult ExportCosimSchema::emit() {
  os << "#########################################################\n"
     << "## ESI generated schema. For use with CosimDpi.capnp\n"
     << "#######\n";

  // Walk and collect the type data.
  module.walk([this](CosimEndpoint ep) { visitEndpoint(ep); });
  os << "#######\n";

  // We need a sorted list to ensure determinism.
  SmallVector<Type> sortedTypes(types.begin(), types.end());
  std::sort(sortedTypes.begin(), sortedTypes.end(), [](Type a, Type b) {
    return getCapnpTypeID(a) > getCapnpTypeID(b);
  });

  // Compute and emit the capnp file id.
  llvm::hash_code fileHash =
      llvm::hash_combine_range(sortedTypes.begin(), sortedTypes.end());
  emitId(fileHash) << ";\n\n";

  // Iterate through the various types and emit their schemas.
  for (auto type : sortedTypes) {
    auto chanPort = type.dyn_cast<ChannelPort>();
    if (!chanPort) {
      diag.emit(unknown, DiagnosticSeverity::Error)
          << "Only ChannelPort types are supported, not " << type;
      continue;
    }

    LogicalResult rc =
        TypeSwitch<Type, LogicalResult>(chanPort.getInner())
            .Case([this](IntegerType it) { return emitSchemaFor(it); })
            .Default([this](Type type) {
              return diag.emit(unknown, DiagnosticSeverity::Error)
                     << "Type '" << type << "' not supported.";
            });
    if (failed(rc))
      // If we fail during an emission, dump out early since the output may be
      // corrupted.
      return failure();
  }

  // diag.emit(module.getLoc(), DiagnosticSeverity::Error)
  // << "Incomplete translation";
  return errorCount == 0 ? success() : failure();
}

static LogicalResult exportCosimSchema(ModuleOp module, llvm::raw_ostream &os) {
  ExportCosimSchema schema(module, os);
  return schema.emit();
}

//===----------------------------------------------------------------------===//
// Register all ESI translations.
//===----------------------------------------------------------------------===//

void circt::esi::registerESITranslations() {
  TranslateFromMLIRRegistration cosimToCapnp(
      "emit-esi-capnp", exportCosimSchema, [](DialectRegistry &registry) {
        registry
            .insert<ESIDialect, circt::rtl::RTLDialect, circt::sv::SVDialect,
                    mlir::StandardOpsDialect, mlir::BuiltinDialect>();
      });
}
