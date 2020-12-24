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
  llvm::raw_ostream &emitId(uint64_t id) {
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
  LogicalResult emitSchemaFor(IntegerType type, uint64_t hash);

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
  llvm::SmallVector<ChannelPort> types;
};
} // anonymous namespace

bool ExportCosimSchema::isTypeSupported(Type type) {
  auto chanPort = type.dyn_cast<ChannelPort>();
  if (chanPort) {
    return isTypeSupported(chanPort.getInner());
  }

  auto i = type.dyn_cast<IntegerType>();
  if (i)
    return i.getWidth() <= 64;
  return false;
}

LogicalResult ExportCosimSchema::emitSchemaFor(IntegerType type,
                                               uint64_t hash) {
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
  emitId(hash) << " {\n";
  addIndent();

  // Specify the actual type, followed by the capnp field.
  indent() << "# Actual type is " << type << ".\n";
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
  types.push_back(inputPort);

  ChannelPort outputPort = ep.output().getType().dyn_cast<ChannelPort>();
  if (!outputPort)
    ep.emitOpError("Expected ChannelPort type for output. Got ") << outputPort;
  if (!isTypeSupported(outputPort))
    return ep.emitOpError("Type '") << outputPort << "' not supported.";
  types.push_back(outputPort);

  os << "# Endpoint ";
  StringAttr epName = ep.getAttrOfType<StringAttr>("name");
  if (epName)
    os << epName << " is endpoint ";
  os << "#" << ep.endpointID() << " at " << ep.getLoc() << ":\n";
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
     << "#########################################################\n";

  // Walk and collect the type data.
  module.walk([this](CosimEndpoint ep) { visitEndpoint(ep); });
  os << "#########################################################\n";

  // We need a sorted list to ensure determinism.
  llvm::sort(types.begin(), types.end(), [](ChannelPort a, ChannelPort b) {
    return getCapnpTypeID(a) > getCapnpTypeID(b);
  });

  // Compute and emit the capnp file id.
  uint64_t fileHash = 2544816649379317016; // Some random number.
  for (ChannelPort chanPort : types)
    fileHash = llvm::hashing::detail::hash_16_bytes(fileHash,
                                                    getCapnpTypeID(chanPort));
  emitId(fileHash) << ";\n\n";

  // Iterate through the various types and emit their schemas.
  auto end = std::unique(types.begin(), types.end());
  for (auto typeIter = types.begin(); typeIter < end; ++typeIter) {
    ChannelPort chanPort = *typeIter;
    uint64_t typeHash = getCapnpTypeID(chanPort);
    LogicalResult rc =
        TypeSwitch<Type, LogicalResult>(chanPort.getInner())
            .Case([this, typeHash](IntegerType it) {
              return emitSchemaFor(it, typeHash);
            })
            .Default([](Type type) {
              assert(false && "Unsupported type should have been filtered out "
                              "in visitEndpoint().");
              return failure();
            });
    if (failed(rc))
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
