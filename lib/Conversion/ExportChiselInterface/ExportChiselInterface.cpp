//===- ExportChiselInterface.cpp - Chisel Interface Emitter ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the Chisel interface emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportChiselInterface.h"
#include "../PassDetail.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/Version.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace circt;
using namespace firrtl;

#define DEBUG_TYPE "export-chisel-package"

//===----------------------------------------------------------------------===//
// Interface emission logic
//===----------------------------------------------------------------------===//

static const unsigned int indentIncrement = 2;

/// Emits type construction expression for the port type, recursing into
/// aggregate types as needed.
static LogicalResult emitPortType(Location location, FIRRTLBaseType type,
                                  Direction direction, llvm::raw_ostream &os,
                                  unsigned int indent,
                                  bool hasEmittedDirection = false) {
  auto emitTypeWithArguments =
      [&](StringRef name,
          // A lambda of type (bool hasEmittedDirection) -> LogicalResult.
          auto emitArguments,
          // Indicates whether parentheses around type arguments should be used.
          bool emitParentheses = true) -> LogicalResult {
    // Include the direction if the type is not composed of flips and analog
    // signals and we haven't already emitted the direction before recursing to
    // this field.
    bool emitDirection =
        type.isPassive() && !type.containsAnalog() && !hasEmittedDirection;
    if (emitDirection) {
      switch (direction) {
      case Direction::In:
        os << "Input(";
        break;
      case Direction::Out:
        os << "Output(";
        break;
      }
    }

    os << name;

    if (emitParentheses)
      os << "(";

    if (failed(emitArguments(hasEmittedDirection || emitDirection)))
      return failure();

    if (emitParentheses)
      os << ')';

    if (emitDirection)
      os << ')';

    return success();
  };

  // Emits a type that does not require arguments.
  auto emitType = [&](StringRef name) -> LogicalResult {
    return emitTypeWithArguments(name, [](bool) { return success(); });
  };

  // Emits a type that requires a known width argument.
  auto emitWidthQualifiedType = [&](auto type,
                                    StringRef name) -> LogicalResult {
    auto width = type.getWidth();
    if (!width.has_value()) {
      return LogicalResult(emitError(
          location, "Expected width to be inferred for exported port"));
    }
    return emitTypeWithArguments(name, [&](bool) {
      os << *width << ".W";
      return success();
    });
  };

  return TypeSwitch<FIRRTLBaseType, LogicalResult>(type)
      .Case<ClockType>([&](ClockType) { return emitType("Clock"); })
      .Case<AsyncResetType>(
          [&](AsyncResetType) { return emitType("AsyncReset"); })
      .Case<ResetType>([&](ResetType) {
        return emitError(
            location, "Expected reset type to be inferred for exported port");
      })
      .Case<UIntType>([&](UIntType uIntType) {
        return emitWidthQualifiedType(uIntType, "UInt");
      })
      .Case<SIntType>([&](SIntType sIntType) {
        return emitWidthQualifiedType(sIntType, "SInt");
      })
      .Case<AnalogType>([&](AnalogType analogType) {
        return emitWidthQualifiedType(analogType, "Analog");
      })
      .Case<BundleType>([&](BundleType bundleType) {
        // Emit an anonymous bundle, emitting a `val` for each field.
        return emitTypeWithArguments(
            "new Bundle ",
            [&](bool hasEmittedDirection) {
              os << "{\n";
              unsigned int nestedIndent = indent + indentIncrement;
              for (const auto &element : bundleType.getElements()) {
                os.indent(nestedIndent)
                    << "val " << element.name.getValue() << " = ";
                auto elementResult = emitPortType(
                    location, element.type,
                    element.isFlip ? direction::flip(direction) : direction, os,
                    nestedIndent, hasEmittedDirection);
                if (failed(elementResult))
                  return failure();
                os << '\n';
              }
              os.indent(indent) << "}";
              return success();
            },
            false);
      })
      .Case<FVectorType>([&](FVectorType vectorType) {
        // Emit a vector type, emitting the type of its element as an argument.
        return emitTypeWithArguments("Vec", [&](bool hasEmittedDirection) {
          os << vectorType.getNumElements() << ", ";
          return emitPortType(location, vectorType.getElementType(), direction,
                              os, indent, hasEmittedDirection);
        });
      })
      .Default([](FIRRTLBaseType) {
        llvm_unreachable("unknown FIRRTL type");
        return failure();
      });
}

/// Emits an `IO` for the `port`.
static LogicalResult emitPort(const PortInfo &port, llvm::raw_ostream &os) {
  os.indent(indentIncrement) << "val " << port.getName() << " = IO(";
  if (failed(emitPortType(port.loc, port.type.cast<FIRRTLBaseType>(),
                          port.direction, os, indentIncrement)))
    return failure();
  os << ")\n";

  return success();
}

/// Emits an `ExtModule` class with port declarations for `module`.
static LogicalResult emitModule(FModuleLike module, llvm::raw_ostream &os) {
  os << "class " << module.moduleName() << " extends ExtModule {\n";

  for (const auto &port : module.getPorts()) {
    if (failed(emitPort(port, os)))
      return failure();
  }

  os << "}\n";

  return success();
}

/// Exports a Chisel interface to the output stream.
static LogicalResult exportChiselInterface(CircuitOp circuit,
                                           llvm::raw_ostream &os) {
  // Emit version, package, and import declarations
  os << circt::getCirctVersionComment() << "package shelf."
     << circuit.getName().lower()
     << "\n\nimport chisel3._\nimport chisel3.experimental._\n\n";

  // Emit a class for the main circuit module.
  auto topModule = circuit.getMainModule();
  if (failed(emitModule(topModule, os)))
    return failure();

  return success();
}

/// Exports Chisel interface files for the circuit to the specified directory.
static LogicalResult exportSplitChiselInterface(CircuitOp circuit,
                                                StringRef outputDirectory) {
  // Create the output directory if needed.
  std::error_code error = llvm::sys::fs::create_directories(outputDirectory);
  if (error) {
    circuit.emitError("cannot create output directory \"")
        << outputDirectory << "\": " << error.message();
    return failure();
  }

  // Open the output file.
  SmallString<128> interfaceFilePath(outputDirectory);
  llvm::sys::path::append(interfaceFilePath, circuit.getName());
  llvm::sys::path::replace_extension(interfaceFilePath, "scala");
  std::string errorMessage;
  auto interfaceFile = mlir::openOutputFile(interfaceFilePath, &errorMessage);
  if (!interfaceFile) {
    circuit.emitError(errorMessage);
    return failure();
  }

  // Export the interface to the file.
  auto result = exportChiselInterface(circuit, interfaceFile->os());
  if (succeeded(result))
    interfaceFile->keep();
  return result;
}

//===----------------------------------------------------------------------===//
// ExportChiselInterfacePass and ExportSplitChiselInterfacePass
//===----------------------------------------------------------------------===//

namespace {
struct ExportChiselInterfacePass
    : public ExportChiselInterfaceBase<ExportChiselInterfacePass> {

  explicit ExportChiselInterfacePass(llvm::raw_ostream &os) : os(os) {}

  void runOnOperation() override {
    if (failed(exportChiselInterface(getOperation(), os)))
      signalPassFailure();
  }

private:
  llvm::raw_ostream &os;
};

struct ExportSplitChiselInterfacePass
    : public ExportSplitChiselInterfaceBase<ExportSplitChiselInterfacePass> {

  explicit ExportSplitChiselInterfacePass(StringRef directory) {
    directoryName = directory.str();
  }

  void runOnOperation() override {
    if (failed(exportSplitChiselInterface(getOperation(), directoryName)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
circt::createExportChiselInterfacePass(llvm::raw_ostream &os) {
  return std::make_unique<ExportChiselInterfacePass>(os);
}

std::unique_ptr<mlir::Pass>
circt::createExportSplitChiselInterfacePass(mlir::StringRef directory) {
  return std::make_unique<ExportSplitChiselInterfacePass>(directory);
}

std::unique_ptr<mlir::Pass> circt::createExportChiselInterfacePass() {
  return createExportChiselInterfacePass(llvm::outs());
}
