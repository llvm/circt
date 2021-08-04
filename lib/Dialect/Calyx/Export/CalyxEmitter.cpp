//===- CalyxEmitter.cpp - Calyx dialect to .futil emitter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements an emitter for the native Calyx language, which uses
// .futil as an alias.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxEmitter.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Translation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace calyx;
using namespace hw;

//===----------------------------------------------------------------------===//
// Emitter
//===----------------------------------------------------------------------===//

namespace {

/// An emitter for Calyx dialect operations to .futil output.
struct Emitter {
  Emitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult finalize();

  // Indentation
  raw_ostream &indent() { return os.indent(currentIndent); }
  void addIndent() { currentIndent += 2; }
  void reduceIndent() {
    assert(currentIndent >= 2 && "Unintended indentation wrap");
    currentIndent -= 2;
  }

  // Program emission
  void emitProgram(ProgramOp op);

  // Component emission
  void emitComponent(ComponentOp op);
  void emitComponentPorts(ArrayRef<ComponentPortInfo> ports);

  // Cell emission
  void emitCell(CellOp op);

  // Wires emission
  void emitWires(WiresOp op);

  // Group emission
  void emitGroup(GroupOp op);

  // Control emission
  void emitControl(ControlOp op);

  // Assignment emission
  void emitAssignment(AssignOp op);

private:
  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitError(message);
  }

  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitOpError(message);
  }

  /// Emits a Calyx section.
  template <typename Func>
  void emitCalyxSection(StringRef sectionName, Func emitBody,
                        StringRef symbolName = "") {
    indent() << sectionName;
    if (!symbolName.empty())
      os << " " << symbolName;
    os << " {\n";
    addIndent();

    emitBody();
    reduceIndent();
    indent() << "}\n";
  }

  /// The stream we are emitting into.
  llvm::raw_ostream &os;

  /// Whether we have encountered any errors during emission.
  bool encounteredError = false;

  /// Current level of indentation. See `indent()` and
  /// `addIndent()`/`reduceIndent()`.
  unsigned currentIndent = 0;
};

} // end anonymous namespace

LogicalResult Emitter::finalize() { return failure(encounteredError); }

/// Emit an entire program.
void Emitter::emitProgram(ProgramOp op) {
  for (auto &bodyOp : *op.getBody()) {
    if (auto componentOp = dyn_cast<ComponentOp>(bodyOp))
      emitComponent(componentOp);
    else
      emitOpError(&bodyOp, "Unexpected op");
  }
}

/// Emit a component.
void Emitter::emitComponent(ComponentOp op) {
  indent() << "component " << op.getName();

  // Emit the ports.
  auto ports = getComponentPortInfo(op);
  emitComponentPorts(ports);
  os << " {\n";
  addIndent();
  WiresOp wires;
  ControlOp control;

  // Emit cells.
  emitCalyxSection("cells", [&]() {
    for (auto &&bodyOp : *op.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<WiresOp>([&](auto op) { wires = op; })
          .Case<ControlOp>([&](auto op) { control = op; })
          .Case<CellOp>([&](auto op) { emitCell(op); })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside component");
          });
    }
  });

  emitWires(wires);
  emitControl(control);
  reduceIndent();
}

/// Emit the ports of a component.
void Emitter::emitComponentPorts(ArrayRef<ComponentPortInfo> ports) {
  std::vector<ComponentPortInfo> inPorts, outPorts;
  for (auto &&port : ports) {
    if (port.direction == calyx::PortDirection::INPUT)
      inPorts.push_back(port);
    else
      outPorts.push_back(port);
  }

  auto emitPorts = [&](auto ports) {
    os << "(";
    for (size_t i = 0, e = ports.size(); i < e; ++i) {
      const auto &port = ports[i];
      auto name = port.name.getValue();
      // We only care about the bit width in the emitted .futil file.
      auto bitWidth = port.type.getIntOrFloatBitWidth();
      os << name << ": " << bitWidth;

      if (i + 1 < e)
        os << ", ";
    }
    os << ")";
  };
  emitPorts(inPorts);
  os << " -> ";
  emitPorts(outPorts);
}

void Emitter::emitCell(CellOp op) {
  indent() << op.instanceName() << " = " << op.componentName() << "();\n";
}

void Emitter::emitAssignment(AssignOp op) {
  // TODO(Calyx): Support guards.
  if (op.guard())
    emitOpError(op, "guard not supported for emission currently");

  auto emitAssignmentValue = [&](auto assignValue) -> StringRef {
    auto definingOp = assignValue.getDefiningOp();
    std::string emitted;
    TypeSwitch<Operation *>(definingOp)
        .Case<CellOp>([&](auto op) {
          // A cell port should be defined as <instance-name>.<port-name>
          auto opResult = assignValue.template cast<OpResult>();
          unsigned portIndex = opResult.getResultNumber();
          auto ports = getComponentPortInfo(op.getReferencedComponent());
          StringAttr portName = ports[portIndex].name;

          emitted = op.instanceName();
          emitted += ".";
          emitted += portName.getValue();
        })
        .template Case<ConstantOp>([&](auto op) {
          // A constant is defined as <bit-width>'<base><value>, where the base
          // is `b` (binary), `o` (octal), `h` hexadecimal, or `d` (decimal).

          // Emit the Radix-10 version of the ConstantOp.
          APInt value = op.value();
          emitted += std::to_string(value.getBitWidth());
          emitted += "'d";

          SmallVector<char> stringValue;
          value.toString(stringValue, /*Radix=*/10, /*Signed=*/false);
          for (char ch : stringValue)
            emitted.push_back(ch);
        })
        .Default([&](auto op) {
          emitOpError(op, "not supported for emission inside an assignment");
        });
    return emitted;
  };
  indent() << emitAssignmentValue(op.dest()) << " = "
           << emitAssignmentValue(op.src()) << ";\n";
}

void Emitter::emitWires(WiresOp op) {
  emitCalyxSection("wires", [&]() {
    for (auto &&bodyOp : *op.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<GroupOp>([&](auto op) { emitGroup(op); })
          .Case<AssignOp>([&](auto op) { emitAssignment(op); })
          .Case<hw::ConstantOp>([&](auto op) { /* Do nothing */ })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside wires section");
          });
    }
  });
}

void Emitter::emitGroup(GroupOp op) {
  auto emitGroupBody = [&]() {
    for (auto &&bodyOp : *op.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<AssignOp>([&](auto op) { emitAssignment(op); })
          .Case<GroupDoneOp>([&](auto op) { /* TODO(Calyx): Unimplemented */ })
          .Case<GroupGoOp>([&](auto op) { /* TODO(Calyx): Unimplemented */ })
          .Case<hw::ConstantOp>([&](auto op) { /* Do nothing. */ })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside group");
          });
    }
  };
  emitCalyxSection("group", emitGroupBody, op.sym_name());
}

void Emitter::emitControl(ControlOp op) {
  // TODO(Calyx): SeqOp, EnableOp.
  emitCalyxSection("control", [&]() {});
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Emit the specified Calyx circuit into the given output stream.
mlir::LogicalResult circt::calyx::exportCalyx(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  Emitter emitter(os);
  for (auto &op : *module.getBody()) {
    if (auto programOp = dyn_cast<ProgramOp>(op))
      emitter.emitProgram(programOp);
  }
  return emitter.finalize();
}

void circt::calyx::registerToCalyxTranslation() {
  static mlir::TranslateFromMLIRRegistration toCalyx(
      "export-calyx",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return exportCalyx(module, os);
      },
      [](mlir::DialectRegistry &registry) {
        registry
            .insert<calyx::CalyxDialect, comb::CombDialect, hw::HWDialect>();
      });
}
