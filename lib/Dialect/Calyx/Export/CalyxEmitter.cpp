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
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Translation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace calyx;

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

private:
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
    if (port.direction == PortDirection::INPUT)
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

void Emitter::emitWires(WiresOp op) {
  emitCalyxSection("wires", [&]() {
    for (auto &&bodyOp : *op.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<GroupOp>([&](auto op) { emitGroup(op); })
          // TODO(Calyx): Assignments
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside wires section");
          });
    }
  });
}

void Emitter::emitGroup(GroupOp op) {
  emitCalyxSection(
      // TODO(Calyx): Assignments, GoOp, DoneOp, combinational guards.
      "group", [&]() {}, op.sym_name());
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
        registry.insert<calyx::CalyxDialect>();
      });
}
