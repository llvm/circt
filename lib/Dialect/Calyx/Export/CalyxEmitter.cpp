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

  // Instance emission
  void emitInstance(InstanceOp op);

  // Wires emission
  void emitWires(WiresOp op);

  // Group emission
  void emitGroup(GroupOp group);

  // Control emission
  void emitControl(ControlOp control);

  // Assignment emission
  void emitAssignment(AssignOp op);

  // Enable emission
  void emitEnable(EnableOp enable);

  // Register emission
  void emitRegister(RegisterOp reg);

  // Memory emission
  void emitMemory(MemoryOp memory);

private:
  std::string LBrace() { return "{"; }
  std::string RBrace() { return "}"; }
  std::string LSquare() { return "["; }
  std::string RSquare() { return "]"; }
  std::string LParen() { return "("; }
  std::string RParen() { return ")"; }
  std::string colon() { return ": "; }
  std::string semicolon() { return ";"; }
  std::string Space() { return " "; }
  std::string period() { return "."; }
  std::string equals() { return " = "; }
  std::string comma() { return ", "; }
  std::string arrow() { return " -> "; }
  std::string apostrophe() { return "'"; }
  std::string endL() { return "\n"; }
  std::string LBraceEndL() { return LBrace() + endL(); }
  std::string RBraceEndL() { return RBrace() + endL(); }
  std::string semicolonEndL() { return semicolon() + endL(); }

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

  /// Helper function for emitting a Calyx section. It emits the body in the
  /// following format:
  /// {
  ///   <body>
  /// }
  template <typename Func>
  void emitCalyxBody(Func emitBody) {
    os << Space() << LBraceEndL();
    addIndent();
    emitBody();
    reduceIndent();
    indent() << RBraceEndL();
  }

  /// Emits a Calyx section.
  template <typename Func>
  void emitCalyxSection(StringRef sectionName, Func emitBody,
                        StringRef symbolName = "") {
    indent() << sectionName;
    if (!symbolName.empty())
      os << Space() << symbolName;
    emitCalyxBody(emitBody);
  }

  /// Overloaded version for names that require port emission in the section
  /// name as well, e.g. WhileOp has the following section name:
  ///      while <port-name> with <group-name> { ... }
  template <typename Func>
  void emitCalyxSection(Func emitBody, StringRef symbolName = "") {
    if (!symbolName.empty())
      os << Space() << symbolName;
    emitCalyxBody(emitBody);
  }

  /// Emits the value of a guard or assignment.
  void emitValue(Value value, bool isIndented) {
    auto definingOp = value.getDefiningOp();
    TypeSwitch<Operation *>(definingOp)
        .Case<InstanceOp>([&](auto op) {
          // A cell port should be defined as <instance-name>.<port-name>
          auto opResult = value.cast<OpResult>();
          unsigned portIndex = opResult.getResultNumber();
          auto ports = getComponentPortInfo(op.getReferencedComponent());
          StringAttr portName = ports[portIndex].name;
          (isIndented ? indent() : os)
              << op.instanceName() << period() << portName.getValue();
        })
        .Case<hw::ConstantOp>([&](auto op) {
          // A constant is defined as <bit-width>'<base><value>, where the base
          // is `b` (binary), `o` (octal), `h` hexadecimal, or `d` (decimal).

          // Emit the Radix-10 version of the ConstantOp.
          APInt value = op.value();

          (isIndented ? indent() : os)
              << std::to_string(value.getBitWidth()) << apostrophe() << "d";
          value.print(os, /*isSigned=*/false);
        })
        .Default(
            [&](auto op) { emitOpError(op, "not supported for emission"); });
  }

  /// Emits a port for a Group.
  template <typename OpTy>
  void emitGroupPort(GroupOp group, OpTy op, StringRef portHole) {
    assert((isa<GroupGoOp>(op) || isa<GroupDoneOp>(op)) &&
           "Required to be a group port.");

    if (op.guard())
      emitOpError(op, "Guards not supported for emission yet.");
    indent() << group.sym_name() << LSquare() << portHole << RSquare()
             << equals();
    emitValue(op.src(), /*isIndented=*/false);
    os << semicolonEndL();
  }

  /// Recursively emits the Calyx control.
  template <typename OpTy>
  void emitCalyxControl(OpTy controlOp) {
    // Check to see if this is a stand-alone EnableOp.
    if (isa<EnableOp>(controlOp)) {
      emitEnable(cast<EnableOp>(controlOp));
      return;
    }
    for (auto &&bodyOp : *controlOp.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .template Case<SeqOp>([&](auto op) {
            emitCalyxSection("seq", [&]() { emitCalyxControl(op); });
          })
          .template Case<WhileOp>([&](auto op) {
            indent() << "while ";
            emitValue(op.cond(), /*isIndented=*/false);
            os << " with " << op.groupName();
            emitCalyxSection([&]() { emitCalyxControl(op); });
          })
          .template Case<IfOp>([&](auto op) {
            indent() << "if ";
            emitValue(op.cond(), /*isIndented=*/false);
            os << " with " << op.groupName();
            emitCalyxSection([&]() { emitCalyxControl(op); });
          })
          .template Case<EnableOp>([&](auto op) { emitEnable(op); })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside control.");
          });
    }
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
  os << Space() << LBraceEndL();
  addIndent();
  WiresOp wires;
  ControlOp control;

  // Emit cells.
  emitCalyxSection("cells", [&]() {
    for (auto &&bodyOp : *op.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<WiresOp>([&](auto op) { wires = op; })
          .Case<ControlOp>([&](auto op) { control = op; })
          .Case<InstanceOp>([&](auto op) { emitInstance(op); })
          .Case<RegisterOp>([&](auto op) { emitRegister(op); })
          .Case<MemoryOp>([&](auto op) { emitMemory(op); })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside component");
          });
    }
  });

  emitWires(wires);
  emitControl(control);
  reduceIndent();
  os << RBraceEndL();
}

/// Emit the ports of a component.
void Emitter::emitComponentPorts(ArrayRef<ComponentPortInfo> ports) {
  std::vector<ComponentPortInfo> inPorts, outPorts;
  for (auto &&port : ports) {
    if (port.direction == Direction::Input)
      inPorts.push_back(port);
    else
      outPorts.push_back(port);
  }

  auto emitPorts = [&](auto ports) {
    os << LParen();
    for (size_t i = 0, e = ports.size(); i < e; ++i) {
      const auto &port = ports[i];
      auto name = port.name.getValue();
      // We only care about the bit width in the emitted .futil file.
      auto bitWidth = port.type.getIntOrFloatBitWidth();
      os << name << colon() << bitWidth;

      if (i + 1 < e)
        os << comma();
    }
    os << RParen();
  };
  emitPorts(inPorts);
  os << arrow();
  emitPorts(outPorts);
}

void Emitter::emitInstance(InstanceOp op) {
  indent() << op.instanceName() << equals() << op.componentName() << LParen()
           << RParen() << semicolonEndL();
}

void Emitter::emitRegister(RegisterOp reg) {
  size_t bitWidth = reg.inPort().getType().getIntOrFloatBitWidth();
  indent() << reg.instanceName() << equals() << "std_reg" << LParen()
           << std::to_string(bitWidth) << RParen() << semicolonEndL();
}

void Emitter::emitMemory(MemoryOp memory) {
  size_t dimension = memory.sizes().size();
  if (dimension < 1 || dimension > 4) {
    emitOpError(memory, "Only memories with dimensionality in range [1, 4] are "
                        "supported by the native Calyx compiler.");
    return;
  }
  indent() << memory.instanceName() << " = std_mem_d"
           << std::to_string(dimension) << LParen() << memory.width()
           << comma();
  for (Attribute size : memory.sizes()) {
    APInt memSize = size.cast<IntegerAttr>().getValue();
    memSize.print(os, /*isSigned=*/false);
    os << comma();
  }

  ArrayAttr addrSizes = memory.addrSizes();
  for (size_t i = 0, e = addrSizes.size(); i != e; ++i) {
    APInt addrSize = addrSizes[i].cast<IntegerAttr>().getValue();
    addrSize.print(os, /*isSigned=*/false);
    if (i + 1 == e)
      continue;
    os << comma();
  }
  os << RParen() << semicolonEndL();
}

void Emitter::emitAssignment(AssignOp op) {
  // TODO(Calyx): Support guards.
  if (op.guard())
    emitOpError(op, "guard not supported for emission currently");

  emitValue(op.dest(), /*isIndented=*/true);
  os << equals();
  emitValue(op.src(), /*isIndented=*/false);
  os << semicolonEndL();
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

void Emitter::emitGroup(GroupOp group) {
  auto emitGroupBody = [&]() {
    for (auto &&bodyOp : *group.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<AssignOp>([&](auto op) { emitAssignment(op); })
          .Case<GroupDoneOp>([&](auto op) { emitGroupPort(group, op, "done"); })
          .Case<GroupGoOp>([&](auto op) { emitGroupPort(group, op, "go"); })
          .Case<hw::ConstantOp>([&](auto op) { /* Do nothing. */ })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside group.");
          });
    }
  };
  emitCalyxSection("group", emitGroupBody, group.sym_name());
}

void Emitter::emitEnable(EnableOp enable) {
  indent() << enable.groupName() << semicolonEndL();
}

void Emitter::emitControl(ControlOp control) {
  emitCalyxSection("control", [&]() { emitCalyxControl(control); });
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
