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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Translation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace calyx;
using namespace mlir;

namespace {

static constexpr std::string_view LSquare() { return "["; }
static constexpr std::string_view RSquare() { return "]"; }
static constexpr std::string_view LAngleBracket() { return "<"; }
static constexpr std::string_view RAngleBracket() { return ">"; }
static constexpr std::string_view LParen() { return "("; }
static constexpr std::string_view RParen() { return ")"; }
static constexpr std::string_view colon() { return ": "; }
static constexpr std::string_view space() { return " "; }
static constexpr std::string_view period() { return "."; }
static constexpr std::string_view questionMark() { return " ? "; }
static constexpr std::string_view exclamationMark() { return "!"; }
static constexpr std::string_view equals() { return "="; }
static constexpr std::string_view comma() { return ", "; }
static constexpr std::string_view arrow() { return " -> "; }
static constexpr std::string_view delimiter() { return "\""; }
static constexpr std::string_view apostrophe() { return "'"; }
static constexpr std::string_view LBraceEndL() { return "{\n"; }
static constexpr std::string_view RBraceEndL() { return "}\n"; }
static constexpr std::string_view semicolonEndL() { return ";\n"; }
static constexpr std::string_view addressSymbol() { return "@"; }

// clang-format off
/// A list of integer attributes supported by the native Calyx compiler.
constexpr std::array<StringRef, 6> CalyxIntegerAttributes{
  "external", "static", "share", "bound", "write_together", "read_together"
};

/// A list of boolean attributes supported by the native Calyx compiler.
constexpr std::array<StringRef, 6> CalyxBooleanAttributes{
  "clk", "done", "go", "reset", "generated", "precious"
};
// clang-format on

/// Determines whether the given identifier is a valid Calyx attribute.
static bool isValidCalyxAttribute(StringRef identifier) {

  return llvm::find(CalyxIntegerAttributes, identifier) !=
             CalyxIntegerAttributes.end() ||
         llvm::find(CalyxBooleanAttributes, identifier) !=
             CalyxBooleanAttributes.end();
}

/// A tracker to determine which libraries should be imported for a given
/// program.
struct ImportTracker {
public:
  /// Returns the list of library names used for in this program.
  /// E.g. if `primitives/core.futil` is used, returns { "core" }.
  llvm::SmallSet<StringRef, 4> getLibraryNames(ProgramOp program) {
    program.walk([&](ComponentOp component) {
      for (auto &op : *component.getBody()) {
        if (!isa<CellInterface>(op) || isa<InstanceOp>(op))
          // It is not a primitive.
          continue;
        usedLibraries.insert(getLibraryFor(&op));
      }
    });
    return usedLibraries;
  }

private:
  /// Returns the library name for a given Operation Type.
  StringRef getLibraryFor(Operation *op) {
    StringRef library;
    TypeSwitch<Operation *>(op)
        .Case<MemoryOp, RegisterOp, NotLibOp, AndLibOp, OrLibOp, XorLibOp,
              AddLibOp, SubLibOp, GtLibOp, LtLibOp, EqLibOp, NeqLibOp, GeLibOp,
              LeLibOp, LshLibOp, RshLibOp, SliceLibOp, PadLibOp>(
            [&](auto op) { library = "core"; })
        .Case<SgtLibOp, SltLibOp, SeqLibOp, SneqLibOp, SgeLibOp, SleLibOp,
              SrshLibOp>([&](auto op) { library = "binary_operators"; })
        /*.Case<>([&](auto op) { library = "math"; })*/
        .Default([&](auto op) {
          llvm_unreachable("Type matching failed for this operation.");
        });
    return library;
  }
  /// Maintains a unique list of libraries used throughout the lifetime of the
  /// tracker.
  llvm::SmallSet<StringRef, 4> usedLibraries;
};

//===----------------------------------------------------------------------===//
// Emitter
//===----------------------------------------------------------------------===//

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

  /// Import emission.
  void emitImports(ProgramOp op) {
    auto emitImport = [&](StringRef library) {
      // Libraries share a common relative path:
      //   primitives/<library-name>.futil
      os << "import " << delimiter() << "primitives/" << library << period()
         << "futil" << delimiter() << semicolonEndL();
    };

    for (StringRef library : importTracker.getLibraryNames(op))
      emitImport(library);
  }

  // Component emission
  void emitComponent(ComponentOp op);
  void emitComponentPorts(ComponentOp op);

  // Instance emission
  void emitInstance(InstanceOp op);

  // Wires emission
  void emitWires(WiresOp op);

  // Group emission
  void emitGroup(GroupInterface group);

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

  // Emits a library primitive with template parameters based on all in- and
  // output ports.
  // e.g.:
  //   $f.in0, $f.in1, $f.in2, $f.out : calyx.std_foo "f" : i1, i2, i3, i4
  // emits:
  //   f = std_foo(1, 2, 3, 4);
  void emitLibraryPrimTypedByAllPorts(Operation *op);

  // Emits a library primitive with a single template parameter based on the
  // first input port.
  // e.g.:
  //   $f.in0, $f.in1, $f.out : calyx.std_foo "f" : i32, i32, i1
  // emits:
  //   f = std_foo(32);
  void emitLibraryPrimTypedByFirstInputPort(Operation *op);

private:
  /// Used to track which imports are required for this program.
  ImportTracker importTracker;

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

  /// Calyx attributes are emitted in one of the two following formats:
  /// (1)  @<attribute-name>(<attribute-value>), e.g. `@go`, `@bound(5)`.
  /// (2)  <"<attribute-name>"=<attribute-value>>, e.g. `<"static"=1>`.
  ///
  /// Since ports are structural in nature and not operations, an
  /// extra boolean value is added to determine whether this is a port of the
  /// given operation.
  std::string getAttribute(Operation *op, StringRef identifier, Attribute attr,
                           bool isPort) {
    // Verify this attribute is supported for emission.
    if (!isValidCalyxAttribute(identifier))
      return "";

    // Determines whether the attribute should follow format (2).
    bool isGroupOrComponentAttr = isa<GroupOp, ComponentOp>(op) && !isPort;

    std::string output;
    llvm::raw_string_ostream buffer(output);
    buffer.reserveExtraSpace(16);

    bool isBooleanAttribute = llvm::find(CalyxBooleanAttributes, identifier) !=
                              CalyxBooleanAttributes.end();
    if (attr.isa<UnitAttr>()) {
      assert(isBooleanAttribute &&
             "Non-boolean attributes must provide an integer value.");
      buffer << addressSymbol() << identifier << space();
    } else if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
      APInt value = intAttr.getValue();
      if (isGroupOrComponentAttr) {
        buffer << LAngleBracket() << delimiter() << identifier << delimiter()
               << equals() << value << RAngleBracket();
      } else {
        buffer << addressSymbol() << identifier;
        // The only time we may omit the value is when it is a Boolean attribute
        // with value 1.
        if (!isBooleanAttribute || intAttr.getValue() != 1)
          buffer << LParen() << value << RParen();
        buffer << space();
      }
    }
    return buffer.str();
  }

  /// Emits the attributes of a dictionary. If the `attributes` dictionary is
  /// not nullptr, we assume this is for a port.
  std::string getAttributes(Operation *op,
                            DictionaryAttr attributes = nullptr) {
    bool isPort = attributes != nullptr;
    if (!isPort)
      attributes = op->getAttrDictionary();

    SmallString<16> calyxAttributes;
    for (auto &&[id, attr] : attributes)
      calyxAttributes.append(getAttribute(op, id, attr, isPort));
    return calyxAttributes.c_str();
  }

  /// Helper function for emitting a Calyx section. It emits the body in the
  /// following format:
  /// {
  ///   <body>
  /// }
  template <typename Func>
  void emitCalyxBody(Func emitBody) {
    os << space() << LBraceEndL();
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
      os << space() << symbolName;
    emitCalyxBody(emitBody);
  }

  /// Helper function for emitting combinational operations.
  template <typename CombinationalOp>
  void emitCombinationalValue(CombinationalOp op, StringRef logicalSymbol) {
    auto inputs = op.inputs();
    os << LParen();
    for (size_t i = 0, e = inputs.size(); i != e; ++i) {
      emitValue(inputs[i], /*isIndented=*/false);
      if (i + 1 == e)
        continue;
      os << space() << logicalSymbol << space();
    }
    os << RParen();
  }

  /// Emits the value of a guard or assignment.
  void emitValue(Value value, bool isIndented) {
    if (auto blockArg = value.dyn_cast<BlockArgument>()) {
      // Emit component block argument.
      StringAttr portName = getPortInfo(blockArg).name;
      (isIndented ? indent() : os) << portName.getValue();
      return;
    }

    auto definingOp = value.getDefiningOp();
    assert(definingOp && "Value does not have a defining operation.");

    TypeSwitch<Operation *>(definingOp)
        .Case<CellInterface>([&](auto cell) {
          // A cell port should be defined as <instance-name>.<port-name>
          (isIndented ? indent() : os)
              << cell.instanceName() << period() << cell.portName(value);
        })
        .Case<hw::ConstantOp>([&](auto op) {
          // A constant is defined as <bit-width>'<base><value>, where the base
          // is `b` (binary), `o` (octal), `h` hexadecimal, or `d` (decimal).
          APInt value = op.value();

          (isIndented ? indent() : os)
              << std::to_string(value.getBitWidth()) << apostrophe() << "d";
          // We currently default to the decimal representation.
          value.print(os, /*isSigned=*/false);
        })
        .Case<comb::AndOp>([&](auto op) { emitCombinationalValue(op, "&"); })
        .Case<comb::OrOp>([&](auto op) { emitCombinationalValue(op, "|"); })
        .Case<comb::XorOp>([&](auto op) {
          // The XorOp is a bit different, since the Combinational dialect
          // uses it to represent binary not.
          if (!op.isBinaryNot()) {
            emitOpError(op, "Only supporting Binary Not for XOR.");
            return;
          }
          // The LHS is the value to be negated, and the RHS is a constant with
          // all ones (guaranteed by isBinaryNot).
          os << exclamationMark();
          emitValue(op.inputs()[0], /*isIndented=*/false);
        })
        .Default(
            [&](auto op) { emitOpError(op, "not supported for emission"); });
  }

  /// Emits a port for a Group.
  template <typename OpTy>
  void emitGroupPort(GroupInterface group, OpTy op, StringRef portHole) {
    assert((isa<GroupGoOp>(op) || isa<GroupDoneOp>(op)) &&
           "Required to be a group port.");
    indent() << group.symName().getValue() << LSquare() << portHole << RSquare()
             << space() << equals() << space();
    if (op.guard()) {
      emitValue(op.guard(), /*isIndented=*/false);
      os << questionMark();
    }
    emitValue(op.src(), /*isIndented=*/false);
    os << semicolonEndL();
  }

  /// Recursively emits the Calyx control.
  void emitCalyxControl(Block *body) {
    Operation *parent = body->getParentOp();
    assert(isa<ControlOp>(parent) ||
           parent->hasTrait<ControlLike>() &&
               "This should only be used to emit Calyx Control structures.");

    // Check to see if this is a stand-alone EnableOp, i.e.
    // calyx.control { calyx.enable @G }
    if (auto enable = dyn_cast<EnableOp>(parent)) {
      emitEnable(enable);
      // Early return since an EnableOp has no body.
      return;
    }
    // Attribute dictionary is always prepended for a control operation.
    auto prependAttributes = [&](Operation *op, StringRef sym) {
      return (getAttributes(op) + sym).str();
    };

    for (auto &&op : *body) {

      TypeSwitch<Operation *>(&op)
          .Case<SeqOp>([&](auto op) {
            emitCalyxSection(prependAttributes(op, "seq"),
                             [&]() { emitCalyxControl(op.getBody()); });
          })
          .Case<ParOp>([&](auto op) {
            emitCalyxSection(prependAttributes(op, "par"),
                             [&]() { emitCalyxControl(op.getBody()); });
          })
          .Case<WhileOp>([&](auto op) {
            indent() << prependAttributes(op, "while ");
            emitValue(op.cond(), /*isIndented=*/false);

            if (auto groupName = op.groupName(); groupName.hasValue())
              os << " with " << groupName.getValue();

            emitCalyxBody([&]() { emitCalyxControl(op.getBody()); });
          })
          .Case<IfOp>([&](auto op) {
            indent() << prependAttributes(op, "if ");
            emitValue(op.cond(), /*isIndented=*/false);

            if (auto groupName = op.groupName(); groupName.hasValue())
              os << " with " << groupName.getValue();

            emitCalyxBody([&]() { emitCalyxControl(op.getThenBody()); });
            if (op.elseRegionExists())
              emitCalyxSection("else",
                               [&]() { emitCalyxControl(op.getElseBody()); });
          })
          .Case<EnableOp>([&](auto op) { emitEnable(op); })
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
  indent() << "component " << op.getName() << getAttributes(op);
  // Emit the ports.
  emitComponentPorts(op);
  os << space() << LBraceEndL();
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
          .Case<hw::ConstantOp>([&](auto op) { /*Do nothing*/ })
          .Case<SliceLibOp, PadLibOp>(
              [&](auto op) { emitLibraryPrimTypedByAllPorts(op); })
          .Case<LtLibOp, GtLibOp, EqLibOp, NeqLibOp, GeLibOp, LeLibOp, SltLibOp,
                SgtLibOp, SeqLibOp, SneqLibOp, SgeLibOp, SleLibOp, AddLibOp,
                SubLibOp, ShruLibOp, RshLibOp, SrshLibOp, LshLibOp, AndLibOp,
                NotLibOp, OrLibOp, XorLibOp>(
              [&](auto op) { emitLibraryPrimTypedByFirstInputPort(op); })
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
void Emitter::emitComponentPorts(ComponentOp op) {
  auto emitPorts = [&](auto ports) {
    os << LParen();
    for (size_t i = 0, e = ports.size(); i < e; ++i) {
      const PortInfo &port = ports[i];

      // We only care about the bit width in the emitted .futil file.
      auto bitWidth = port.type.getIntOrFloatBitWidth();
      os << getAttributes(op, port.attributes) << port.name.getValue()
         << colon() << bitWidth;

      if (i + 1 < e)
        os << comma();
    }
    os << RParen();
  };
  emitPorts(op.getInputPortInfo());
  os << arrow();
  emitPorts(op.getOutputPortInfo());
}

void Emitter::emitInstance(InstanceOp op) {
  indent() << getAttributes(op) << op.instanceName() << space() << equals()
           << space() << op.componentName() << LParen() << RParen()
           << semicolonEndL();
}

void Emitter::emitRegister(RegisterOp reg) {
  size_t bitWidth = reg.inPort().getType().getIntOrFloatBitWidth();
  indent() << getAttributes(reg) << reg.instanceName() << space() << equals()
           << space() << "std_reg" << LParen() << std::to_string(bitWidth)
           << RParen() << semicolonEndL();
}

void Emitter::emitMemory(MemoryOp memory) {
  size_t dimension = memory.sizes().size();
  if (dimension < 1 || dimension > 4) {
    emitOpError(memory, "Only memories with dimensionality in range [1, 4] are "
                        "supported by the native Calyx compiler.");
    return;
  }
  indent() << getAttributes(memory) << memory.instanceName() << space()
           << equals() << space() << "std_mem_d" << std::to_string(dimension)
           << LParen() << memory.width() << comma();
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

/// Calling getName() on a calyx operation will return "calyx.${opname}". This
/// function returns whatever is left after the first '.' in the string,
/// removing the 'calyx' prefix.
static StringRef removeCalyxPrefix(StringRef s) { return s.split(".").second; }

void Emitter::emitLibraryPrimTypedByAllPorts(Operation *op) {
  auto cell = cast<CellInterface>(op);
  indent() << getAttributes(op) << cell.instanceName() << space() << equals()
           << space() << removeCalyxPrefix(op->getName().getStringRef())
           << LParen();
  llvm::interleaveComma(op->getResults(), os, [&](auto res) {
    os << std::to_string(res.getType().getIntOrFloatBitWidth());
  });
  os << RParen() << semicolonEndL();
}

void Emitter::emitLibraryPrimTypedByFirstInputPort(Operation *op) {
  auto cell = cast<CellInterface>(op);
  unsigned bitWidth = cell.getInputPorts()[0].getType().getIntOrFloatBitWidth();
  StringRef opName = op->getName().getStringRef();
  indent() << getAttributes(op) << cell.instanceName() << space() << equals()
           << space() << removeCalyxPrefix(opName) << LParen() << bitWidth
           << RParen() << semicolonEndL();
}

void Emitter::emitAssignment(AssignOp op) {

  emitValue(op.dest(), /*isIndented=*/true);
  os << space() << equals() << space();
  if (op.guard()) {
    emitValue(op.guard(), /*isIndented=*/false);
    os << questionMark();
  }
  emitValue(op.src(), /*isIndented=*/false);
  os << semicolonEndL();
}

void Emitter::emitWires(WiresOp op) {
  emitCalyxSection("wires", [&]() {
    for (auto &&bodyOp : *op.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<GroupInterface>([&](auto op) { emitGroup(op); })
          .Case<AssignOp>([&](auto op) { emitAssignment(op); })
          .Case<hw::ConstantOp, comb::AndOp, comb::OrOp, comb::XorOp>(
              [&](auto op) { /* Do nothing. */ })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside wires section");
          });
    }
  });
}

void Emitter::emitGroup(GroupInterface group) {
  auto emitGroupBody = [&]() {
    for (auto &&bodyOp : *group.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<AssignOp>([&](auto op) { emitAssignment(op); })
          .Case<GroupDoneOp>([&](auto op) { emitGroupPort(group, op, "done"); })
          .Case<GroupGoOp>([&](auto op) { emitGroupPort(group, op, "go"); })
          .Case<hw::ConstantOp, comb::AndOp, comb::OrOp, comb::XorOp>(
              [&](auto op) { /* Do nothing. */ })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside group.");
          });
    }
  };

  StringRef prefix = isa<CombGroupOp>(group) ? "comb group" : "group";
  auto groupHeader = (group.symName().getValue() + getAttributes(group)).str();
  emitCalyxSection(prefix, emitGroupBody, groupHeader);
}

void Emitter::emitEnable(EnableOp enable) {
  indent() << getAttributes(enable) << enable.groupName() << semicolonEndL();
}

void Emitter::emitControl(ControlOp control) {
  emitCalyxSection("control", [&]() { emitCalyxControl(control.getBody()); });
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Emit the specified Calyx circuit into the given output stream.
mlir::LogicalResult circt::calyx::exportCalyx(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  Emitter emitter(os);
  for (auto &op : *module.getBody()) {
    op.walk([&](ProgramOp program) {
      emitter.emitImports(program);
      emitter.emitProgram(program);
    });
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
