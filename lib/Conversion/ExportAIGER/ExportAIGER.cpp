//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AIGER file export functionality. AIGER is a format
// for representing sequential circuits as AIGs (And-Inverter Graphs). This
// implementation supports both ASCII and binary AIGER formats.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportAIGER.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOpInterfaces.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/Version.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::aig;
using namespace circt::seq;
using namespace circt::aiger;

#define DEBUG_TYPE "export-aiger"

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Returns the bit width of a value.
static int64_t getBitWidth(Value value) {
  return hw::getBitWidth(value.getType());
}

//===----------------------------------------------------------------------===//
// AIGERExporter Implementation
//===----------------------------------------------------------------------===//

namespace {

/// Main AIGER exporter class that handles the conversion of MLIR modules to
/// AIGER format. This class manages the state and provides methods for
/// analyzing the module and writing the AIGER file.
class AIGERExporter {
public:
  AIGERExporter(hw::HWModuleOp module, llvm::raw_ostream &os,
                const ExportAIGEROptions &options, ExportAIGERHandler *handler)
      : module(module), os(os), options(options), handler(handler) {}

  /// Export the module to AIGER format.
  LogicalResult exportModule();

  // Data type for tracking values and their bit positions
  using Object = std::pair<Value, int>;

private:
  hw::HWModuleOp module;
  llvm::raw_ostream &os;
  const ExportAIGEROptions &options;
  ExportAIGERHandler *handler;

  // Map of values to their literals
  DenseMap<Object, unsigned> valueLiteralMap;

  // AIGER file data
  unsigned getNumInputs() { return inputs.size(); }
  unsigned getNumLatches() { return latches.size(); }
  unsigned getNumOutputs() { return outputs.size(); }
  unsigned getNumAnds() { return andGates.size(); }

  llvm::EquivalenceClasses<Object> valueEquivalence;
  SmallVector<std::pair<Object, StringAttr>> inputs;
  SmallVector<std::tuple<Object, StringAttr, Object>>
      latches; // current, name, next
  SmallVector<std::pair<Object, StringAttr>> outputs;
  SmallVector<std::tuple<Object, Object, Object>> andGates; // lhs, rhs0, rhs1

  std::optional<Value> clock;

  /// Analyze the module and collect information
  LogicalResult analyzeModule();

  /// Get the index name for an object, handling bit positions
  StringAttr getIndexName(Object obj, StringAttr name) {
    if (!options.includeSymbolTable || !name || name.getValue().empty() ||
        obj.first.getType().isInteger(1))
      return name;

    return StringAttr::get(name.getContext(),
                           name.getValue() + "[" + Twine(obj.second) + "]");
  }

  /// Add an input to the AIGER file
  void addInput(Object obj, StringAttr name = {}) {
    inputs.push_back({obj, getIndexName(obj, name)});
  }

  /// Add a latch to the AIGER file
  void addLatch(Object current, Object next, StringAttr name = {}) {
    latches.push_back({current, getIndexName(current, name), next});
  }

  /// Add an output to the AIGER file
  void addOutput(Object obj, StringAttr name = {}) {
    outputs.push_back({obj, getIndexName(obj, name)});
  }

  /// Analyze module ports (inputs/outputs)
  LogicalResult analyzePorts(hw::HWModuleOp module);

  /// Analyze operations in the module
  LogicalResult analyzeOperations(hw::HWModuleOp module);

  /// Assign literals to all values
  LogicalResult assignLiterals();

  /// Writers
  LogicalResult writeHeader();
  LogicalResult writeInputs();
  LogicalResult writeLatches();
  LogicalResult writeOutputs();
  LogicalResult writeAndGates();
  LogicalResult writeSymbolTable();
  LogicalResult writeComments();

  /// Visit an operation and process it
  LogicalResult visit(Operation *op);

  /// Visit specific operation types
  LogicalResult visit(aig::AndInverterOp op);
  LogicalResult visit(seq::CompRegOp op);
  LogicalResult visit(comb::ConcatOp op);
  LogicalResult visit(comb::ExtractOp op);
  LogicalResult visit(comb::ReplicateOp op);

  /// Get or assign a literal for a value
  unsigned getLiteral(Object obj, bool inverted = false);

  /// Helper method to write unsigned LEB128 encoded integers
  void writeUnsignedLEB128(unsigned value);
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// AIGERExporter Implementation
//===----------------------------------------------------------------------===//

LogicalResult AIGERExporter::exportModule() {
  LLVM_DEBUG(llvm::dbgs() << "Starting AIGER export\n");
  if (failed(analyzeModule()) || failed(writeHeader()) ||
      failed(writeInputs()) || failed(writeLatches()) ||
      failed(writeOutputs()) || failed(writeAndGates()) ||
      failed(writeSymbolTable()) || failed(writeComments()))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "AIGER export completed successfully\n");
  return success();
}

LogicalResult AIGERExporter::analyzeModule() {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing module\n");

  auto topModule = module;
  LLVM_DEBUG(llvm::dbgs() << "Found top module: " << topModule.getModuleName()
                          << "\n");

  // Analyze module ports
  if (failed(analyzePorts(topModule)))
    return failure();

  // Walk through all operations in the module body
  if (failed(analyzeOperations(topModule)))
    return failure();

  // Assign literals to all values
  if (failed(assignLiterals()))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Analysis complete: M="
                          << getNumInputs() + getNumLatches() + getNumAnds()
                          << " I=" << getNumInputs() << " L=" << getNumLatches()
                          << " O=" << getNumOutputs() << " A=" << getNumAnds()
                          << "\n");

  return success();
}

LogicalResult AIGERExporter::writeHeader() {
  LLVM_DEBUG(llvm::dbgs() << "Writing AIGER header\n");

  // Write format identifier
  if (options.binaryFormat)
    os << "aig";
  else
    os << "aag";

  // Write M I L O A
  os << " " << getNumInputs() + getNumLatches() + getNumAnds() << " "
     << getNumInputs() << " " << getNumLatches() << " " << getNumOutputs()
     << " " << getNumAnds() << "\n";

  return success();
}

LogicalResult AIGERExporter::writeInputs() {
  LLVM_DEBUG(llvm::dbgs() << "Writing inputs\n");

  if (options.binaryFormat)
    // In binary format, inputs are implicit
    return success();

  // Write input literals
  for (auto [input, name] : inputs)
    os << getLiteral(input) << "\n";

  return success();
}

LogicalResult AIGERExporter::writeLatches() {
  LLVM_DEBUG(llvm::dbgs() << "Writing latches\n");

  // Write latch definitions
  for (auto [current, currentName, next] : latches) {

    // For next state, we need to handle potential inversions
    // Check if next state comes from an inverter or has inversions
    unsigned nextLiteral = getLiteral(next);

    if (options.binaryFormat) {
      // In binary format, current literal is not needed
      os << nextLiteral << "\n";
    } else {
      os << getLiteral(current) << " " << nextLiteral << "\n";
    }
  }

  return success();
}

LogicalResult AIGERExporter::writeOutputs() {
  LLVM_DEBUG(llvm::dbgs() << "Writing outputs\n");

  // Write output literals
  for (auto [output, _] : outputs)
    os << getLiteral(output) << "\n";

  return success();
}

LogicalResult AIGERExporter::writeAndGates() {
  LLVM_DEBUG(llvm::dbgs() << "Writing AND gates\n");

  if (options.binaryFormat) {
    // Implement binary format encoding for AND gates
    for (auto [lhs, rhs0, rhs1] : andGates) {
      unsigned lhsLiteral = getLiteral(lhs);

      // Get the AND-inverter operation to check inversion flags
      auto andInvOp = lhs.first.getDefiningOp<aig::AndInverterOp>();
      assert(andInvOp && andInvOp.getInputs().size() == 2);

      // Get operand literals with inversion
      bool rhs0Inverted = andInvOp.getInverted()[0];
      bool rhs1Inverted = andInvOp.getInverted()[1];

      unsigned rhs0Literal = getLiteral(rhs0, rhs0Inverted);
      unsigned rhs1Literal = getLiteral(rhs1, rhs1Inverted);

      // Ensure rhs0 >= rhs1 as required by AIGER format
      if (rhs0Literal < rhs1Literal) {
        std::swap(rhs0Literal, rhs1Literal);
        std::swap(rhs0, rhs1);
        std::swap(rhs0Inverted, rhs1Inverted);
      }

      // In binary format, we need to write the delta values
      // Delta0 = lhs - rhs0
      // Delta1 = rhs0 - rhs1
      unsigned delta0 = lhsLiteral - rhs0Literal;
      unsigned delta1 = rhs0Literal - rhs1Literal;

      LLVM_DEBUG(llvm::dbgs()
                 << "Writing AND gate: " << lhsLiteral << " = " << rhs0Literal
                 << " & " << rhs1Literal << " (deltas: " << delta0 << ", "
                 << delta1 << ")\n");
      assert(lhsLiteral >= rhs0Literal && "lhsLiteral >= rhs0Literal");
      assert(rhs0Literal >= rhs1Literal && "rhs0Literal >= rhs1Literal");

      // Write deltas using variable-length encoding
      writeUnsignedLEB128(delta0);
      writeUnsignedLEB128(delta1);
    }
  } else {
    // ASCII format - no need for structural hashing
    for (auto [lhs, rhs0, rhs1] : andGates) {
      unsigned lhsLiteral = getLiteral(lhs);

      // Get the AND-inverter operation to check inversion flags
      auto andInvOp = lhs.first.getDefiningOp<aig::AndInverterOp>();

      assert(andInvOp && andInvOp.getInputs().size() == 2);

      // Get operand literals with inversion
      bool rhs0Inverted = andInvOp.getInverted()[0];
      bool rhs1Inverted = andInvOp.getInverted()[1];

      unsigned rhs0Literal = getLiteral(rhs0, rhs0Inverted);
      unsigned rhs1Literal = getLiteral(rhs1, rhs1Inverted);

      os << lhsLiteral << " " << rhs0Literal << " " << rhs1Literal << "\n";
    }
  }

  return success();
}

LogicalResult AIGERExporter::writeSymbolTable() {
  LLVM_DEBUG(llvm::dbgs() << "Writing symbol table\n");

  if (!options.includeSymbolTable)
    return success();

  for (auto [index, elem] : llvm::enumerate(inputs)) {
    auto [obj, name] = elem;
    // Don't write inputs without a name
    if (!name)
      continue;
    os << "i" << index << " " << name.getValue() << "\n";
  }

  for (auto [index, elem] : llvm::enumerate(latches)) {
    auto [current, currentName, next] = elem;
    // Don't write latches without a name
    if (!currentName)
      continue;
    os << "l" << index << " " << currentName.getValue() << "\n";
  }

  for (auto [index, elem] : llvm::enumerate(outputs)) {
    auto [obj, name] = elem;
    // Don't write outputs without a name
    if (!name)
      continue;
    os << "o" << index << " " << name.getValue() << "\n";
  }

  return success();
}

LogicalResult AIGERExporter::writeComments() {
  LLVM_DEBUG(llvm::dbgs() << "Writing comments\n");

  // Write comment section
  os << "c\n";
  os << "Generated by " << circt::getCirctVersion() << "\n";
  os << "module: " << module.getModuleName() << "\n";

  return success();
}

unsigned AIGERExporter::getLiteral(Object obj, bool inverted) {
  // Get the leader value from the equivalence class
  obj = valueEquivalence.getOrInsertLeaderValue(obj);

  // Handle constants
  auto value = obj.first;
  auto pos = obj.second;
  {
    auto it = valueLiteralMap.find({value, pos});
    if (it != valueLiteralMap.end()) {
      unsigned literal = it->second;
      return inverted ? literal ^ 1 : literal;
    }
  }

  if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
    APInt constValue = constOp.getValue();
    if (constValue[obj.second])
      // TRUE constant = literal 1, inverted = literal 0 (FALSE)
      return inverted ? 0 : 1;
    // FALSE constant = literal 0, inverted = literal 1 (TRUE)
    return inverted ? 1 : 0;
  }

  // Handle single-input AND-inverter (pure inverter)
  if (auto andInvOp = value.getDefiningOp<aig::AndInverterOp>()) {
    if (andInvOp.getInputs().size() == 1) {
      // This is a pure inverter - we need to find the input's literal
      Value input = andInvOp.getInputs()[0];
      bool inputInverted = andInvOp.getInverted()[0];
      unsigned inputLiteral = getLiteral({input, pos}, inputInverted);
      // Apply additional inversion if requested
      valueLiteralMap[{value, pos}] = inputLiteral;
      return inverted ? inputLiteral ^ 1 : inputLiteral;
    }
  }

  // Concatenation, replication, and extraction
  assert((!isa_and_nonnull<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp>(
             value.getDefiningOp())) &&
         "Unhandled operation type");

  // Look up in the literal map
  auto it = valueLiteralMap.find({value, pos});
  if (it != valueLiteralMap.end()) {
    unsigned literal = it->second;
    return inverted ? literal ^ 1 : literal;
  }

  llvm::errs() << "Unhandled: Value not found in literal map: " << value << "["
               << pos << "]\n";

  // This should not happen if analysis was done correctly
  llvm_unreachable("Value not found in literal map");
}

LogicalResult AIGERExporter::analyzePorts(hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing module ports\n");

  auto inputNames = hwModule.getInputNames();
  auto outputNames = hwModule.getOutputNames();

  // Analyze input ports
  for (auto [index, arg] :
       llvm::enumerate(hwModule.getBodyBlock()->getArguments())) {
    // Skip non integer inputs.
    if (!arg.getType().isInteger())
      continue;

    // All other inputs should be i1 (1-bit assumption)
    for (int64_t i = 0; i < getBitWidth(arg); ++i) {
      if (!handler || handler->valueCallback(arg, i, inputs.size()))
        addInput({arg, i},
                 llvm::dyn_cast_or_null<StringAttr>(inputNames[index]));
    }

    LLVM_DEBUG(llvm::dbgs() << "  Input " << index << ": " << arg << "\n");
  }

  // Analyze output ports by looking at hw.output operation
  auto *outputOp = hwModule.getBodyBlock()->getTerminator();
  for (auto [operand, name] :
       llvm::zip(outputOp->getOpOperands(), outputNames)) {
    for (int64_t i = 0; i < getBitWidth(operand.get()); ++i)
      if (!handler || handler->operandCallback(operand, i, outputs.size()))
        addOutput({operand.get(), i}, llvm::dyn_cast_or_null<StringAttr>(name));
    LLVM_DEBUG(llvm::dbgs() << "  Output: " << operand.get() << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << getNumInputs() << " inputs, "
                          << getNumOutputs() << " outputs from ports\n");
  return success();
}

LogicalResult AIGERExporter::analyzeOperations(hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing operations\n");

  // Sort the operations topologically
  auto result = hwModule.walk([&](Region *region) {
    auto regionKindOp = dyn_cast<RegionKindInterface>(region->getParentOp());
    if (!regionKindOp ||
        regionKindOp.hasSSADominance(region->getRegionNumber()))
      return WalkResult::advance();

    // Graph region.
    for (auto &block : *region) {
      if (!sortTopologically(&block, [&](Value value, Operation *op) {
            // Topologically sort AND-inverters and purely dataflow ops. Other
            // operations can be scheduled.
            return !(isa<aig::AndInverterOp, comb::ExtractOp, comb::ReplicateOp,
                         comb::ConcatOp>(op));
          }))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return mlir::emitError(hwModule.getLoc(),
                           "failed to sort operations topologically");

  // Visit each operation in the module
  for (Operation &op : hwModule.getBodyBlock()->getOperations()) {
    if (failed(visit(&op)))
      return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << getNumAnds() << " AND gates, "
                          << getNumLatches() << " latches\n");
  return success();
}

LogicalResult AIGERExporter::visit(Operation *op) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<aig::AndInverterOp>([&](auto op) { return visit(op); })
      .Case<seq::CompRegOp>([&](auto op) { return visit(op); })
      .Case<comb::ConcatOp>([&](auto op) { return visit(op); })
      .Case<comb::ExtractOp>([&](auto op) { return visit(op); })
      .Case<comb::ReplicateOp>([&](auto op) { return visit(op); })
      .Case<hw::HWModuleOp, hw::ConstantOp, hw::OutputOp>(
          [](auto) { return success(); })
      .Default([&](Operation *op) -> LogicalResult {
        // Handle unknown operations if handler is set
        if (!handler)
          return mlir::emitError(op->getLoc(), "unhandled operation: ") << *op;

        // Handle unknown operations through the handler
        for (mlir::OpOperand &operand : op->getOpOperands()) {
          if (operand.get().getType().isInteger())
            for (int64_t i = 0; i < getBitWidth(operand.get()); ++i) {
              if (handler->operandCallback(operand, i, outputs.size()))
                addOutput({operand.get(), i});
            }
        }

        for (mlir::OpResult result : op->getOpResults()) {
          if (!result.getType().isInteger())
            continue;
          for (int64_t i = 0; i < getBitWidth(result); ++i) {
            if (handler->valueCallback(result, i, inputs.size()))
              addInput({result, i});
            else
              // Treat it as a constant
              valueLiteralMap[{result, i}] = 0;
          }
        }

        return success();
      });
}

LogicalResult AIGERExporter::visit(aig::AndInverterOp op) {
  // Handle AIG AND-inverter operations
  if (op.getInputs().size() == 1) {
    // Single input = inverter (not counted as AND gate in AIGER)
    // Inversion flag is looked at in getLiteral()
    LLVM_DEBUG(llvm::dbgs() << "  Found inverter: " << op << "\n");
  } else if (op.getInputs().size() == 2) {
    // Two inputs = AND gate
    for (int64_t i = 0; i < getBitWidth(op); ++i)
      andGates.push_back({{op.getResult(), i},
                          {op.getInputs()[0], i},
                          {op.getInputs()[1], i}});

    LLVM_DEBUG(llvm::dbgs() << "  Found AND gate: " << op << "\n");
  } else {
    // Variadic AND gates need to be lowered first
    return mlir::emitError(op->getLoc(),
                           "variadic AND gates not supported, run "
                           "aig-lower-variadic pass first");
  }
  if (handler)
    handler->notifyEmitted(op);
  return success();
}

LogicalResult AIGERExporter::visit(seq::CompRegOp op) {
  // Handle registers (latches in AIGER)
  for (int64_t i = 0; i < getBitWidth(op); ++i)
    addLatch({op.getResult(), i}, {op.getInput(), i}, op.getNameAttr());

  LLVM_DEBUG(llvm::dbgs() << "  Found latch: " << op << "\n");
  if (handler)
    handler->notifyEmitted(op);
  return success();
}

LogicalResult AIGERExporter::visit(comb::ConcatOp op) {
  size_t pos = 0;
  for (auto operand : llvm::reverse(op.getInputs()))
    for (int64_t i = 0, e = getBitWidth(operand); i < e; ++i)
      valueEquivalence.unionSets({operand, i}, {op.getResult(), pos++});
  return success();
}

LogicalResult AIGERExporter::visit(comb::ExtractOp op) {
  for (int64_t i = 0, e = getBitWidth(op); i < e; ++i)
    valueEquivalence.unionSets({op.getInput(), op.getLowBit() + i},
                               {op.getResult(), i});
  return success();
}

LogicalResult AIGERExporter::visit(comb::ReplicateOp op) {
  auto operandWidth = getBitWidth(op.getInput());
  for (int64_t i = 0, e = getBitWidth(op); i < e; ++i)
    valueEquivalence.unionSets({op.getInput(), i % operandWidth},
                               {op.getResult(), i});
  return success();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const AIGERExporter::Object &obj) {
  return os << obj.first << "[" << obj.second << "]";
}

llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const std::pair<AIGERExporter::Object, StringAttr> &obj) {
  return os << obj.first
            << (obj.second ? " (" + obj.second.getValue() + ")" : "");
}

LogicalResult AIGERExporter::assignLiterals() {
  LLVM_DEBUG(llvm::dbgs() << "Assigning literals\n");

  unsigned nextLiteral =
      2; // Start from 2 (literal 0 = FALSE, literal 1 = TRUE)

  // Assign literals to inputs first
  for (auto input : inputs) {
    valueLiteralMap[input.first] = nextLiteral;
    LLVM_DEBUG(llvm::dbgs()
               << "  Input literal " << nextLiteral << ": " << input << "\n");
    nextLiteral += 2; // Even literals only (odd = inverted)
  }

  // Assign literals to latches (current state)
  for (auto [current, currentName, next] : latches) {
    valueLiteralMap[current] = nextLiteral;
    LLVM_DEBUG(llvm::dbgs()
               << "  Latch literal " << nextLiteral << ": " << current << "\n");
    nextLiteral += 2;
    auto clocked = current.first.getDefiningOp<seq::Clocked>();
    assert(clocked);
    if (!clock)
      clock = clocked.getClk();
    else if (clock != clocked.getClk()) {
      auto diag = mlir::emitError(clocked.getClk().getLoc(),
                                  "multiple clocks found in the module");
      diag.attachNote(clock->getLoc()) << "previous clock is here";
      return diag;
    }
  }

  if (clock && handler)
    handler->notifyClock(*clock);

  for (auto [lhs, rhs0, rhs1] : andGates) {
    valueLiteralMap[lhs] = nextLiteral;
    LLVM_DEBUG(llvm::dbgs()
               << "  AND gate literal " << nextLiteral << ": " << lhs << "\n");
    nextLiteral += 2;
  }

  LLVM_DEBUG(llvm::dbgs() << "Assigned " << valueLiteralMap.size()
                          << " literals\n");
  return success();
}

//===----------------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------------===//

LogicalResult circt::aiger::exportAIGER(hw::HWModuleOp module,
                                        llvm::raw_ostream &os,
                                        const ExportAIGEROptions *options,
                                        ExportAIGERHandler *handler) {
  ExportAIGEROptions defaultOptions;
  if (!options)
    options = &defaultOptions;

  AIGERExporter exporter(module, os, *options, handler);
  return exporter.exportModule();
}

llvm::cl::opt<bool>
    emitTextFormat("emit-text-format",
                   llvm::cl::desc("Export AIGER in text format"),
                   llvm::cl::init(false));

llvm::cl::opt<bool>
    includeSymbolTable("exclude-symbol-table",
                       llvm::cl::desc("Exclude symbol table from the output"),
                       llvm::cl::init(false));

void circt::aiger::registerExportAIGERTranslation() {
  static mlir::TranslateFromMLIRRegistration toAIGER(
      "export-aiger", "Export AIG to AIGER format",
      [](mlir::ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        auto ops = module.getOps<hw::HWModuleOp>();
        if (ops.empty())
          return module.emitError("no HW module found in the input");
        if (std::next(ops.begin()) != ops.end())
          return module.emitError(
              "multiple HW modules found, expected single top module");

        ExportAIGEROptions options;
        options.binaryFormat = !emitTextFormat;
        options.includeSymbolTable = !includeSymbolTable;

        return exportAIGER(*ops.begin(), os, &options);
      },
      [](DialectRegistry &registry) {
        registry.insert<aig::AIGDialect, hw::HWDialect, seq::SeqDialect,
                        comb::CombDialect>();
      });
}

// Helper method to write unsigned LEB128 encoded integers
void AIGERExporter::writeUnsignedLEB128(unsigned value) {
  do {
    uint8_t byte = value & 0x7f;
    value >>= 7;
    if (value != 0)
      byte |= 0x80; // Set high bit if more bytes follow
    os.write(reinterpret_cast<char *>(&byte), 1);
  } while (value != 0);
}
