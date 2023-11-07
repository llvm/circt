//===- LowerHWtoBTOR2.cpp - Lowers a hw module to btor2 --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Converts a hw module to a btor2 format and prints it out
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace hw;

// Macros for C-style error handling
#define NO_LID -1UL
#define NO_WIDTH -1L

namespace {
// The goal here is to traverse the operations in order and convert them one by
// one into btor2
struct LowerHWtoBTOR2Pass : public LowerHWtoBTOR2Base<LowerHWtoBTOR2Pass> {
public:
  // Executes the pass
  void runOnOperation() override;

private:
  // Create a counter that attributes a unique id to each generated btor2 line
  size_t lid = 1;           // btor2 line identifiers usually start at 1
  size_t resetLID = NO_LID; // keeps track of the reset's LID
  bool isAsync = false; // Records whether or not the reset is async (changes
                        // how we handle resets)

  // Create maps to keep track of lid associations
  // Proper maps wouldn't work for some reason so I'll use a vector of pairs
  // instead
  SmallVector<std::pair<size_t, size_t>>
      sortToLIDMap; // Keeps track of the ids associated to each declared sort
  DenseMap<std::pair<int64_t, size_t>, size_t>
      constToLIDMap; // Keeps track of the
  DenseMap<Operation *, size_t>
      opLIDMap; // Connects an operation to it's most recent update line
  DenseMap<Operation *, Operation *>
      opAliasMap;                     // key: alias, value: original op
  DenseMap<size_t, size_t> inputLIDs; // Stores the LID of the associated input
                                      // (key: block argument index)
  SmallVector<Operation *> regOps; // Stores all of the register declaration ops
                                   // (for next instruction generation)

  // Set of often reused strings in btor2 emission (to avoid typos and enable
  // auto-complete)
  const static constexpr StringLiteral SORT = "sort";
  const static constexpr StringLiteral BITVEC = "bitvec";
  const static constexpr StringLiteral INPUT = "input";
  const static constexpr StringLiteral RESET = "reset";
  const static constexpr StringLiteral OUTPUT = "output";
  const static constexpr StringLiteral ZERO = "zero";
  const static constexpr StringLiteral ONE = "one";
  const static constexpr StringLiteral CONST = "const";
  const static constexpr StringLiteral CONSTD = "constd";
  const static constexpr StringLiteral CONSTH = "consth";
  const static constexpr StringLiteral SLICE = "slice";
  const static constexpr StringLiteral UEXT = "uext";
  const static constexpr StringLiteral ADD = "add";
  const static constexpr StringLiteral SUB = "sub";
  const static constexpr StringLiteral MUL = "mul";
  const static constexpr StringLiteral AND = "and";
  const static constexpr StringLiteral OR = "or";
  const static constexpr StringLiteral XOR = "xor";
  const static constexpr StringLiteral SLL = "sll";
  const static constexpr StringLiteral SRL =
      "srl"; // a.k.a. unsigned right shift
  const static constexpr StringLiteral SRA = "sra"; // a.k.a. signed right shift
  const static constexpr StringLiteral SDIV = "sdiv";
  const static constexpr StringLiteral UDIV = "udiv";
  const static constexpr StringLiteral SMOD = "smod";
  const static constexpr StringLiteral CONCAT = "concat";
  const static constexpr StringLiteral NOT = "not";
  const static constexpr StringLiteral NEQ = "neq";
  const static constexpr StringLiteral HW_NEQ = "ne";
  const static constexpr StringLiteral ITE = "ite";
  const static constexpr StringLiteral IMPLIES =
      "implies";                                        // logical implication
  const static constexpr StringLiteral STATE = "state"; // Register state
  const static constexpr StringLiteral NEXT =
      "next"; // Register state transition
  const static constexpr StringLiteral BAD = "bad";
  const static constexpr StringLiteral CONSTRAINT = "constraint";
  const static constexpr StringLiteral WS = " ";  // WhiteSpace
  const static constexpr StringLiteral NL = "\n"; // NewLine

  /// Field helper functions

  // Checks if a constant of a given size has been declared
  // If so, its lid will be returned
  // Otherwise -1 will be returned
  size_t getConstLID(int64_t val, size_t w) {
    // Look for the pair
    for (auto p : constToLIDMap) {
      // Compare the given value and width to those in the entry
      if ((val == p.getFirst().first) && (w == p.getFirst().second)) {
        return p.getSecond();
      }
    }
    // if no lid was found return -1
    return NO_LID;
  }

  // Updates or creates an entryfor  a constant of a given size
  // associating it with the current lid
  void setConstLID(int64_t val, size_t w) {
    // Create a new entry for the pair
    constToLIDMap[std::make_pair<int64_t, size_t>(
        std::forward<int64_t>(val), std::forward<size_t>(w))] = lid;
  }

  // Checks if a sort was declared with the given width
  // If so, its lid will be returned
  // Otherwise -1 will be returned
  size_t getSortLID(size_t w) {
    // Look for presence of the width in a pair
    for (auto p : sortToLIDMap) {
      // If the width was declared, return it's lid
      if (w == p.first) {
        return p.second;
      }
    }
    // If no lid was found return -1
    return NO_LID;
  }

  // Updates or creates an entry for the given width
  // associating it with the current lid
  void setSortLID(size_t w) {
    // Check for the existence of the sort
    for (size_t i = 0; i < sortToLIDMap.size(); ++i) {
      auto p = sortToLIDMap[i];

      // If the width was declared, update it's lid
      if (w == p.first) {
        sortToLIDMap[i] = std::make_pair<size_t, size_t>(
            std::forward<size_t>(w), std::forward<size_t>(lid));
        return;
      }
    }

    // Otherwise simply create a new entry
    sortToLIDMap.push_back(std::make_pair<size_t, size_t>(
        std::forward<size_t>(w), std::forward<size_t>(lid)));
  }

  // Checks if an operation was declared
  // If so, its lid will be returned
  // Otherwise -1 will be returned
  size_t getOpLID(Operation *op) {
    // Look for the operation declaration
    Operation *defOp = getOpAlias(op);
    if (opLIDMap.contains(defOp)) {
      return opLIDMap[defOp];
    }

    // If no lid was found return -1
    return NO_LID;
  }

  // Checks if an operation was declared
  // If so, its lid will be returned
  // Otherwise -1 will be returned
  size_t getOpLID(Value op) {
    // Look for the operation declaration
    Operation *defOp = getOpAlias(op.getDefiningOp());

    if (opLIDMap.contains(defOp)) {
      return opLIDMap[defOp];
    }

    // Check for special case where op is actually a port
    // To do so, we start by checking if our operation isa block argument
    if (BlockArgument barg = dyn_cast<BlockArgument>(op)) {
      // Extract the block argument index and use that to get the line number
      size_t argIdx = barg.getArgNumber();

      // Check that the extracted argument is in range
      if (inputLIDs.contains(argIdx)) {
        return inputLIDs[argIdx];
      }
    }

    // If no lid was found return -1
    return NO_LID;
  }

  // Updates or creates an entry for the given operation
  // associating it with the current lid
  void setOpLID(Operation *op) {
    if (op != nullptr) {
      opLIDMap[op] = lid;
    }
  }

  // Checks if an operation has an alias
  // If so, the original operation is returned
  // Otherwise the argument is returned as it is the original op
  Operation *getOpAlias(Operation *op) {
    // Look for the operation declaration
    if (opAliasMap.contains(op)) {
      return opAliasMap[op];
    }
    // If no lid was found return -1
    return op;
  }

  // Updates or creates an entry for the given operation
  // associating it with the current lid
  void setOpAlias(Operation *alias, Operation *op) { opAliasMap[alias] = op; }

  /// String generation helper functions

  // Generates a sort declaration instruction given a type (bitvec or array) and
  // a width
  void genSort(StringRef type, size_t width) {
    // Check that the sort wasn't already declared
    if (getSortLID(width) != NO_LID) {
      return; // If it has already been declared then return an empty string
    }

    // Add the width to the declared sorts along with the associated line id
    setSortLID(width);

    // Build and return a sort declaration
    llvm::outs() << lid++ << WS << SORT << WS << BITVEC << WS << width << NL;
  }

  // Generates an input declaration given a sort lid and a name
  void genInput(size_t width, StringRef name) {
    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(width);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    // Generate input declaration
    llvm::outs() << lid++ << WS << INPUT << WS << sid << WS << name << NL;
  }

  // Generates a constant declaration given a value, a width and a name
  void genConst(int64_t value, size_t width, Operation *op) {
    // For now we're going to assume that the name isn't taken, given that hw is
    // already in SSA form
    setOpLID(op);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(width);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    llvm::outs() << lid++ << WS << CONSTD << WS << sid << WS << value << NL;
  }

  // Generates a zero constant expression
  void genZero(size_t width) {
    // Check if the constant has been created yet
    if (getConstLID(0, width) != NO_LID) {
      return;
    }

    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(width);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    // Keep track of this value in a constant declaration tracker
    setConstLID(0, width);

    // Build and return the zero btor instruction
    llvm::outs() << lid++ << WS << ZERO << WS << sid << NL;
  }

  // Generates a binary operation instruction given an op name, two operands and
  // a result width
  void genBinOp(StringRef inst, Operation *binop, Value op1, Value op2,
                size_t width) {
    // Set the LID for this operation
    setOpLID(binop);

    // Find the sort's lid
    size_t sid = getSortLID(width);

    // Sanity check
    assert(sid != NO_LID);

    // Assuming that the operands were already emitted
    // Find the LIDs associated to the operands
    size_t op1LID = getOpLID(op1);
    size_t op2LID = getOpLID(op2);

    // Build and return the string
    llvm::outs() << lid++ << WS << inst << WS << sid << WS << op1LID << WS
                 << op2LID << NL;
  }

  // Emits a btor2 string for the given binary operation
  void emitBinOp(StringRef &btor2Res, StringRef inst, Operation *binop,
                 int64_t width) {
    // Start by extracting the operands
    Value op1 = binop->getOperand(0);
    Value op2 = binop->getOperand(1);

    // Make sure that the correct sort definition exists
    genSort(BITVEC, width);

    // Generate the line
    genBinOp(inst, binop, op1, op2, width);
  }

  // Generates a slice instruction given an operand, the lowbit, and the width
  void genSlice(Operation *srcop, Value op0, size_t lowbit, int64_t width) {
    // Set the LID for this operation
    setOpLID(srcop);

    // Find the sort's lid
    size_t sid = getSortLID(width);

    // Sanity check
    assert(sid != NO_LID);

    // Assuming that the operand has already been emitted
    // Find the LID associated to the operand
    size_t op0LID = getOpLID(op0);

    // Build and return the slice instruction
    llvm::outs() << lid++ << WS << SLICE << WS << sid << WS << op0LID << WS
                 << (width - 1) << WS << lowbit << NL;
  }

  // Generates a constant declaration given a value, a width and a name
  void genUnaryOp(Operation *srcop, Operation *op0, StringRef inst,
                  size_t width) {
    // Register the source operation with the current line id
    setOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(width);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    // Assuming that the operand has already been emitted
    // Find the LID associated to the operand
    size_t op0LID = getOpLID(op0);

    llvm::outs() << lid++ << WS << inst << WS << sid << WS << op0LID << NL;
  }

  // Generates a constant declaration given a value, a width and a name
  void genUnaryOp(Operation *srcop, Value op0, StringRef inst, size_t width) {
    genUnaryOp(srcop, op0.getDefiningOp(), inst, width);
  }

  // Generate a btor2 assertion given an assertion operation
  // Note that a predicate inversion must have already been generated at this
  // point
  void genBad(Operation *assertop) {
    // Start by finding the expression lid
    size_t assertLID = getOpLID(assertop);

    // Build and return the btor2 string
    llvm::outs() << lid++ << WS << BAD << WS << assertLID << NL;
  }

  // Generate a btor2 constraint given an expression from an assumption
  // operation
  void genConstraint(Value expr) {
    // Start by finding the expression lid
    size_t exprLID = getOpLID(expr);

    // Build and return the btor2 string
    llvm::outs() << lid++ << WS << CONSTRAINT << WS << exprLID << NL;
  }

  // Generate a btor2 constraint given an expression from an assumption
  // operation
  void genConstraint(size_t exprLID) {
    // Build and return the btor2 string
    llvm::outs() << lid++ << WS << CONSTRAINT << WS << exprLID << NL;
  }

  // Generate an ITE instruction (if then else) given a predicate, two values
  // and a res width
  void genIte(Operation *srcop, Value cond, Value t, Value f, int64_t width) {
    // Register the source operation with the current line id
    setOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(width);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    // Retrieve the operand lids, assuming they were emitted
    size_t condLID = getOpLID(cond);
    size_t tLID = getOpLID(t);
    size_t fLID = getOpLID(f);

    // Build and return the ite instruction
    llvm::outs() << lid++ << WS << ITE << WS << sid << WS << condLID << WS
                 << tLID << WS << fLID << NL;
  }

  // Generate an ITE instruction (if then else) given a predicate, two values
  // and a res width
  void genIte(Operation *srcop, size_t condLID, size_t tLID, size_t fLID,
              int64_t width) {
    // Register the source operation with the current line id
    setOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(width);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    // Build and return the ite instruction
    llvm::outs() << lid++ << WS << ITE << WS << sid << WS << condLID << WS
                 << tLID << WS << fLID << NL;
  }

  // Generate a logical implication given a lhs and a rhs
  void genImplies(Operation *srcop, Value lhs, Value rhs) {
    // Register the source operation with the current line id
    setOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(1);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    // Retrieve LIDs for the lhs and rhs
    size_t lhsLID = getOpLID(lhs);
    size_t rhsLID = getOpLID(rhs);

    // Build and return the implies operation
    llvm::outs() << lid++ << WS << IMPLIES << WS << sid << WS << lhsLID << WS
                 << rhsLID << NL;
  }

  // Generate a logical implication given a lhs and a rhs
  void genImplies(Operation *srcop, size_t lhsLID, size_t rhsLID) {
    // Register the source operation with the current line id
    setOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(1);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    // Build and return the implies operation
    llvm::outs() << lid++ << WS << IMPLIES << WS << sid << WS << lhsLID << WS
                 << rhsLID << NL;
  }

  // Generates a state instruction given a width and a name
  void genState(Operation *srcop, int64_t width, StringRef name) {
    // Register the source operation with the current line id
    setOpLID(srcop);

    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(width);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    // Build and return the state instruction
    llvm::outs() << lid++ << WS << STATE << WS << sid << WS << name << NL;
  }

  // Generates a next instruction, given a width, a state LID, and a next value
  // LID
  void genNext(Operation *next, Operation *reg, int64_t width) {
    // Retrieve the lid associated with the sort (sid)
    size_t sid = getSortLID(width);

    // Check that a result was found before continuing
    assert(sid != NO_LID);

    // Retrieve the LIDs associated to reg and next
    size_t regLID = getOpLID(reg);
    size_t nextLID = getOpLID(next);

    // Build and return the next instruction
    llvm::outs() << lid++ << WS << NEXT << WS << sid << WS << regLID << WS
                 << nextLID << NL;
  }

  // Verifies that the sort required for the given operation's btor2 emission
  // has been generated
  int64_t requireSort(mlir::Type type) {
    // Start by figuring out what sort needs to be generated
    int64_t width = hw::getBitWidth(type);
    assert(width != NO_WIDTH);

    // Generate the sort regardles of resulting width (nothing will be added if
    // the sort already exists)
    genSort(BITVEC, width);
    return width;
  }

  /// Visitor Methods used later on for pattern matching

  // Visitor for the inputs of the module.
  // This will generate additional sorts and input declaration explicitly for
  // btor2 Note that outputs are ignored in btor2 as they do not contribute to
  // the final assertions
  void visit(hw::PortInfo &port) {
    // Separate the inputs from outputs and generate the first btor2 lines for
    // input declaration We only consider ports with an explicit bit-width (so
    // ignore clocks)
    if (port.isInput() && !port.type.isa<seq::ClockType>()) {
      // Generate the associated btor declaration for the inputs
      StringRef iName = port.getName();

      // Guarantees that a sort will exist for the generation of this port's
      // translation into btor2
      int64_t w = requireSort(port.type);

      // Record the defining operation's line ID (the module itself in the case
      // of ports)
      inputLIDs[port.argNum] = lid;

      // We assume that the explicit name is always %reset for reset ports
      if (iName == RESET)
        resetLID = lid;

      genInput(w, iName);
    }
  }

  // Outputs don't actually mean much in btor, only assertions matter
  // Additionally, btormc doesn't support outputs, so we're just going to
  // ignore them
  void visit(hw::OutputOp op) {}

  // Emits the associated btor2 operation for a constant. Note that for
  // simplicity, we will only emit `constd` in order to avoid bit-string
  // conversions
  void visit(hw::ConstantOp op) {
    // Make sure that a sort has been created for our operation
    int64_t w = requireSort(op.getType());

    // Prepare for for const generation by extracting the const value and
    // generting the btor2 string
    int64_t value = op.getValue().getSExtValue();
    genConst(value, w, op);
  }

  // Wires can generally be ignored in bto2, however we do need
  // to keep track of the new alias it creates
  void visit(hw::WireOp op) {
    // Retrieve the aliased operation
    Operation *defOp = op.getOperand().getDefiningOp();
    // Wires don't output anything so just record alias
    setOpAlias(op, defOp);
  }

  // Binary operations are all emitted the same way, so we can group them into
  // a single method.
  // @param {StringRef} inst, the btor2 name of the operation
  void visitBinOp(Operation *op, StringRef inst) {
    TypeSwitch<Operation *, void>(op)
        .template Case<
            // All supported binary ops
            comb::AddOp, comb::SubOp, comb::MulOp, comb::DivUOp, comb::DivSOp,
            comb::ModSOp, comb::ShlOp, comb::ShrUOp, comb::ShrSOp, comb::AndOp,
            comb::OrOp, comb::XorOp, comb::ConcatOp>([&](auto expr) {
          int64_t w = requireSort(expr.getType());

          // Start by extracting the operands
          Value op1 = binop->getOperand(0);
          Value op2 = binop->getOperand(1);

          // Generate the line
          genBinOp(inst, op, op1, op2, w);
        })
        // Ignore anything else
        .Default([&](auto expr) {});
  }

  // Visitors for the binary ops
  void visit(comb::AddOp op) { visitBinOp(op, ADD); }
  void visit(comb::SubOp op) { visitBinOp(op, SUB); }
  void visit(comb::MulOp op) { visitBinOp(op, MUL); }
  void visit(comb::DivSOp op) { visitBinOp(op, SDIV); }
  void visit(comb::DivUOp op) { visitBinOp(op, UDIV); }
  void visit(comb::ModSOp op) { visitBinOp(op, SMOD); }
  void visit(comb::ShlOp op) { visitBinOp(op, SLL); }
  void visit(comb::ShrUOp op) { visitBinOp(op, SRL); }
  void visit(comb::ShrSOp op) { visitBinOp(op, SRA); }
  void visit(comb::AndOp op) { visitBinOp(op, AND); }
  void visit(comb::OrOp op) { visitBinOp(op, OR); }
  void visit(comb::XorOp op) { visitBinOp(op, XOR); }
  void visit(comb::ConcatOp op) { visitBinOp(op, CONCAT); }

  // Extract ops translate to a slice operation in btor2 in a one-to-one manner
  void visit(comb::ExtractOp op) {
    int64_t w = requireSort(op.getType());

    // Start by extracting the necessary information for the emission (i.e.
    // operand, low bit, ...)
    Value op0 = op.getOperand();
    size_t lb = op.getLowBit();

    // Generate the slice instruction
    genSlice(op, op0, lb, w);
  }

  // Btor2 uses similar syntax as hw for its comparisons
  // So we simply need to emit the cmpop name and check for corner cases
  // where the namings differ.
  void visit(comb::ICmpOp op) {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    // Extract the predicate name (assuming that its a valid btor2
    // predicate)
    StringRef pred = stringifyICmpPredicate(op.getPredicate());

    // Check for special cases where hw doesn't align with btor syntax
    if (pred == HW_NEQ)
      pred = NEQ;

    // Width of result is always 1 for comparison
    genSort(BITVEC, 1);

    // With the special cases out of the way, the emission is the same as that
    // of a binary op
    genBinOp(pred, op, lhs, rhs, 1);
  }

  // Muxes generally convert to an ite statement
  void visit(comb::MuxOp op) {
    // Extract predicate, true and false values
    Value pred = op.getOperand(0);
    Value tval = op.getOperand(1);
    Value fval = op.getOperand(2);

    // We assume that both tval and fval have the same width
    // This width should be the same as the output width
    int64_t w = requireSort(op.getType());

    // Generate the ite instruction
    genIte(op, pred, tval, fval, w);
  }

  // Assertions are negated then converted to a btor2 BAD instruction
  void visit(sv::AssertOp op) {
    // Expression is what we will try to invert for our assertion
    Value expr = op.getExpression();

    // This sort is for assertion inversion and potential implies
    genSort(BITVEC, 1);

    // Check for an overaching enable
    // In our case the sv.if operation will probably only be used when
    // conditioning an sv.assert on an enable signal. This means that
    // its condition is probably used to imply our assertion
    if (auto ifop = dyn_cast<sv::IfOp>(((Operation *)op)->getParentOp())) {
      Value en = ifop.getOperand();

      // Generate the implication
      genImplies(ifop, en, expr);

      // Generate the implies inversion
      genUnaryOp(op, ifop, NOT, 1);
    } else {
      // Generate the expression inversion
      genUnaryOp(op, expr, NOT, 1);
    }

    // Genrate the BAD btor2 intruction
    genBad(op);
  }
  // Assumptions are converted to a btor2 constraint instruction
  void visit(sv::AssumeOp op) {
    // Extract the expression that we want our constraint to be about
    Value expr = op.getExpression();
    genConstraint(expr);
  }

  // Firrtl registers generate a state instruction
  // The final update is also used to generate a set of next btor
  // instructions
  void visit(seq::FirRegOp reg) {
    // Start by retrieving the register's name and width
    StringRef regName = reg.getName();
    int64_t w = requireSort(reg.getType());

    // Generate state instruction (represents the register declaration)
    genState(reg, w, regName);

    // Record the operation for future `next` instruction generation
    // This is required to model transitions between states (i.e. how a
    // register's value evolves over time)
    regOps.push_back(reg);
  }
};
} // end anonymous namespace

void LowerHWtoBTOR2Pass::runOnOperation() {
  // Start by checking for each module in the circt, we only consider
  // the 1st one, as btor2 does not have the concept of modules or module
  // hierarchies. We assume that no nested modules exist at this point.
  // This greatly simplifies translation.
  getOperation().walk([&](hw::HWModuleOp module) {
    // Start by extracting the inputs and generating appropriate instructions
    for (auto &port : module.getPortList()) {
      visit(port);
    }

    // Visit all of the operations in our module
    module.walk([&](Operation *op) {
      TypeSwitch<Operation *, void>(op)
          .template Case<
              // All supported hw operations
              hw::OutputOp, hw::ConstantOp, hw::WireOp,
              // All supported comb ops
              comb::AddOp, comb::SubOp, comb::MulOp, comb::DivUOp, comb::DivSOp,
              comb::ModSOp, comb::ShlOp, comb::ShrUOp, comb::ShrSOp,
              comb::AndOp, comb::OrOp, comb::XorOp, comb::ConcatOp,
              comb::ExtractOp, comb::ICmpOp, comb::MuxOp,
              // All supported sv operations
              sv::AssertOp, sv::AssumeOp,
              // All supported seq operations
              seq::FirRegOp>([&](auto expr) { visit(expr); })
          // Ignore anything else
          .Default([&](auto expr) {});
    });

    // Iterate through the registers and generate the `next` instructions
    for (size_t i = 0; i < regOps.size(); ++i) {
      // Check the register type (done to support non-firrtl registers as well)
      if (seq::FirRegOp reg = dyn_cast<seq::FirRegOp>(regOps[i])) {
        // Extract the `next` operation for each register (used to define the
        // transition)
        Operation *next = reg.getNext().getDefiningOp();

        // Genrate the reset condition (for sync & async resets)
        // We assume for now that the reset value is always 0
        size_t width = hw::getBitWidth(reg.getType());
        genSort(BITVEC, width);
        genZero(width);

        // Next should already be associated to an LID at this point
        // As we are going to override it, we need to keep track of the original
        // instruction
        size_t nextLID = getOpLID(next);

        // Generate the ite for the register update reset condition
        // i.e. reg <= reset ? 0 : next
        genIte(next, resetLID, getConstLID(0, width), nextLID, width);

        // Finally generate the next statement
        genNext(next, reg, width);
      }
    }

    llvm::outs() << "\n===============================\n\n";
  });
}

// Basic constructor for the pass
std::unique_ptr<mlir::Pass> circt::hw::createLowerHWtoBTOR2Pass() {
  return std::make_unique<LowerHWtoBTOR2Pass>();
}
