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
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"

using namespace circt;
using namespace hw;

// Macros for C-style error handling
#define NO_LID -1UL

namespace {
// The goal here is to traverse the operations in order and convert them one by one into btor2
struct LowerHWtoBTOR2Pass : public LowerHWtoBTOR2Base<LowerHWtoBTOR2Pass> {
  public:
    // Executes the pass
    void runOnOperation() override;

  private:
    // Create a counter that attributes a unique id to each generated btor2 line
    size_t lid = 1; // btor2 line identifiers usually start at 1

    // Create maps to keep track of lid associations
    // Proper maps wouldn't work for some reason so I'll use a vector of pairs instead
    llvm::SmallVector<std::pair<size_t, size_t>> sortToLIDMap; // Keeps track of the ids associated to each declared sort 
    llvm::DenseMap<Operation*, size_t> opLIDMap; // Connects an operation to it's most recent update line
    llvm::DenseMap<Operation*, Operation*> opAliasMap; // key: alias, value: original op

    // Set of often reused strings in btor2 emission (to avoid typos and enable auto-complete)
    const std::string SORT    = "sort";
    const std::string BITVEC  = "bitvec";
    const std::string INPUT   = "input";
    const std::string OUTPUT  = "output";
    const std::string ZERO    = "zero";
    const std::string ONE     = "one";
    const std::string CONST   = "const";
    const std::string CONSTD  = "constd";
    const std::string CONSTH  = "consth";
    const std::string SLICE   = "slice";
    const std::string UEXT    = "uext";
    const std::string ADD     = "add";
    const std::string SUB     = "sub";
    const std::string MUL     = "mul";
    const std::string CONCAT  = "concat";
    const std::string NOT     = "not";
    const std::string BAD     = "bad";
    const std::string WS      = " "; // WhiteSpace
    const std::string NL      = "\n"; // NewLine

    /// Field helper functions

    // Checks if a sort was declared with the given width
    // If so, its lid will be returned
    // Otherwise -1 will be returned
    size_t getSortLID(size_t w) {
      // Look for presence of the width in a pair
      for(auto p : sortToLIDMap) {
        // If the width was declared, return it's lid
        if(w == p.first) {
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
      for(size_t i = 0; i < sortToLIDMap.size(); ++i) {
        auto p = sortToLIDMap[i];

        // If the width was declared, update it's lid
        if(w == p.first) {
          sortToLIDMap[i] = std::make_pair<size_t, size_t>(
            std::forward<size_t>(w), 
            std::forward<size_t>(lid)
          );
          return;
        }
      }

        // Otherwise simply create a new entry 
        sortToLIDMap.push_back(std::make_pair<size_t, size_t>(
          std::forward<size_t>(w), 
          std::forward<size_t>(lid)
        ));
    }

    // Checks if an operation was declared
    // If so, its lid will be returned
    // Otherwise -1 will be returned
    size_t getOpLID(Operation* op) {
      // Look for the operation declaration
      if(opLIDMap.contains(op)) {
        return opLIDMap[op];
      }
      // If no lid was found return -1
      return NO_LID;
    }

    // Updates or creates an entry for the given operation
    // associating it with the current lid
    void setOpLID(Operation* op) {
      opLIDMap[op] = lid;
    }

    // Checks if an operation has an alias
    // If so, the original operation is returned
    // Otherwise the argument is returned as it is the original op
    Operation* getOpAlias(Operation* op) {
      // Look for the operation declaration
      if(opAliasMap.contains(op)) {
        return opAliasMap[op];
      }
      // If no lid was found return -1
      return op;
    }

    // Updates or creates an entry for the given operation
    // associating it with the current lid
    void setOpAlias(Operation* alias, Operation* op) {
      opAliasMap[alias] = op;
    }

    /// String generation helper functions

    // Generates a sort declaration instruction given a type (bitvec or array) and a width
    std::string genSort(std::string type, size_t width) {
      // Check that the sort wasn't already declared
      if(getSortLID(width) != NO_LID) {
        return ""; // If it has already been declared then return an empty string
      }

      // Add the width to the declared sorts along with the associated line id
      setSortLID(width);

      // Build and return a sort declaration
      return std::to_string(lid++) + WS + SORT + WS + BITVEC + WS + std::to_string(width) + NL; 
    }

    // Generates an input declaration given a sort lid and a name
    std::string genInput(size_t sortLid, std::string name) {
      // Sanity check: The given lid must be associated to a sort declaration
      bool found = false;
      for(auto entry : sortToLIDMap) {
        // Iterate until we've found our association
        if((found = (entry.second == sortLid))) break; 
      }

      // Check that a result was found before continuing
      assert(found);

      // Generate input declaration
      return std::to_string(lid++) + WS + INPUT + WS + std::to_string(sortLid) + WS + name + NL;
    }

    // Generates a constant declaration given a value, a width and a name
    std::string genConst(int64_t value, size_t width, Operation* op) {
      // For now we're going to assume that the name isn't taken, given that hw is already in SSA form
      setOpLID(op);

      // Retrieve the lid associated with the sort (sid)
      size_t sid = getSortLID(width);

      // Check that a result was found before continuing
      assert(sid != NO_LID);

      return std::to_string(lid++) + WS + CONSTD + WS + std::to_string(sid) + WS 
            + std::to_string(value) + NL;
    }

    // Generates a binary operation instruction given an op name, two operands and a result width
    std::string genBinOp(
      std::string inst, Operation* binop, Operation* op1, Operation* op2, size_t width
    ) {
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
      return std::to_string(lid++) + WS + inst + WS + std::to_string(sid) + WS 
            + std::to_string(op1LID) + WS + std::to_string(op2LID) + NL;
    }

    // Emits a btor2 string for the given binary operation
    void emitBinOp(std::string & btor2Res, std::string inst, Operation* binop, int64_t width) {
      // Start by extracting the operands
      Operation* op1 = binop->getOperand(0).getDefiningOp();
      Operation* op2 = binop->getOperand(1).getDefiningOp();

      // Generate a sort for the width (nothing is done is sort is defined)
      btor2Res += genSort(BITVEC, width);

      // Generate the line
      btor2Res += genBinOp(inst, binop, op1, op2, width);
    }

    // Generates a slice instruction given an operand, the lowbit, and the width
    std::string genSlice(Operation* srcop, Operation* op0, size_t lowbit, int64_t width) {
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
      return std::to_string(lid++) + WS + SLICE + WS + std::to_string(sid) + WS 
        + std::to_string(op0LID) + WS + std::to_string(width - 1) + WS + std::to_string(lowbit) + NL;
    }

    // Generates a constant declaration given a value, a width and a name
    std::string genUnaryOp(Operation* srcop, Operation* op0, std::string inst, size_t width) {
      // For now we're going to assume that the name isn't taken, given that hw is already in SSA form
      setOpLID(srcop);

      // Retrieve the lid associated with the sort (sid)
      size_t sid = getSortLID(width);

      // Check that a result was found before continuing
      assert(sid != NO_LID);

      // Assuming that the operand has already been emitted
      // Find the LID associated to the operand
      size_t op0LID = getOpLID(op0);

      return std::to_string(lid++) + WS + inst + WS + std::to_string(sid) + WS 
            + std::to_string(op0LID) + NL;
    }
    
    // Generate a btor2 assertion given an assertion operation
    // Note that a predicate inversion must have already been generated at this point
    std::string genBad(Operation* assertop) {
      // Start by finding the expression lid
      size_t assertLID = getOpLID(assertop);

      // Build and return the btor2 string
      return std::to_string(lid++) + WS + BAD + WS + std::to_string(assertLID) + NL;
    }
};
} // end anonymous namespace

void LowerHWtoBTOR2Pass::runOnOperation() {
  // String used to build out our emitted btor2
  std::string btor2Res; 

  // Start by checking for each module in the circt, for now we only consider the 1st one
  // As we are not support multi-modules yet. We assume that no nested modules exist at this point.
  // This simplifies lowering to btor as btor does not support the concept of module instanciation.
  getOperation().walk([&](hw::HWModuleOp module) {
    
    // Start by extracting the inputs and outputs from the module
    SmallVector<PortInfo> inputs, outputs;
    for(auto &port : module.getPortList()) {
      // Separate the inputs from outputs and generate the first btor2 lines for input declaration
      // We only consider ports with an explicit bit-width for now (so ignore clocks)
      if (port.isInput() && port.type.isIntOrFloat()) {
        inputs.push_back(port);
        
        // Generate the associated btor declaration for the inputs
        std::string iName = port.getName().str();  // Start by retrieving the name

        // Then only retrieve the width if the input is not a clock
        size_t width = port.type.getIntOrFloatBitWidth();

        // Generate the input and sort declaration using the extracted information
        btor2Res += genSort(BITVEC, width); // We assume all sorts are bitvectors for now
        btor2Res += genInput(getSortLID(width), iName);
      
      } else if (port.isOutput()) {
        outputs.push_back(port);
      }

    }

    // Go over all operations in our module
    module.walk([&](Operation* op) {
      // Pattern match the operation to figure out what type it is
      llvm::TypeSwitch<Operation*, void>(op) 
        // Constants are directly mapped to the btor2 constants
        .Case<hw::ConstantOp>([&](hw::ConstantOp cop) {
          // Start by figuring out what sort needs to be generated
          int64_t width = hw::getBitWidth(cop.getType());

          // Generate the sort (nothing will be added if the sort already exists)
          btor2Res += genSort(BITVEC, width);

          // Extract the associated constant value
          int64_t value = cop.getValue().getSExtValue();

          // Simply generate the operation
          btor2Res += genConst(value, width, cop);
        })
        // Wires can generally be ignored in bto2, however we do need
        // to keep track of the new alias it creates
        .Case<hw::WireOp>([&](hw::WireOp wop) {
          // Retrieve the aliased operation
          Operation* defOp = wop.getOperand().getDefiningOp();
          // Wires don't output anything so just record alias
          setOpAlias(wop, defOp);
        })
        // Outputs map to btor2 outputs on their final assignment
        .Case<hw::OutputOp>([&](hw::OutputOp op) {
          // Outputs don't actually mean much in btor, only assertions matter
        })
        // Supported Comb operations
        // Concat operations can directly be mapped to btor2 concats
        .Case<comb::ConcatOp>([&](comb::ConcatOp concatop) {
          // Extract the target width
          int64_t width = hw::getBitWidth(concatop.getType());
          
          // Emit the concat instruction
          emitBinOp(btor2Res, CONCAT, op, width);
        })
        // All binary ops are emitted the same way
        .Case<comb::AddOp>([&](comb::AddOp addop) {
          // Extract the target width
          int64_t width = hw::getBitWidth(addop.getType());
          
          // Emit the concat instruction
          emitBinOp(btor2Res, ADD, op, width);
        })
        .Case<comb::SubOp>([&](comb::SubOp subop) {
          // Extract the target width
          int64_t width = hw::getBitWidth(subop.getType());
          
          // Emit the concat instruction
          emitBinOp(btor2Res, SUB, op, width);
        })
        .Case<comb::MulOp>([&](comb::MulOp mulop) {
          // Extract the target width
          int64_t width = hw::getBitWidth(mulop.getType());
          
          // Emit the concat instruction
          emitBinOp(btor2Res, MUL, op, width);
        })
        // Extract op will translate into a slice op
        .Case<comb::ExtractOp>([&](comb::ExtractOp extop) {
          // Start by extracting the operand
          Operation* op0 = extop.getOperand().getDefiningOp();

          // Extract low bit from attributes
          size_t lb = extop.getLowBit();

          // Extract result width
          int64_t width = hw::getBitWidth(extop.getType());

          // Generate the sort (nothing will be added if the sort already exists)
          btor2Res += genSort(BITVEC, width);

          // Generate the slice instruction
          btor2Res += genSlice(extop, op0, lb, width);
        })
        .Case<comb::ICmpOp>([&](comb::ICmpOp cmpop) {
          // Extract operands
          Operation* lhs = cmpop.getOperand(0).getDefiningOp();
          Operation* rhs = cmpop.getOperand(1).getDefiningOp();

          // Extract the predicate name (assuming that its a valid btor2 predicate)
          std::string pred = stringifyICmpPredicate(cmpop.getPredicate()).str();

          // Generate a sort (width of res is always 1 for cmp)
          btor2Res += genSort(BITVEC, 1);

          // Generate the comparison btor2 instruction
          btor2Res += genBinOp(pred, cmpop, lhs, rhs, 1);
        })
        // Supported SV Operations
        .Case<circt::sv::AssertOp>([&](circt::sv::AssertOp assertop) {
          // Extract the expression
          Operation* expr = assertop.getExpression().getDefiningOp();

          // Generate a sort (for assertion inversion)
          btor2Res += genSort(BITVEC, 1);

          // Generate the expression inversion
          btor2Res += genUnaryOp(assertop, expr, NOT, 1);

          // Genrate the BAD btor2 intruction
          btor2Res += genBad(assertop);
        })
        // All other operations should be ignored for the time being
        .Default([](auto) {});
    });

    // Print out the resuling btor2
    llvm::errs() << "==========BTOR2 FORM:==========\n" 
                 << btor2Res 
                 << "\n===============================\n\n";

  });
}

// Basic constructor for the pass
std::unique_ptr<mlir::Pass> circt::hw::createLowerHWtoBTOR2Pass() {
  return std::make_unique<LowerHWtoBTOR2Pass>();
}
