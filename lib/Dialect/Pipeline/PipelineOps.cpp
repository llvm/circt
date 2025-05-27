//===- PipelineOps.h - Pipeline MLIR Operations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the Pipeline ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Pipeline/PipelineOps.h"
#include "circt/Support/ParsingUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace circt;
using namespace circt::pipeline;
using namespace circt::parsing_util;

#include "circt/Dialect/Pipeline/PipelineDialect.cpp.inc"

#define DEBUG_TYPE "pipeline-ops"

llvm::SmallVector<Value>
circt::pipeline::detail::getValuesDefinedOutsideRegion(Region &region) {
  llvm::SetVector<Value> values;
  region.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (region.isAncestor(operand.getParentRegion()))
        continue;
      values.insert(operand);
    }
  });
  return values.takeVector();
}

Block *circt::pipeline::getParentStageInPipeline(ScheduledPipelineOp pipeline,
                                                 Block *block) {
  // Optional debug check - ensure that 'block' eventually leads to the
  // pipeline.
  LLVM_DEBUG({
    Operation *directParent = block->getParentOp();
    if (directParent != pipeline) {
      auto indirectParent =
          directParent->getParentOfType<ScheduledPipelineOp>();
      assert(indirectParent == pipeline && "block is not in the pipeline");
    }
  });

  while (block && block->getParent() != &pipeline.getRegion()) {
    // Go one level up.
    block = block->getParent()->getParentOp()->getBlock();
  }

  // This is a block within the pipeline region, so it must be a stage.
  return block;
}

Block *circt::pipeline::getParentStageInPipeline(ScheduledPipelineOp pipeline,
                                                 Operation *op) {
  return getParentStageInPipeline(pipeline, op->getBlock());
}

Block *circt::pipeline::getParentStageInPipeline(ScheduledPipelineOp pipeline,
                                                 Value v) {
  if (isa<BlockArgument>(v))
    return getParentStageInPipeline(pipeline,
                                    cast<BlockArgument>(v).getOwner());
  return getParentStageInPipeline(pipeline, v.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// Fancy pipeline-like op printer/parser functions.
//===----------------------------------------------------------------------===//

// Parses a list of operands on the format:
//   (name : type, ...)
static ParseResult parseOutputList(OpAsmParser &parser,
                                   llvm::SmallVector<Type> &inputTypes,
                                   mlir::ArrayAttr &outputNames) {

  llvm::SmallVector<Attribute> names;
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            StringRef name;
            Type type;
            if (parser.parseKeyword(&name) || parser.parseColonType(type))
              return failure();

            inputTypes.push_back(type);
            names.push_back(StringAttr::get(parser.getContext(), name));
            return success();
          }))
    return failure();

  outputNames = ArrayAttr::get(parser.getContext(), names);
  return success();
}

static void printOutputList(OpAsmPrinter &p, TypeRange types, ArrayAttr names) {
  p << "(";
  llvm::interleaveComma(llvm::zip(types, names), p, [&](auto it) {
    auto [type, name] = it;
    p.printKeywordOrString(cast<StringAttr>(name).str());
    p << " : " << type;
  });
  p << ")";
}

static ParseResult parseKeywordAndOperand(OpAsmParser &p, StringRef keyword,
                                          OpAsmParser::UnresolvedOperand &op) {
  if (p.parseKeyword(keyword) || p.parseLParen() || p.parseOperand(op) ||
      p.parseRParen())
    return failure();
  return success();
}

// Assembly format is roughly:
// ( $name )? initializer-list stall (%stall = $stall)?
//   clock (%clock) reset (%reset) go(%go) entryEnable(%en) {
//   --- elided inner block ---
static ParseResult parsePipelineOp(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  std::string name;
  if (succeeded(parser.parseOptionalString(&name)))
    result.addAttribute("name", parser.getBuilder().getStringAttr(name));

  llvm::SmallVector<OpAsmParser::UnresolvedOperand> inputOperands;
  llvm::SmallVector<OpAsmParser::Argument> inputArguments;
  llvm::SmallVector<Type> inputTypes;
  ArrayAttr inputNames;
  if (parseInitializerList(parser, inputArguments, inputOperands, inputTypes,
                           inputNames))
    return failure();
  result.addAttribute("inputNames", inputNames);

  Type i1 = parser.getBuilder().getI1Type();

  OpAsmParser::UnresolvedOperand stallOperand, clockOperand, resetOperand,
      goOperand;

  // Parse optional 'stall (%stallArg)'
  bool withStall = false;
  if (succeeded(parser.parseOptionalKeyword("stall"))) {
    if (parser.parseLParen() || parser.parseOperand(stallOperand) ||
        parser.parseRParen())
      return failure();
    withStall = true;
  }

  // Parse clock, reset, and go.
  if (parseKeywordAndOperand(parser, "clock", clockOperand))
    return failure();

  // Parse optional 'reset (%resetArg)'
  bool withReset = false;
  if (succeeded(parser.parseOptionalKeyword("reset"))) {
    if (parser.parseLParen() || parser.parseOperand(resetOperand) ||
        parser.parseRParen())
      return failure();
    withReset = true;
  }

  if (parseKeywordAndOperand(parser, "go", goOperand))
    return failure();

  // Parse entry stage enable block argument.
  OpAsmParser::Argument entryEnable;
  entryEnable.type = i1;
  if (parser.parseKeyword("entryEn") || parser.parseLParen() ||
      parser.parseArgument(entryEnable) || parser.parseRParen())
    return failure();

  // Optional attribute dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the output assignment list
  if (parser.parseArrow())
    return failure();

  llvm::SmallVector<Type> outputTypes;
  ArrayAttr outputNames;
  if (parseOutputList(parser, outputTypes, outputNames))
    return failure();
  result.addTypes(outputTypes);
  result.addAttribute("outputNames", outputNames);

  // And the implicit 'done' output.
  result.addTypes({i1});

  // All operands have been parsed - resolve.
  if (parser.resolveOperands(inputOperands, inputTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  if (withStall) {
    if (parser.resolveOperand(stallOperand, i1, result.operands))
      return failure();
  }

  Type clkType = seq::ClockType::get(parser.getContext());
  if (parser.resolveOperand(clockOperand, clkType, result.operands))
    return failure();

  if (withReset && parser.resolveOperand(resetOperand, i1, result.operands))
    return failure();

  if (parser.resolveOperand(goOperand, i1, result.operands))
    return failure();

  // Assemble the body region block arguments - this is where the magic happens
  // and why we're doing a custom printer/parser - if the user had to magically
  // know the order of these block arguments, we're asking for issues.
  SmallVector<OpAsmParser::Argument> regionArgs;

  // First we add the input arguments.
  llvm::append_range(regionArgs, inputArguments);
  // Then the internal entry stage enable block argument.
  regionArgs.push_back(entryEnable);

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(inputTypes.size()),
                           static_cast<int32_t>(withStall ? 1 : 0),
                           /*clock*/ static_cast<int32_t>(1),
                           /*reset*/ static_cast<int32_t>(withReset ? 1 : 0),
                           /*go*/ static_cast<int32_t>(1)}));

  return success();
}

static void printKeywordOperand(OpAsmPrinter &p, StringRef keyword,
                                Value value) {
  p << keyword << "(";
  p.printOperand(value);
  p << ")";
}

template <typename TPipelineOp>
static void printPipelineOp(OpAsmPrinter &p, TPipelineOp op) {
  if (auto name = op.getNameAttr()) {
    p << " \"" << name.getValue() << "\"";
  }

  // Print the input list.
  printInitializerList(p, op.getInputs(), op.getInnerInputs());
  p << " ";

  // Print the optional stall.
  if (op.hasStall()) {
    printKeywordOperand(p, "stall", op.getStall());
    p << " ";
  }

  // Print the clock, reset, and go.
  printKeywordOperand(p, "clock", op.getClock());
  p << " ";
  if (op.hasReset()) {
    printKeywordOperand(p, "reset", op.getReset());
    p << " ";
  }
  printKeywordOperand(p, "go", op.getGo());
  p << " ";

  // Print the entry enable block argument.
  p << "entryEn(";
  p.printRegionArgument(
      cast<BlockArgument>(op.getStageEnableSignal(static_cast<size_t>(0))), {},
      /*omitType*/ true);
  p << ") ";

  // Print the optional attribute dict.
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"name", "operandSegmentSizes",
                                           "outputNames", "inputNames"});
  p << " -> ";

  // Print the output list.
  printOutputList(p, op.getDataOutputs().getTypes(), op.getOutputNames());

  p << " ";

  // Print the inner region, eliding the entry block arguments - we've already
  // defined these in our initializer lists.
  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// UnscheduledPipelineOp
//===----------------------------------------------------------------------===//

static void buildPipelineLikeOp(OpBuilder &odsBuilder, OperationState &odsState,
                                TypeRange dataOutputs, ValueRange inputs,
                                ArrayAttr inputNames, ArrayAttr outputNames,
                                Value clock, Value go, Value reset, Value stall,
                                StringAttr name, ArrayAttr stallability) {
  odsState.addOperands(inputs);
  if (stall)
    odsState.addOperands(stall);
  odsState.addOperands(clock);
  odsState.addOperands(reset);
  odsState.addOperands(go);
  if (name)
    odsState.addAttribute("name", name);

  odsState.addAttribute(
      "operandSegmentSizes",
      odsBuilder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(inputs.size()),
           static_cast<int32_t>(stall ? 1 : 0), static_cast<int32_t>(1),
           static_cast<int32_t>(1), static_cast<int32_t>(1)}));

  odsState.addAttribute("inputNames", inputNames);
  odsState.addAttribute("outputNames", outputNames);

  auto *region = odsState.addRegion();
  odsState.addTypes(dataOutputs);

  // Add the implicit done output signal.
  Type i1 = odsBuilder.getIntegerType(1);
  odsState.addTypes({i1});

  // Add the entry stage - arguments order:
  // 1. Inputs
  // 2. Stall (opt)
  // 3. Clock
  // 4. Reset
  // 5. Go
  auto &entryBlock = region->emplaceBlock();
  llvm::SmallVector<Location> entryArgLocs(inputs.size(), odsState.location);
  entryBlock.addArguments(
      inputs.getTypes(),
      llvm::SmallVector<Location>(inputs.size(), odsState.location));
  if (stall)
    entryBlock.addArgument(i1, odsState.location);
  entryBlock.addArgument(i1, odsState.location);
  entryBlock.addArgument(i1, odsState.location);

  // entry stage valid signal.
  entryBlock.addArgument(i1, odsState.location);

  if (stallability)
    odsState.addAttribute("stallability", stallability);
}

template <typename TPipelineOp>
static void getPipelineAsmResultNames(TPipelineOp op,
                                      OpAsmSetValueNameFn setNameFn) {
  for (auto [res, name] :
       llvm::zip(op.getDataOutputs(),
                 op.getOutputNames().template getAsValueRange<StringAttr>()))
    setNameFn(res, name);
  setNameFn(op.getDone(), "done");
}

template <typename TPipelineOp>
static void
getPipelineAsmBlockArgumentNames(TPipelineOp op, mlir::Region &region,
                                 mlir::OpAsmSetValueNameFn setNameFn) {
  for (auto [i, block] : llvm::enumerate(op.getRegion())) {
    if (Block *predBlock = block.getSinglePredecessor()) {
      // Predecessor stageOp might have register and passthrough names
      // specified, which we can use to name the block arguments.
      auto predStageOp = cast<StageOp>(predBlock->getTerminator());
      size_t nRegs = predStageOp.getRegisters().size();
      auto nPassthrough = predStageOp.getPassthroughs().size();

      auto regNames = predStageOp.getRegisterNames();
      auto passthroughNames = predStageOp.getPassthroughNames();

      // Register naming...
      for (size_t regI = 0; regI < nRegs; ++regI) {
        auto arg = block.getArguments()[regI];

        if (regNames) {
          auto nameAttr = dyn_cast<StringAttr>((*regNames)[regI]);
          if (nameAttr && !nameAttr.strref().empty()) {
            setNameFn(arg, nameAttr);
            continue;
          }
        }
        setNameFn(arg, llvm::formatv("s{0}_reg{1}", i, regI).str());
      }

      // Passthrough naming...
      for (size_t passthroughI = 0; passthroughI < nPassthrough;
           ++passthroughI) {
        auto arg = block.getArguments()[nRegs + passthroughI];

        if (passthroughNames) {
          auto nameAttr =
              dyn_cast<StringAttr>((*passthroughNames)[passthroughI]);
          if (nameAttr && !nameAttr.strref().empty()) {
            setNameFn(arg, nameAttr);
            continue;
          }
        }
        setNameFn(arg, llvm::formatv("s{0}_pass{1}", i, passthroughI).str());
      }
    } else {
      // This is the entry stage - name the arguments according to the input
      // names.
      for (auto [inputArg, inputName] :
           llvm::zip(op.getInnerInputs(),
                     op.getInputNames().template getAsValueRange<StringAttr>()))
        setNameFn(inputArg, inputName);
    }

    // Last argument in any stage is the stage enable signal.
    setNameFn(block.getArguments().back(),
              llvm::formatv("s{0}_enable", i).str());
  }
}

void UnscheduledPipelineOp::print(OpAsmPrinter &p) {
  printPipelineOp(p, *this);
}

ParseResult UnscheduledPipelineOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parsePipelineOp(parser, result);
}

void UnscheduledPipelineOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getPipelineAsmResultNames(*this, setNameFn);
}

void UnscheduledPipelineOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  getPipelineAsmBlockArgumentNames(*this, region, setNameFn);
}

void UnscheduledPipelineOp::build(OpBuilder &odsBuilder,
                                  OperationState &odsState,
                                  TypeRange dataOutputs, ValueRange inputs,
                                  ArrayAttr inputNames, ArrayAttr outputNames,
                                  Value clock, Value go, Value reset,
                                  Value stall, StringAttr name,
                                  ArrayAttr stallability) {
  buildPipelineLikeOp(odsBuilder, odsState, dataOutputs, inputs, inputNames,
                      outputNames, clock, go, reset, stall, name, stallability);
}

//===----------------------------------------------------------------------===//
// ScheduledPipelineOp
//===----------------------------------------------------------------------===//

void ScheduledPipelineOp::print(OpAsmPrinter &p) { printPipelineOp(p, *this); }

ParseResult ScheduledPipelineOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parsePipelineOp(parser, result);
}

void ScheduledPipelineOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TypeRange dataOutputs, ValueRange inputs,
                                ArrayAttr inputNames, ArrayAttr outputNames,
                                Value clock, Value go, Value reset, Value stall,
                                StringAttr name, ArrayAttr stallability) {
  buildPipelineLikeOp(odsBuilder, odsState, dataOutputs, inputs, inputNames,
                      outputNames, clock, go, reset, stall, name, stallability);
}

Block *ScheduledPipelineOp::addStage() {
  OpBuilder builder(getContext());
  Block *stage = builder.createBlock(&getRegion());

  // Add the stage valid signal.
  stage->addArgument(builder.getIntegerType(1), getLoc());
  return stage;
}

void ScheduledPipelineOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  getPipelineAsmBlockArgumentNames(*this, region, setNameFn);
}

void ScheduledPipelineOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getPipelineAsmResultNames(*this, setNameFn);
}

// Implementation of getOrderedStages which also produces an error if
// there are any cfg cycles in the pipeline.
static FailureOr<llvm::SmallVector<Block *>>
getOrderedStagesFailable(ScheduledPipelineOp op) {
  llvm::DenseSet<Block *> visited;
  Block *currentStage = op.getEntryStage();
  llvm::SmallVector<Block *> orderedStages;
  do {
    if (!visited.insert(currentStage).second)
      return op.emitOpError("pipeline contains a cycle.");

    orderedStages.push_back(currentStage);
    if (auto stageOp = dyn_cast<StageOp>(currentStage->getTerminator()))
      currentStage = stageOp.getNextStage();
    else
      currentStage = nullptr;
  } while (currentStage);

  return {orderedStages};
}

llvm::SmallVector<Block *> ScheduledPipelineOp::getOrderedStages() {
  // Should always be safe, seeing as the pipeline itself has already been
  // verified.
  return *getOrderedStagesFailable(*this);
}

llvm::DenseMap<Block *, unsigned> ScheduledPipelineOp::getStageMap() {
  llvm::DenseMap<Block *, unsigned> stageMap;
  auto orderedStages = getOrderedStages();
  for (auto [index, stage] : llvm::enumerate(orderedStages))
    stageMap[stage] = index;

  return stageMap;
}

Block *ScheduledPipelineOp::getLastStage() { return getOrderedStages().back(); }

bool ScheduledPipelineOp::isMaterialized() {
  // We determine materialization as if any pipeline stage has an explicit
  // input (apart from the stage enable signal).
  return llvm::any_of(getStages(), [this](Block &block) {
    // The entry stage doesn't count since it'll always have arguments.
    if (&block == getEntryStage())
      return false;
    return block.getNumArguments() > 1;
  });
}

// Check whether the value referenced by `use` is defined within the provided
// `stage`. It is assumed that `use` originates from within `stage`.
static bool useDefinedInStage(Block *stage, OpOperand &use) {
  Block *useBlock = use.getOwner()->getBlock();
  Block *definingBlock = use.get().getParentBlock();

  if (useBlock == definingBlock)
    return true;

  if (stage == definingBlock)
    return true;

  Block *currBlock = definingBlock;
  while (currBlock) {
    currBlock = currBlock->getParentOp()->getBlock();
    if (currBlock == stage)
      return true;
  }

  return false;
}

LogicalResult ScheduledPipelineOp::verify() {
  // Verify that all block are terminated properly.
  auto &stages = getStages();
  for (Block &stage : stages) {
    if (stage.empty() || !isa<ReturnOp, StageOp>(stage.back()))
      return emitOpError("all blocks must be terminated with a "
                         "`pipeline.stage` or `pipeline.return` op.");
  }

  if (failed(getOrderedStagesFailable(*this)))
    return failure();

  // Verify that every stage has a stage valid block argument.
  for (auto [i, block] : llvm::enumerate(stages)) {
    bool err = true;
    if (block.getNumArguments() != 0) {
      auto lastArgType =
          dyn_cast<IntegerType>(block.getArguments().back().getType());
      err = !lastArgType || lastArgType.getWidth() != 1;
    }
    if (err)
      return emitOpError("block " + std::to_string(i) +
                         " must have an i1 argument as the last block argument "
                         "(stage valid signal).");
  }

  // Cache external inputs in a set for fast lookup (also includes clock, reset,
  // and stall).
  llvm::DenseSet<Value> extLikeInputs;
  for (auto extInput : getExtInputs())
    extLikeInputs.insert(extInput);

  extLikeInputs.insert(getClock());
  extLikeInputs.insert(getReset());
  if (hasStall())
    extLikeInputs.insert(getStall());

  // Phase invariant - Check that all values used within a stage are valid
  // based on the materialization mode. This is a walk, since this condition
  // should also apply to nested operations.
  bool materialized = isMaterialized();
  for (auto &stage : stages) {
    auto walkRes = stage.walk([&](Operation *op) {
      // Skip pipeline.src operations in non-materialized mode
      if (isa<SourceOp>(op)) {
        if (materialized) {
          op->emitOpError(
              "Pipeline is in register materialized mode - pipeline.src "
              "operations are not allowed");
          return WalkResult::interrupt();
        }

        // In non-materialized mode, pipeline.src operations are required, and
        // is what is implicitly allowing cross-stage referenced by not
        // reaching the below verification code.
        return WalkResult::advance();
      }

      for (auto [index, operand] : llvm::enumerate(op->getOpOperands())) {
        // External inputs (including clock, reset, stall) are allowed
        // everywhere
        if (extLikeInputs.contains(operand.get()))
          continue;

        // Constant-like inputs are allowed everywhere
        if (auto *definingOp = operand.get().getDefiningOp()) {
          // Constants are allowed to be used across stages.
          if (definingOp->hasTrait<OpTrait::ConstantLike>())
            continue;
        }

        // In any materialization mode, values must be defined in the same
        // stage.
        if (!useDefinedInStage(&stage, operand)) {
          auto err = op->emitOpError("operand ")
                     << index << " is defined in a different stage. ";
          if (materialized) {
            err << "Value should have been passed through block arguments";
          } else {
            err << "Value should have been passed through a `pipeline.src` "
                   "op";
          }
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (walkRes.wasInterrupted())
      return failure();
  }

  if (auto stallability = getStallability()) {
    // Only allow specifying stallability if there is a stall signal.
    if (!hasStall())
      return emitOpError("cannot specify stallability without a stall signal.");

    // Ensure that the # of stages is equal to the length of the stallability
    // array - the exit stage is never stallable.
    size_t nRegisterStages = stages.size() - 1;
    if (stallability->size() != nRegisterStages)
      return emitOpError("stallability array must be the same length as the "
                         "number of stages. Pipeline has ")
             << nRegisterStages << " stages but array had "
             << stallability->size() << " elements.";
  }

  return success();
}

StageKind ScheduledPipelineOp::getStageKind(size_t stageIndex) {
  assert(stageIndex < getNumStages() && "invalid stage index");

  if (!hasStall())
    return StageKind::Continuous;

  // There is a stall signal - also check whether stage-level stallability is
  // specified.
  std::optional<ArrayAttr> stallability = getStallability();
  if (!stallability) {
    // All stages are stallable.
    return StageKind::Stallable;
  }

  if (stageIndex < stallability->size()) {
    bool stageIsStallable =
        cast<BoolAttr>((*stallability)[stageIndex]).getValue();
    if (!stageIsStallable) {
      // This is a non-stallable stage.
      return StageKind::NonStallable;
    }
  }

  // Walk backwards from this stage to see if any non-stallable stage exists.
  // If so, this is a runoff stage.
  // TODO: This should be a pre-computed property.
  if (stageIndex == 0)
    return StageKind::Stallable;

  for (size_t i = stageIndex - 1; i > 0; --i) {
    if (getStageKind(i) == StageKind::NonStallable)
      return StageKind::Runoff;
  }
  return StageKind::Stallable;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  size_t nInputs = getInputs().size();
  auto expectedResults = TypeRange(parent->getResultTypes()).drop_back();
  size_t expectedNResults = expectedResults.size();
  if (nInputs != expectedNResults)
    return emitOpError("expected ")
           << expectedNResults << " return values, got " << nInputs << ".";

  for (auto [inType, reqType] :
       llvm::zip(getInputs().getTypes(), expectedResults)) {
    if (inType != reqType)
      return emitOpError("expected return value of type ")
             << reqType << ", got " << inType << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

// Parses the form:
// ($name `=`)? $register : type($register)

static ParseResult
parseOptNamedTypedAssignment(OpAsmParser &parser,
                             OpAsmParser::UnresolvedOperand &v, Type &t,
                             StringAttr &name) {
  // Parse optional name.
  std::string nameref;
  if (succeeded(parser.parseOptionalString(&nameref))) {
    if (nameref.empty())
      return parser.emitError(parser.getCurrentLocation(),
                              "name cannot be empty");

    if (failed(parser.parseEqual()))
      return parser.emitError(parser.getCurrentLocation(),
                              "expected '=' after name");
    name = parser.getBuilder().getStringAttr(nameref);
  } else {
    name = parser.getBuilder().getStringAttr("");
  }

  // Parse mandatory value and type.
  if (failed(parser.parseOperand(v)) || failed(parser.parseColonType(t)))
    return failure();

  return success();
}

// Parses the form:
// parseOptNamedTypedAssignment (`gated by` `[` $clockGates `]`)?
static ParseResult parseSingleStageRegister(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &v, Type &t,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand> &clockGates,
    StringAttr &name) {
  if (failed(parseOptNamedTypedAssignment(parser, v, t, name)))
    return failure();

  // Parse optional gated-by clause.
  if (failed(parser.parseOptionalKeyword("gated")))
    return success();

  if (failed(parser.parseKeyword("by")) ||
      failed(
          parser.parseOperandList(clockGates, OpAsmParser::Delimiter::Square)))
    return failure();

  return success();
}

// Parses the form:
// regs( ($name `=`)? $register : type($register) (`gated by` `[` $clockGates
// `]`)?, ...)
ParseResult parseStageRegisters(
    OpAsmParser &parser,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> &registers,
    llvm::SmallVector<mlir::Type, 1> &registerTypes,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> &clockGates,
    ArrayAttr &clockGatesPerRegister, ArrayAttr &registerNames) {

  if (failed(parser.parseOptionalKeyword("regs"))) {
    clockGatesPerRegister = parser.getBuilder().getI64ArrayAttr({});
    return success(); // no registers to parse.
  }

  llvm::SmallVector<int64_t> clockGatesPerRegisterList;
  llvm::SmallVector<Attribute> registerNamesList;
  bool withNames = false;
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
        OpAsmParser::UnresolvedOperand v;
        Type t;
        llvm::SmallVector<OpAsmParser::UnresolvedOperand> cgs;
        StringAttr name;
        if (parseSingleStageRegister(parser, v, t, cgs, name))
          return failure();
        registers.push_back(v);
        registerTypes.push_back(t);
        registerNamesList.push_back(name);
        withNames |= static_cast<bool>(name);
        llvm::append_range(clockGates, cgs);
        clockGatesPerRegisterList.push_back(cgs.size());
        return success();
      })))
    return failure();

  clockGatesPerRegister =
      parser.getBuilder().getI64ArrayAttr(clockGatesPerRegisterList);
  if (withNames)
    registerNames = parser.getBuilder().getArrayAttr(registerNamesList);

  return success();
}

void printStageRegisters(OpAsmPrinter &p, Operation *op, ValueRange registers,
                         TypeRange registerTypes, ValueRange clockGates,
                         ArrayAttr clockGatesPerRegister, ArrayAttr names) {
  if (registers.empty())
    return;

  p << "regs(";
  size_t clockGateStartIdx = 0;
  llvm::interleaveComma(
      llvm::enumerate(
          llvm::zip(registers, registerTypes, clockGatesPerRegister)),
      p, [&](auto it) {
        size_t idx = it.index();
        auto &[reg, type, nClockGatesAttr] = it.value();
        if (names) {
          if (auto nameAttr = dyn_cast<StringAttr>(names[idx]);
              nameAttr && !nameAttr.strref().empty())
            p << nameAttr << " = ";
        }

        p << reg << " : " << type;
        int64_t nClockGates = cast<IntegerAttr>(nClockGatesAttr).getInt();
        if (nClockGates == 0)
          return;
        p << " gated by [";
        llvm::interleaveComma(clockGates.slice(clockGateStartIdx, nClockGates),
                              p);
        p << "]";
        clockGateStartIdx += nClockGates;
      });
  p << ")";
}

void printPassthroughs(OpAsmPrinter &p, Operation *op, ValueRange passthroughs,
                       TypeRange passthroughTypes, ArrayAttr names) {

  if (passthroughs.empty())
    return;

  p << "pass(";
  llvm::interleaveComma(
      llvm::enumerate(llvm::zip(passthroughs, passthroughTypes)), p,
      [&](auto it) {
        size_t idx = it.index();
        auto &[reg, type] = it.value();
        if (names) {
          if (auto nameAttr = dyn_cast<StringAttr>(names[idx]);
              nameAttr && !nameAttr.strref().empty())
            p << nameAttr << " = ";
        }
        p << reg << " : " << type;
      });
  p << ")";
}

// Parses the form:
// (`pass` `(` ($name `=`)? $register : type($register), ... `)` )?
ParseResult parsePassthroughs(
    OpAsmParser &parser,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> &passthroughs,
    llvm::SmallVector<mlir::Type, 1> &passthroughTypes,
    ArrayAttr &passthroughNames) {
  if (failed(parser.parseOptionalKeyword("pass")))
    return success(); // no passthroughs to parse.

  llvm::SmallVector<Attribute> passthroughsNameList;
  bool withNames = false;
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
        OpAsmParser::UnresolvedOperand v;
        Type t;
        StringAttr name;
        if (parseOptNamedTypedAssignment(parser, v, t, name))
          return failure();
        passthroughs.push_back(v);
        passthroughTypes.push_back(t);
        passthroughsNameList.push_back(name);
        withNames |= static_cast<bool>(name);
        return success();
      })))
    return failure();

  if (withNames)
    passthroughNames = parser.getBuilder().getArrayAttr(passthroughsNameList);

  return success();
}

void StageOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    Block *dest, ValueRange registers,
                    ValueRange passthroughs) {
  odsState.addSuccessors(dest);
  odsState.addOperands(registers);
  odsState.addOperands(passthroughs);
  odsState.addAttribute("operandSegmentSizes",
                        odsBuilder.getDenseI32ArrayAttr(
                            {static_cast<int32_t>(registers.size()),
                             static_cast<int32_t>(passthroughs.size()),
                             /*clock gates*/ static_cast<int32_t>(0)}));
  llvm::SmallVector<int64_t> clockGatesPerRegister(registers.size(), 0);
  odsState.addAttribute("clockGatesPerRegister",
                        odsBuilder.getI64ArrayAttr(clockGatesPerRegister));
}

void StageOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    Block *dest, ValueRange registers, ValueRange passthroughs,
                    llvm::ArrayRef<llvm::SmallVector<Value>> clockGateList,
                    mlir::ArrayAttr registerNames,
                    mlir::ArrayAttr passthroughNames) {
  build(odsBuilder, odsState, dest, registers, passthroughs);

  llvm::SmallVector<Value> clockGates;
  llvm::SmallVector<int64_t> clockGatesPerRegister(registers.size(), 0);
  for (auto gates : clockGateList) {
    llvm::append_range(clockGates, gates);
    clockGatesPerRegister.push_back(gates.size());
  }
  odsState.attributes.set("clockGatesPerRegister",
                          odsBuilder.getI64ArrayAttr(clockGatesPerRegister));
  odsState.addOperands(clockGates);

  if (registerNames)
    odsState.addAttribute("registerNames", registerNames);

  if (passthroughNames)
    odsState.addAttribute("passthroughNames", passthroughNames);
}

ValueRange StageOp::getClockGatesForReg(unsigned regIdx) {
  assert(regIdx < getRegisters().size() && "register index out of bounds.");

  // TODO: This could be optimized quite a bit if we didn't store clock
  // gates per register as an array of sizes... look into using properties
  // and maybe attaching a more complex datastructure to reduce compute
  // here.

  unsigned clockGateStartIdx = 0;
  for (auto [index, nClockGatesAttr] :
       llvm::enumerate(getClockGatesPerRegister().getAsRange<IntegerAttr>())) {
    int64_t nClockGates = nClockGatesAttr.getInt();
    if (index == regIdx) {
      // This is the register we are looking for.
      return getClockGates().slice(clockGateStartIdx, nClockGates);
    }
    // Increment the start index by the number of clock gates for this
    // register.
    clockGateStartIdx += nClockGates;
  }

  llvm_unreachable("register index out of bounds.");
}

LogicalResult StageOp::verify() {
  // Verify that the target block has the correct arguments as this stage
  // op.
  llvm::SmallVector<Type> expectedTargetArgTypes;
  llvm::append_range(expectedTargetArgTypes, getRegisters().getTypes());
  llvm::append_range(expectedTargetArgTypes, getPassthroughs().getTypes());
  Block *targetStage = getNextStage();
  // Expected types is everything but the stage valid signal.
  TypeRange targetStageArgTypes =
      TypeRange(targetStage->getArgumentTypes()).drop_back();

  if (targetStageArgTypes.size() != expectedTargetArgTypes.size())
    return emitOpError("expected ") << expectedTargetArgTypes.size()
                                    << " arguments in the target stage, got "
                                    << targetStageArgTypes.size() << ".";

  for (auto [index, it] : llvm::enumerate(
           llvm::zip(expectedTargetArgTypes, targetStageArgTypes))) {
    auto [arg, barg] = it;
    if (arg != barg)
      return emitOpError("expected target stage argument ")
             << index << " to have type " << arg << ", got " << barg << ".";
  }

  // Verify that the clock gate index list is equally sized to the # of
  // registers.
  if (getClockGatesPerRegister().size() != getRegisters().size())
    return emitOpError("expected clockGatesPerRegister to be equally sized to "
                       "the number of registers.");

  // Verify that, if provided, the list of register names is equally sized
  // to the number of registers.
  if (auto regNames = getRegisterNames()) {
    if (regNames->size() != getRegisters().size())
      return emitOpError("expected registerNames to be equally sized to "
                         "the number of registers.");
  }

  // Verify that, if provided, the list of passthrough names is equally sized
  // to the number of passthroughs.
  if (auto passthroughNames = getPassthroughNames()) {
    if (passthroughNames->size() != getPassthroughs().size())
      return emitOpError("expected passthroughNames to be equally sized to "
                         "the number of passthroughs.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LatencyOp
//===----------------------------------------------------------------------===//

LogicalResult LatencyOp::verify() {
  ScheduledPipelineOp scheduledPipelineParent =
      dyn_cast<ScheduledPipelineOp>(getOperation()->getParentOp());

  if (!scheduledPipelineParent) {
    // Nothing to verify, got to assume that anything goes in an unscheduled
    // pipeline.
    return success();
  }

  // Verify that there's at least one result type. Latency ops don't make
  // sense if they're not delaying anything, and we're not yet prepared to
  // support side-effectful bodies.
  if (getNumResults() == 0)
    return emitOpError("expected at least one result type.");

  // Verify that the resulting values aren't referenced before they are
  // accessible.
  size_t latency = getLatency();
  Block *definingStage = getOperation()->getBlock();

  llvm::DenseMap<Block *, unsigned> stageMap =
      scheduledPipelineParent.getStageMap();

  auto stageDistance = [&](Block *from, Block *to) {
    assert(stageMap.count(from) && "stage 'from' not contained in pipeline");
    assert(stageMap.count(to) && "stage 'to' not contained in pipeline");
    int64_t fromStage = stageMap[from];
    int64_t toStage = stageMap[to];
    return toStage - fromStage;
  };

  for (auto [i, res] : llvm::enumerate(getResults())) {
    for (auto &use : res.getUses()) {
      auto *user = use.getOwner();

      // The user may reside within a block which is not a stage (e.g.
      // inside a pipeline.latency op). Determine the stage which this use
      // resides within.
      Block *userStage =
          getParentStageInPipeline(scheduledPipelineParent, user);
      unsigned useDistance = stageDistance(definingStage, userStage);

      // Is this a stage op and is the value passed through? if so, this is
      // a legal use.
      StageOp stageOp = dyn_cast<StageOp>(user);
      if (userStage == definingStage && stageOp) {
        if (llvm::is_contained(stageOp.getPassthroughs(), res))
          continue;
      }

      // The use is not a passthrough. Check that the distance between
      // the defining stage and the user stage is at least the latency of
      // the result.
      if (useDistance < latency) {
        auto diag = emitOpError("result ")
                    << i << " is used before it is available.";
        diag.attachNote(user->getLoc())
            << "use was operand " << use.getOperandNumber()
            << ". The result is available " << latency - useDistance
            << " stages later than this use.";
        return diag;
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LatencyReturnOp
//===----------------------------------------------------------------------===//

LogicalResult LatencyReturnOp::verify() {
  LatencyOp parent = cast<LatencyOp>(getOperation()->getParentOp());
  size_t nInputs = getInputs().size();
  size_t nResults = parent->getNumResults();
  if (nInputs != nResults)
    return emitOpError("expected ")
           << nResults << " return values, got " << nInputs << ".";

  for (auto [inType, reqType] :
       llvm::zip(getInputs().getTypes(), parent->getResultTypes())) {
    if (inType != reqType)
      return emitOpError("expected return value of type ")
             << reqType << ", got " << inType << ".";
  }

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"

void PipelineDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"
      >();
}
