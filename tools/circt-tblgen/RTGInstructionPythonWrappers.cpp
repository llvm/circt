//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements generation of Python instruction wrappers from RTG
// instruction definitions. It generates Python functions decorated with
// @instruction that wrap MLIR operations.
//
//===----------------------------------------------------------------------===//

#include "RTGInstructionUtils.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;
using namespace circt::tblgen::rtg;

//===----------------------------------------------------------------------===//
// CLI Arguments
//===----------------------------------------------------------------------===//

static cl::OptionCategory
    rtgPythonWrapperCat("Options for -gen-rtg-instruction-python-wrappers");

static cl::opt<std::string> returnValFunc(
    "return-val-func",
    cl::desc("Return value function name for @instruction decorator"),
    cl::cat(rtgPythonWrapperCat));

static cl::opt<std::string>
    headerFile("header-file",
               cl::desc("Path to file containing header to prepend to output"),
               cl::cat(rtgPythonWrapperCat));

static cl::opt<std::string>
    extension("extension",
              cl::desc("Only generate wrappers for instructions with "
                       "matching extension field"),
              cl::cat(rtgPythonWrapperCat));

static cl::list<int> xlen("xlen",
                          cl::desc("Only generate wrappers for instructions "
                                   "valid on all specified xlens"),
                          cl::cat(rtgPythonWrapperCat), cl::CommaSeparated);

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {
enum class SideEffect {
  READ,
  WRITE,
  READ_WRITE,
};
} // namespace

static SideEffect getSideEffectForOperand(const Operator &op, unsigned argIndex,
                                          const OperandTypeSet &kinds) {
  auto arg = op.getArg(argIndex);
  auto *operand = dyn_cast<NamedTypeConstraint *>(arg);
  if (!operand)
    PrintFatalError(op.getLoc(), "Expected operand to be a type constraint");

  if (kinds.contains<Register>()) {
    bool hasSourceReg = isSourceRegister(op, argIndex);
    bool hasDestReg = isDestinationRegister(op, argIndex);
    if (hasSourceReg == hasDestReg)
      return SideEffect::READ_WRITE;
    if (hasDestReg)
      return SideEffect::WRITE;
    if (hasSourceReg)
      return SideEffect::READ;
  } else if (kinds.contains<Immediate, AnyImmediate, Label>()) {
    return SideEffect::READ;
  } else if (kinds.contains<Memory, AnyMemory>()) {
    return SideEffect::READ_WRITE;
  }

  PrintFatalError(op.getLoc(),
                  "Unsupported operand type for instruction operand type");
}

static std::string getTypeSuffix(OperandType type) {
  if (std::holds_alternative<Immediate>(type)) {
    auto imm = std::get<Immediate>(type);
    return "_imm" + std::to_string(imm.bitWidth);
  }
  if (std::holds_alternative<Label>(type))
    return "_lbl";
  if (std::holds_alternative<Memory>(type)) {
    auto mem = std::get<Memory>(type);
    return "_mem" + std::to_string(mem.addressWidth);
  }
  if (std::holds_alternative<Register>(type)) {
    auto reg = std::get<Register>(type);
    return "_" + reg.name.lower();
  }

  PrintFatalError("Unsupported type for instruction operand type");
}

static void emitPythonTypeHint(OperandType type, raw_ostream &os) {
  if (std::holds_alternative<Immediate>(type) ||
      std::holds_alternative<AnyImmediate>(type))
    os << "Immediate";
  else if (std::holds_alternative<Label>(type))
    os << "Label";
  else if (std::holds_alternative<Memory>(type) ||
           std::holds_alternative<AnyMemory>(type))
    os << "Memory";
  else if (std::holds_alternative<Register>(type))
    os << std::get<Register>(type).name;
  else
    PrintFatalError("Unsupported type for instruction operand type");
}

static void emitPythonTypeExpr(OperandType type, raw_ostream &os) {
  if (std::holds_alternative<Immediate>(type))
    os << "ImmediateType(" << std::get<Immediate>(type).bitWidth << ")";
  else if (std::holds_alternative<Label>(type))
    os << "LabelType()";
  else if (std::holds_alternative<Memory>(type))
    os << "MemoryType(" << std::get<Memory>(type).addressWidth << ")";
  else if (std::holds_alternative<Register>(type))
    os << std::get<Register>(type).typeName << "()";
  else
    PrintFatalError("Unsupported type for instruction operand type");
}

static void emitSideEffect(SideEffect sideEffect, raw_ostream &os) {
  os << "SideEffect.";
  switch (sideEffect) {
  case SideEffect::READ:
    os << "READ";
    break;
  case SideEffect::WRITE:
    os << "WRITE";
    break;
  case SideEffect::READ_WRITE:
    os << "READ_WRITE";
    break;
  }
}

static void sanitizePythonFunctionName(std::string &name) {
  static const char *pythonKeywords[] = {
      "and", "as", "assert", "async", "await", "break", "class", "continue",
      "def", "del", "elif", "else", "except", "False", "finally", "for", "from",
      "global", "if", "import", "in", "is", "lambda", "None", "nonlocal", "not",
      "or", "pass", "raise", "return", "True", "try", "while", "with", "yield",
      // Also include common built-in names that shouldn't be shadowed
      "callable", "issubclass", "type"};

  if (name.empty())
    PrintFatalError("Empty string is not a valid Python identifier");

  // Replace non-alphanumeric characters (except underscore) with underscores
  for (char &ch : name) {
    if (!llvm::isAlnum(ch) && ch != '_')
      ch = '_';
  }

  // Handle the case where all characters were replaced (result is all
  // underscores)
  if (llvm::all_of(name, [](char ch) { return ch == '_'; }))
    PrintFatalError(
        "String contains no valid characters for Python identifier");

  // Prepend underscore if the name starts with a digit
  if (llvm::isDigit(name[0]))
    name = "_" + name;

  // Append underscore if the name is a Python keyword
  for (const char *keyword : pythonKeywords) {
    if (name == keyword) {
      name.push_back('_');
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Python Code Generation
//===----------------------------------------------------------------------===//

static void emitInstructionDecorator(ArrayRef<OperandType> operandTypes,
                                     ArrayRef<SideEffect> operandSideEffects,
                                     std::optional<StringRef> mnemonic,
                                     std::optional<StringRef> extension,
                                     raw_ostream &os) {
  os << "@instruction(return_val_func=" << returnValFunc << ",\n";
  os << "             args=[";

  llvm::interleaveComma(llvm::zip(operandTypes, operandSideEffects), os,
                        [&](auto tyAndSideEffect) {
                          auto [ty, sideEffect] = tyAndSideEffect;
                          os << "(";
                          emitPythonTypeExpr(ty, os);
                          os << ", ";
                          emitSideEffect(sideEffect, os);
                          os << ")";
                        });

  os << "]";

  if (mnemonic && !mnemonic->empty())
    os << ",\n             mnemonic=\"" << *mnemonic << "\"";

  if (extension && !extension->empty())
    os << ",\n             extension=\"" << *extension << "\"";

  os << ")\n";
}

static std::string getFunctionName(const Operator &op) {
  // TODO: get this from a tablegen field directly instead of inferring it from
  // the op name
  std::string opName = op.getOperationName();
  size_t dotPos = opName.find_last_of('.');
  std::string mnemonic = opName;
  if (dotPos != std::string::npos)
    mnemonic = opName.substr(dotPos + 1);

  return mnemonic;
}

static void emitFunctionSignature(StringRef opMnemonic,
                                  ArrayRef<OperandType> combination,
                                  ArrayRef<size_t> unionOperandIndices,
                                  ArrayRef<StringRef> operandNames,
                                  raw_ostream &os) {
  std::string name = opMnemonic.str();
  for (auto i : unionOperandIndices)
    name += getTypeSuffix(combination[i]);
  sanitizePythonFunctionName(name);

  os << "def " << name << "(";

  llvm::interleaveComma(llvm::zip(operandNames, combination), os,
                        [&](auto nameAndType) {
                          auto [name, ty] = nameAndType;
                          os << name << ": ";
                          emitPythonTypeHint(ty, os);
                        });

  os << "):\n";
}

static void emitFunctionBody(const Operator &op,
                             ArrayRef<OperandType> combination,
                             ArrayRef<StringRef> operandNames,
                             raw_ostream &os) {
  os << "  " << op.getDialectName() << "." << op.getCppClassName() << "(";
  llvm::interleaveComma(operandNames, os);
  os << ")\n";
}

static void
generateTypeCombinations(ArrayRef<OperandTypeSet> operandKinds,
                         SmallVector<SmallVector<OperandType>> &combinations) {
  if (operandKinds.empty()) {
    combinations.push_back({});
    return;
  }

  // Initialize with the first operand's kinds
  for (auto kind : operandKinds[0])
    combinations.push_back({kind});

  // Iteratively build the cartesian product
  for (size_t i = 1; i < operandKinds.size(); ++i) {
    const auto &currentKinds = operandKinds[i];
    assert(!currentKinds.empty() && "expected non-empty operand kinds");

    SmallVector<SmallVector<OperandType>> newCombinations;
    for (const auto &existingCombination : combinations) {
      for (OperandType newKind : currentKinds) {
        SmallVector<OperandType> newCombination = existingCombination;
        newCombination.push_back(newKind);
        newCombinations.push_back(std::move(newCombination));
      }
    }
    combinations = std::move(newCombinations);
  }
}

static void
emitDispatcherSignature(StringRef opMnemonic,
                        ArrayRef<SmallVector<OperandType>> combinations,
                        ArrayRef<StringRef> operandNames, raw_ostream &os) {
  assert(combinations.size() > 0 && "Expected at least one signature option");

  std::string funcName = opMnemonic.str();
  sanitizePythonFunctionName(funcName);
  os << "def " << funcName << "(";

  // Transpose combinations to get all types for each operand position
  SmallVector<OperandTypeSet> operandTypes(combinations[0].size());
  for (const auto &combination : combinations) {
    for (auto [i, type] : llvm::enumerate(combination)) {
      operandTypes[i].appendAndUnique(type);
    }
  }

  auto emitArgument = [&](auto nameAndTypes) {
    auto [name, types] = nameAndTypes;
    os << name << ": ";

    if (types.size() == 1) {
      emitPythonTypeHint(types[0], os);
      return;
    }

    // Generate Union[Type1, Type2, ...]
    os << "Union[";
    llvm::interleaveComma(types, os,
                          [&](auto ty) { emitPythonTypeHint(ty, os); });
    os << "]";
  };
  llvm::interleaveComma(llvm::zip(operandNames, operandTypes), os,
                        emitArgument);

  os << "):\n";
}

static void emitDispatcherBody(StringRef opMnemonic,
                               ArrayRef<SmallVector<OperandType>> combinations,
                               ArrayRef<size_t> unionOperandIndices,
                               ArrayRef<StringRef> operandNames,
                               raw_ostream &os) {
  for (auto [i, combination] : llvm::enumerate(combinations)) {
    if (i == 0)
      os << "  if ";
    else if (i < combinations.size() - 1)
      os << "  elif ";
    else
      os << "  else:\n";

    if (i < combinations.size() - 1) {
      auto comb = combination;
      llvm::interleave(
          unionOperandIndices, os,
          [&](auto opIdx) {
            os << "isinstance(" << operandNames[opIdx] << ", ";
            emitPythonTypeHint(comb[opIdx], os);
            os << ")";
          },
          " and ");
      os << ":\n";
    }

    std::string name = opMnemonic.str();
    for (auto k : unionOperandIndices)
      name += getTypeSuffix(combination[k]);
    sanitizePythonFunctionName(name);

    os << "    return " << name << "(";
    llvm::interleaveComma(operandNames, os);
    os << ")\n";
  }
}

static void genPythonWrapperForOp(const Operator &op, raw_ostream &os) {
  // Skip operations that do not implement InstructionOpInterface
  if (!hasInstructionOpInterface(op))
    return;

  if (!extension.empty()) {
    auto opExtension = op.getDef().getValueAsOptionalString("extension");
    if (!opExtension || *opExtension != extension)
      return;
  }

  // If the xlen filter is enabled, check if the instruction is valid on all
  // specified xlens.
  if (!xlen.empty()) {
    if (!op.getDef().getValue("xlen"))
      return;
    auto opXlenList = op.getDef().getValueAsListOfInts("xlen");
    for (int requiredXlen : xlen) {
      if (llvm::none_of(opXlenList,
                        [&](int64_t val) { return val == requiredXlen; }))
        return;
    }
  }

  SmallVector<OperandTypeSet> operandKinds;
  SmallVector<SideEffect> operandSideEffect;
  SmallVector<StringRef> operandNames;

  for (auto [i, arg] : llvm::enumerate(op.getArgs())) {
    if (auto *operand = dyn_cast<NamedTypeConstraint *>(arg)) {
      auto &kinds = operandKinds.emplace_back();
      classifyOperandType(operand->constraint.getDef(), kinds);
      if (kinds.empty())
        PrintFatalError(op.getLoc(), "failed to classify operand type for '" +
                                         operand->name + "'");
      operandSideEffect.emplace_back(getSideEffectForOperand(op, i, kinds));
      // Operand names are already sanitized or rejected by TableGen
      operandNames.emplace_back(operand->name);
    }
  }

  // Generate all type combinations
  SmallVector<SmallVector<OperandType>> combinations;
  generateTypeCombinations(operandKinds, combinations);

  SmallVector<size_t> unionOperandIndices;
  for (auto [i, kinds] : llvm::enumerate(operandKinds)) {
    if (kinds.size() > 1)
      unionOperandIndices.push_back(i);
  }

  auto isaMnemonic = op.getDef().getValueAsOptionalString("isaMnemonic");
  auto isaExtension = op.getDef().getValueAsOptionalString("extension");

  auto opMnemonic = getFunctionName(op);
  for (const auto &combination : combinations) {
    emitInstructionDecorator(combination, operandSideEffect, isaMnemonic,
                             isaExtension, os);
    emitFunctionSignature(opMnemonic, combination, unionOperandIndices,
                          operandNames, os);
    emitFunctionBody(op, combination, operandNames, os);
    os << "\n";
  }

  if (combinations.size() > 1) {
    emitDispatcherSignature(opMnemonic, combinations, operandNames, os);
    emitDispatcherBody(opMnemonic, combinations, unionOperandIndices,
                       operandNames, os);
    os << "\n";
  }
}

static bool genRTGInstructionPythonWrappers(const RecordKeeper &records,
                                            raw_ostream &os) {
  if (returnValFunc.empty()) {
    llvm::errs() << "error: --return-val-func is required for "
                    "-gen-rtg-instruction-python-wrappers\n";
    return true;
  }

  // Emit header file if specified
  if (!headerFile.empty()) {
    auto fileOrErr = MemoryBuffer::getFile(headerFile);
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "error: Failed to open header file '" << headerFile
                   << "': " << ec.message() << "\n";
      return true;
    }
    os << fileOrErr.get()->getBuffer();
    os << "\n";
  }

  for (const Record *opDef : records.getAllDerivedDefinitions("Op"))
    genPythonWrapperForOp(Operator(opDef), os);

  return false;
}

static mlir::GenRegistration
    genRTGInstructionMethodsReg("gen-rtg-instruction-python-wrappers",
                                "Generate Python wrappers for RTG instructions",
                                genRTGInstructionPythonWrappers);
