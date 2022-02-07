//===- ExportVerilog.cpp - Verilog Emitter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Verilog emitter implementation.
//
// CAREFUL: This file covers the emission phase of `ExportVerilog` which mainly
// walks the IR and produces output. Do NOT modify the IR during this walk, as
// emission occurs in a highly parallel fashion. If you need to modify the IR,
// do so during the preparation phase which lives in `PrepareForEmission.cpp`.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportVerilog.h"
#include "../PassDetail.h"
#include "ExportVerilogInternals.h"
#include "RearrangableOStream.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/Path.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Translation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;

using namespace comb;
using namespace hw;
using namespace sv;
using namespace ExportVerilog;

#define DEBUG_TYPE "export-verilog"

constexpr int INDENT_AMOUNT = 2;
constexpr int SPACE_PER_INDENT_IN_EXPRESSION_FORMATTING = 8;

namespace {
/// This enum keeps track of the precedence level of various binary operators,
/// where a lower number binds tighter.
enum VerilogPrecedence {
  // Normal precedence levels.
  Symbol,          // Atomic symbol like "foo" and {a,b}
  Selection,       // () , [] , :: , ., $signed()
  Unary,           // Unary operators like ~foo
  Multiply,        // * , / , %
  Addition,        // + , -
  Shift,           // << , >>, <<<, >>>
  Comparison,      // > , >= , < , <=
  Equality,        // == , !=
  And,             // &
  Xor,             // ^ , ^~
  Or,              // |
  AndShortCircuit, // &&
  Conditional,     // ? :

  LowestPrecedence,  // Sentinel which is always the lowest precedence.
  ForceEmitMultiUse, // Sentinel saying to recursively emit a multi-used expr.
};

/// This enum keeps track of whether the emitted subexpression is signed or
/// unsigned as seen from the Verilog language perspective.
enum SubExprSignResult { IsSigned, IsUnsigned };

/// This is information precomputed about each subexpression in the tree we
/// are emitting as a unit.
struct SubExprInfo {
  /// The precedence of this expression.
  VerilogPrecedence precedence;

  /// The signedness of the expression.
  SubExprSignResult signedness;

  SubExprInfo(VerilogPrecedence precedence, SubExprSignResult signedness)
      : precedence(precedence), signedness(signedness) {}
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helper routines
//===----------------------------------------------------------------------===//

static Attribute getInt32Attr(MLIRContext *ctx, uint32_t value) {
  return Builder(ctx).getI32IntegerAttr(value);
}

/// Return true for nullary operations that are better emitted multiple
/// times as inline expression (when they have multiple uses) rather than having
/// a temporary wire.
///
/// This can only handle nullary expressions, because we don't want to replicate
/// subtrees arbitrarily.
static bool isDuplicatableNullaryExpression(Operation *op) {
  // We don't want wires that are just constants aesthetically.
  if (isConstantExpression(op))
    return true;

  // If this is a small verbatim expression with no side effects, duplicate it
  // inline.
  if (isa<VerbatimExprOp>(op)) {
    if (op->getNumOperands() == 0 &&
        op->getAttrOfType<StringAttr>("string").getValue().size() <= 32)
      return true;
  }

  return false;
}

// Return true if the expression can be inlined even when the op has multiple
// uses. Be careful to add operations here since it might cause exponential
// emission without proper restrictions.
static bool isDuplicatableExpression(Operation *op) {
  if (op->getNumOperands() == 0)
    return isDuplicatableNullaryExpression(op);

  // It is cheap to inline extract op.
  if (isa<comb::ExtractOp>(op))
    return true;

  // We only inline array_get with a constant index.
  if (auto array = dyn_cast<hw::ArrayGetOp>(op))
    return array.index().getDefiningOp<ConstantOp>();

  return false;
}

/// Return the verilog name of the operations that can define a symbol.
/// Except for <WireOp, RegOp, LocalParamOp, InstanceOp>, check global state
/// `getDeclarationVerilogName` for them.
static StringRef getSymOpName(Operation *symOp) {
  // Typeswitch of operation types which can define a symbol.
  // If legalizeNames has renamed it, then the attribute must be set.
  if (auto attr = symOp->getAttrOfType<StringAttr>("hw.verilogName"))
    return attr.getValue();
  return TypeSwitch<Operation *, StringRef>(symOp)
      .Case<HWModuleOp, HWModuleExternOp, HWModuleGeneratedOp>(
          [](Operation *op) { return getVerilogModuleName(op); })
      .Case<InterfaceOp>([&](InterfaceOp op) {
        return getVerilogModuleNameAttr(op).getValue();
      })
      .Case<InterfaceSignalOp>(
          [&](InterfaceSignalOp op) { return op.sym_name(); })
      .Case<InterfaceModportOp>(
          [&](InterfaceModportOp op) { return op.sym_name(); })
      .Default([&](Operation *op) {
        if (auto attr = op->getAttrOfType<StringAttr>("name"))
          return attr.getValue();
        if (auto attr = op->getAttrOfType<StringAttr>("instanceName"))
          return attr.getValue();
        if (auto attr =
                op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
          return attr.getValue();
        return StringRef("");
      });
}

/// Return the verilog name of the port for the module.
StringRef getPortVerilogName(Operation *module, ssize_t portArgNum) {
  auto numInputs = hw::getModuleNumInputs(module);
  // portArgNum is the index into the result of getAllModulePortInfos.
  // Also ensure the correct index into the input/output list is computed.
  ssize_t portId = portArgNum;
  char verilogNameAttr[] = "hw.verilogName";
  // Check for input ports.
  if (portArgNum < numInputs) {
    if (auto argAttr = module->getAttrOfType<ArrayAttr>(
            mlir::function_like_impl::getArgDictAttrName()))
      if (auto argDict = argAttr[portArgNum].cast<DictionaryAttr>())
        if (auto updatedName = argDict.get(verilogNameAttr))
          return updatedName.cast<StringAttr>().getValue();
    // Get the original name of input port if no renaming.
    return module->getAttrOfType<ArrayAttr>("argNames")[portArgNum]
        .cast<StringAttr>()
        .getValue();
  }

  // If its an output port, get the index into the output port array.
  portId = portArgNum - numInputs;
  if (auto argAttr = module->getAttrOfType<ArrayAttr>(
          mlir::function_like_impl::getResultDictAttrName()))
    if (auto argDict = argAttr[portId].cast<DictionaryAttr>())
      if (auto updatedName = argDict.get(verilogNameAttr))
        return updatedName.cast<StringAttr>().getValue();
  // Get the original name of output port if no renaming.
  return module->getAttrOfType<ArrayAttr>("resultNames")[portId]
      .cast<StringAttr>()
      .getValue();
}

StringRef getPortVerilogName(Operation *module, PortInfo port) {
  return getPortVerilogName(
      module, port.isOutput() ? port.argNum + hw::getModuleNumInputs(module)
                              : port.argNum);
}

/// This predicate returns true if the specified operation is considered a
/// potentially inlinable Verilog expression.  These nodes always have a single
/// result, but may have side effects (e.g. `sv.verbatim.expr.se`).
/// MemoryEffects should be checked if a client cares.
bool ExportVerilog::isVerilogExpression(Operation *op) {
  // These are SV dialect expressions.
  if (isa<ReadInOutOp, ArrayIndexInOutOp, IndexedPartSelectInOutOp,
          StructFieldInOutOp, IndexedPartSelectOp, ParamValueOp, XMROp,
          SampledOp>(op))
    return true;

  // All HW combinational logic ops and SV expression ops are Verilog
  // expressions.
  return isCombinational(op) || isExpression(op);
}

/// Return the width of the specified type in bits or -1 if it isn't
/// supported.
static int getBitWidthOrSentinel(Type type) {
  return TypeSwitch<Type, int>(type)
      .Case<IntegerType>([](IntegerType integerType) {
        // Verilog doesn't support zero bit integers.  We only support them in
        // limited cases.
        return integerType.getWidth();
      })
      .Case<InOutType>([](InOutType inoutType) {
        return getBitWidthOrSentinel(inoutType.getElementType());
      })
      .Case<TypeAliasType>([](TypeAliasType alias) {
        return getBitWidthOrSentinel(alias.getInnerType());
      })
      .Default([](Type) { return -1; });
}

/// Push this type's dimension into a vector.
static void getTypeDims(SmallVectorImpl<Attribute> &dims, Type type,
                        Location loc) {
  if (auto integer = hw::type_dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() != 1)
      dims.push_back(getInt32Attr(type.getContext(), integer.getWidth()));
    return;
  }
  if (auto array = hw::type_dyn_cast<ArrayType>(type)) {
    dims.push_back(getInt32Attr(type.getContext(), array.getSize()));
    getTypeDims(dims, array.getElementType(), loc);

    return;
  }
  if (auto intType = hw::type_dyn_cast<IntType>(type)) {
    dims.push_back(intType.getWidth());
    return;
  }

  if (auto inout = hw::type_dyn_cast<InOutType>(type))
    return getTypeDims(dims, inout.getElementType(), loc);
  if (auto uarray = hw::type_dyn_cast<hw::UnpackedArrayType>(type))
    return getTypeDims(dims, uarray.getElementType(), loc);
  if (hw::type_isa<InterfaceType>(type) || hw::type_isa<StructType>(type))
    return;

  mlir::emitError(loc, "value has an unsupported verilog type ") << type;
}

/// True iff 'a' and 'b' have the same wire dims.
static bool haveMatchingDims(Type a, Type b, Location loc) {
  SmallVector<Attribute, 4> aDims;
  getTypeDims(aDims, a, loc);

  SmallVector<Attribute, 4> bDims;
  getTypeDims(bDims, b, loc);

  return aDims == bDims;
}

/// Return true if this is a zero bit type, e.g. a zero bit integer or array
/// thereof.
static bool isZeroBitType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.getWidth() == 0;
  if (auto inout = type.dyn_cast<hw::InOutType>())
    return isZeroBitType(inout.getElementType());
  if (auto uarray = type.dyn_cast<hw::UnpackedArrayType>())
    return isZeroBitType(uarray.getElementType());
  if (auto array = type.dyn_cast<hw::ArrayType>())
    return isZeroBitType(array.getElementType());
  if (auto structType = type.dyn_cast<hw::StructType>())
    return llvm::all_of(structType.getElements(),
                        [](auto elem) { return isZeroBitType(elem.type); });

  // We have an open type system, so assume it is ok.
  return false;
}

/// Given a set of known nested types (those supported by this pass), strip off
/// leading unpacked types.  This strips off portions of the type that are
/// printed to the right of the name in verilog.
static Type stripUnpackedTypes(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<InOutType>([](InOutType inoutType) {
        return stripUnpackedTypes(inoutType.getElementType());
      })
      .Case<UnpackedArrayType>([](UnpackedArrayType arrayType) {
        return stripUnpackedTypes(arrayType.getElementType());
      })
      .Default([](Type type) { return type; });
}

/// Return true if type has a struct type as a subtype.
static bool hasStructType(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<InOutType, UnpackedArrayType, ArrayType>([](auto parentType) {
        return hasStructType(parentType.getElementType());
      })
      .Case<StructType>([](auto) { return true; })
      .Default([](auto) { return false; });
}

/// Return the word (e.g. "reg") in Verilog to declare the specified thing.
static StringRef getVerilogDeclWord(Operation *op,
                                    const LoweringOptions &options) {
  if (isa<RegOp>(op)) {
    // Check if the type stored in this register is a struct or array of
    // structs. In this case, according to spec section 6.8, the "reg" prefix
    // should be left off.
    auto elementType =
        op->getResult(0).getType().cast<InOutType>().getElementType();
    if (elementType.isa<StructType>())
      return "";
    if (auto innerType = elementType.dyn_cast<ArrayType>()) {
      while (innerType.getElementType().isa<ArrayType>())
        innerType = innerType.getElementType().cast<ArrayType>();
      if (innerType.getElementType().isa<StructType>() ||
          innerType.getElementType().isa<TypeAliasType>())
        return "";
    }
    if (elementType.isa<TypeAliasType>())
      return "";

    return "reg";
  }
  if (isa<WireOp>(op))
    return "wire";
  if (isa<ConstantOp, LocalParamOp, ParamValueOp>(op))
    return "localparam";

  // Interfaces instances use the name of the declared interface.
  if (auto interface = dyn_cast<InterfaceInstanceOp>(op))
    return interface.getInterfaceType().getInterface().getValue();

  // If 'op' is in a module, output 'wire'. If 'op' is in a procedural block,
  // fall through to default.
  bool isProcedural = op->getParentOp()->hasTrait<ProceduralRegion>();

  if (!isProcedural)
    return "wire";

  // "automatic" values aren't allowed in disallowLocalVariables mode.
  assert(!options.disallowLocalVariables && "automatic variables not allowed");

  // If the type contains a struct type, we have to use only "automatic" because
  // "automatic struct" is syntactically correct.
  return hasStructType(op->getResult(0).getType()) ? "automatic"
                                                   : "automatic logic";
}

/// Pull any FileLineCol locs out of the specified location and add it to the
/// specified set.
static void collectFileLineColLocs(Location loc,
                                   SmallPtrSet<Attribute, 8> &locationSet) {
  if (auto fileLoc = loc.dyn_cast<FileLineColLoc>())
    locationSet.insert(fileLoc);

  if (auto fusedLoc = loc.dyn_cast<FusedLoc>())
    for (auto loc : fusedLoc.getLocations())
      collectFileLineColLocs(loc, locationSet);
}

/// Return the location information as a (potentially empty) string.
static std::string
getLocationInfoAsString(const SmallPtrSet<Operation *, 8> &ops) {
  std::string resultStr;
  llvm::raw_string_ostream sstr(resultStr);

  // Multiple operations may come from the same location or may not have useful
  // location info.  Unique it now.
  SmallPtrSet<Attribute, 8> locationSet;
  for (auto *op : ops)
    collectFileLineColLocs(op->getLoc(), locationSet);

  auto printLoc = [&](FileLineColLoc loc) {
    sstr << loc.getFilename().getValue();
    if (auto line = loc.getLine()) {
      sstr << ':' << line;
      if (auto col = loc.getColumn())
        sstr << ':' << col;
    }
  };

  // Fast pass some common cases.
  switch (locationSet.size()) {
  case 1:
    printLoc((*locationSet.begin()).cast<FileLineColLoc>());
    LLVM_FALLTHROUGH;
  case 0:
    return sstr.str();
  default:
    break;
  }

  // Sort the entries.
  SmallVector<FileLineColLoc, 8> locVector;
  locVector.reserve(locationSet.size());
  for (auto loc : locationSet)
    locVector.push_back(loc.cast<FileLineColLoc>());

  llvm::array_pod_sort(
      locVector.begin(), locVector.end(),
      [](const FileLineColLoc *lhs, const FileLineColLoc *rhs) -> int {
        if (auto fn = lhs->getFilename().compare(rhs->getFilename()))
          return fn;
        if (lhs->getLine() != rhs->getLine())
          return lhs->getLine() < rhs->getLine() ? -1 : 1;
        return lhs->getColumn() < rhs->getColumn() ? -1 : 1;
      });

  // The entries are sorted by filename, line, col.  Try to merge together
  // entries to reduce verbosity on the column info.
  StringRef lastFileName;
  for (size_t i = 0, e = locVector.size(); i != e;) {
    if (i != 0)
      sstr << ", ";

    // Print the filename if it changed.
    auto first = locVector[i];
    if (first.getFilename() != lastFileName) {
      lastFileName = first.getFilename();
      sstr << lastFileName;
    }

    // Scan for entries with the same file/line.
    size_t end = i + 1;
    while (end != e && first.getFilename() == locVector[end].getFilename() &&
           first.getLine() == locVector[end].getLine())
      ++end;

    // If we have one entry, print it normally.
    if (end == i + 1) {
      if (auto line = first.getLine()) {
        sstr << ':' << line;
        if (auto col = first.getColumn())
          sstr << ':' << col;
      }
      ++i;
      continue;
    }

    // Otherwise print a brace enclosed list.
    sstr << ':' << first.getLine() << ":{";
    while (i != end) {
      sstr << locVector[i++].getColumn();

      if (i != end)
        sstr << ',';
    }
    sstr << '}';
  }

  return sstr.str();
}

/// Most expressions are invalid to bit-select from in Verilog, but some
/// things are ok.  Return true if it is ok to inline bitselect from the
/// result of this expression.  It is conservatively correct to return false.
static bool isOkToBitSelectFrom(Value v) {
  // Module ports are always ok to bit select from.
  if (v.isa<BlockArgument>())
    return true;

  // Uses of a wire or register can be done inline.
  if (auto read = v.getDefiningOp<ReadInOutOp>()) {
    if (read.input().getDefiningOp<WireOp>() ||
        read.input().getDefiningOp<RegOp>())
      return true;
  }

  // Aggregate access can be inlined.
  if (v.getDefiningOp<StructExtractOp>())
    return true;

  // Interface signal can be inlined.
  if (v.getDefiningOp<ReadInterfaceSignalOp>())
    return true;

  // TODO: We could handle concat and other operators here.
  return false;
}

//===----------------------------------------------------------------------===//
// ModuleNameManager Implementation
//===----------------------------------------------------------------------===//

namespace {
/// This class keeps track of names for values within a module.
struct ModuleNameManager {
  ModuleNameManager() {}

  StringRef addName(Value value, StringRef name) {
    return addName(ValueOrOp(value), name);
  }
  StringRef addName(Operation *op, StringRef name) {
    return addName(ValueOrOp(op), name);
  }
  StringRef addName(Value value, StringAttr name) {
    return addName(ValueOrOp(value), name);
  }
  StringRef addName(Operation *op, StringAttr name) {
    return addName(ValueOrOp(op), name);
  }

  StringRef getName(Value value) { return getName(ValueOrOp(value)); }
  StringRef getName(Operation *op) {
    // If RegOp or WireOp, then result has the name.
    if (isa<sv::WireOp, sv::RegOp>(op))
      return getName(op->getResult(0));
    return getName(ValueOrOp(op));
  }

  bool hasName(Value value) { return nameTable.count(ValueOrOp(value)); }

  bool hasName(Operation *op) {
    // If RegOp or WireOp, then result has the name.
    if (isa<sv::WireOp, sv::RegOp>(op))
      return nameTable.count(op->getResult(0));
    return nameTable.count(ValueOrOp(op));
  }

private:
  using ValueOrOp = PointerUnion<Value, Operation *>;

  /// Retrieve a name from the name table.  The name must already have been
  /// added.
  StringRef getName(ValueOrOp valueOrOp) {
    auto entry = nameTable.find(valueOrOp);
    assert(entry != nameTable.end() &&
           "value expected a name but doesn't have one");
    return entry->getSecond();
  }

  /// Add the specified name to the name table, auto-uniquing the name if
  /// required.  If the name is empty, then this creates a unique temp name.
  ///
  /// "valueOrOp" is typically the Value for an intermediate wire etc, but it
  /// can also be an op for an instance, since we want the instances op uniqued
  /// and tracked.  It can also be null for things like outputs which are not
  /// tracked in the nameTable.
  StringRef addName(ValueOrOp valueOrOp, StringRef name);

  StringRef addName(ValueOrOp valueOrOp, StringAttr nameAttr) {
    return addName(valueOrOp, nameAttr ? nameAttr.getValue() : "");
  }

  /// nameTable keeps track of mappings from Value's and operations (for
  /// instances) to their string table entry.
  llvm::DenseMap<ValueOrOp, StringRef> nameTable;

  NameCollisionResolver nameResolver;
};
} // end anonymous namespace

/// Add the specified name to the name table, auto-uniquing the name if
/// required.  If the name is empty, then this creates a unique temp name.
///
/// "valueOrOp" is typically the Value for an intermediate wire etc, but it
/// can also be an op for an instance, since we want the instances op uniqued
/// and tracked.  It can also be null for things like outputs which are not
/// tracked in the nameTable.
StringRef ModuleNameManager::addName(ValueOrOp valueOrOp, StringRef name) {
  auto updatedName = nameResolver.getLegalName(name);
  if (valueOrOp)
    nameTable[valueOrOp] = updatedName;
  return updatedName;
}

//===----------------------------------------------------------------------===//
// VerilogEmitterState
//===----------------------------------------------------------------------===//

namespace {

/// This class maintains the mutable state that cross-cuts and is shared by the
/// various emitters.
class VerilogEmitterState {
public:
  explicit VerilogEmitterState(ModuleOp designOp,
                               const SharedEmitterState &shared,
                               const LoweringOptions &options,
                               const SymbolCache &symbolCache,
                               const GlobalNameTable &globalNames,
                               raw_ostream &os)
      : designOp(designOp), shared(shared), options(options),
        symbolCache(symbolCache), globalNames(globalNames), os(os) {}

  /// This is the root mlir::ModuleOp that holds the whole design being emitted.
  ModuleOp designOp;

  const SharedEmitterState &shared;

  /// The emitter options which control verilog emission.
  const LoweringOptions &options;

  /// This is a cache of various information about the IR, in frozen state.
  const SymbolCache &symbolCache;

  /// This tracks global names where the Verilog name needs to be different than
  /// the IR name.
  const GlobalNameTable &globalNames;

  /// The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

private:
  VerilogEmitterState(const VerilogEmitterState &) = delete;
  void operator=(const VerilogEmitterState &) = delete;
};
} // namespace

//===----------------------------------------------------------------------===//
// EmitterBase
//===----------------------------------------------------------------------===//

namespace {

class EmitterBase {
public:
  // All of the mutable state we are maintaining.
  VerilogEmitterState &state;

  /// The stream to emit to.
  raw_ostream &os;

  EmitterBase(VerilogEmitterState &state, raw_ostream &os)
      : state(state), os(os) {}
  explicit EmitterBase(VerilogEmitterState &state)
      : EmitterBase(state, state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitOpError(message);
  }

  raw_ostream &indent() { return os.indent(state.currentIndent); }

  void addIndent() { state.currentIndent += INDENT_AMOUNT; }
  void reduceIndent() {
    assert(state.currentIndent >= INDENT_AMOUNT &&
           "Unintended indent wrap-around.");
    state.currentIndent -= INDENT_AMOUNT;
  }

  /// If we have location information for any of the specified operations,
  /// aggregate it together and print a pretty comment specifying where the
  /// operations came from.  In any case, print a newline.
  void emitLocationInfoAndNewLine(const SmallPtrSet<Operation *, 8> &ops) {
    auto locInfo = getLocationInfoAsString(ops);
    if (!locInfo.empty())
      os << "\t// " << locInfo;
    os << '\n';
  }

  void emitTextWithSubstitutions(StringRef string, Operation *op,
                                 std::function<void(Value)> operandEmitter,
                                 ArrayAttr symAttrs, ModuleNameManager &names);

  /// Emit the value of a StringAttr as one or more Verilog "one-line" comments
  /// ("//").  Break the comment to respect the emittedLineLength and trim
  /// whitespace after a line break.  Do nothing if the StringAttr is null or
  /// the value is empty.
  void emitComment(StringAttr comment);

  /// Given an expression that is spilled into a temporary wire, try to
  /// synthesize a better name than "_T_42" based on the structure of the
  /// expression.
  StringAttr inferStructuralNameForTemporary(Value expr);

private:
  void operator=(const EmitterBase &) = delete;
  EmitterBase(const EmitterBase &) = delete;
};
} // end anonymous namespace

void EmitterBase::emitTextWithSubstitutions(
    StringRef string, Operation *op, std::function<void(Value)> operandEmitter,
    ArrayAttr symAttrs, ModuleNameManager &names) {

  // Perform operand substitions as we emit the line string.  We turn {{42}}
  // into the value of operand 42.
  auto namify = [&](Attribute sym, SymbolCache::Item item) {
    // CAVEAT: These accesses can reach into other modules through inner name
    // references, which are currently being processed. Do not add those remote
    // operations to this module's `names`, which is reserved for things named
    // *within* this module. Instead, you have to rely on those remote
    // operations to have been named inside the global names table. If they
    // haven't, take a look at name name legalization first.
    if (auto itemOp = item.getOp()) {
      if (item.hasPort()) {
        return getPortVerilogName(itemOp, item.getPort());
      }
      StringRef symOpName = getSymOpName(itemOp);
      if (!symOpName.empty())
        return symOpName;
      itemOp->emitError("cannot get name for symbol ") << sym;
    } else {
      op->emitError("cannot get name for symbol ") << sym;
    }
    return StringRef("<INVALID>");
  };

  // Scan 'line' for a substitution, emitting any non-substitution prefix,
  // then the mentioned operand, chopping the relevant text off 'line' and
  // returning true.  This returns false if no substitution is found.
  unsigned numSymOps = symAttrs.size();
  auto emitUntilSubstitution = [&](size_t next = 0) -> bool {
    size_t start = 0;
    while (1) {
      next = string.find("{{", next);
      if (next == StringRef::npos)
        return false;

      // Check to make sure we have a number followed by }}.  If not, we
      // ignore the {{ sequence as something that could happen in Verilog.
      next += 2;
      start = next;
      while (next < string.size() && isdigit(string[next]))
        ++next;
      // We need at least one digit.
      if (start == next) {
        next--;
        continue;
      }

      // We must have a }} right after the digits.
      if (!string.substr(next).startswith("}}"))
        continue;

      // We must be able to decode the integer into an unsigned.
      unsigned operandNo = 0;
      if (string.drop_front(start)
              .take_front(next - start)
              .getAsInteger(10, operandNo)) {
        emitError(op, "operand substitution too large");
        continue;
      }
      next += 2;

      // Emit any text before the substitution.
      os << string.take_front(start - 2);

      // operandNo can either refer to Operands or symOps.  symOps are
      // numbered after the operands.
      if (operandNo < op->getNumOperands())
        // Emit the operand.
        operandEmitter(op->getOperand(operandNo));
      else if ((operandNo - op->getNumOperands()) < numSymOps) {
        unsigned symOpNum = operandNo - op->getNumOperands();
        auto sym = symAttrs[symOpNum];
        StringRef symVerilogName;
        if (auto fsym = sym.dyn_cast<FlatSymbolRefAttr>()) {
          if (auto *symOp = state.symbolCache.getDefinition(fsym))
            symVerilogName = namify(sym, symOp);
        } else if (auto isym = sym.dyn_cast<InnerRefAttr>()) {
          auto symOp =
              state.symbolCache.getDefinition(isym.getModule(), isym.getName());
          symVerilogName = namify(sym, symOp);
        }
        os << symVerilogName;
      } else {
        emitError(op, "operand " + llvm::utostr(operandNo) + " isn't valid");
        continue;
      }
      // Forget about the part we emitted.
      string = string.drop_front(next);
      return true;
    }
  };

  // Emit all the substitutions.
  while (emitUntilSubstitution())
    ;

  // Emit any text after the last substitution.
  os << string;
}

void EmitterBase::emitComment(StringAttr comment) {
  if (!comment)
    return;

  // Set a line length for the comment.  Subtract off the leading comment and
  // space ("// ") as well as the current indent level to simplify later
  // arithmetic.  Ensure that this line length doesn't go below zero.
  auto lineLength = state.options.emittedLineLength - state.currentIndent - 3;
  if (lineLength > state.options.emittedLineLength)
    lineLength = 0;

  // Process the comment in line chunks extracted from manually specified line
  // breaks.  This is done to preserve user-specified line breaking if used.
  auto ref = comment.getValue();
  StringRef line;
  while (!ref.empty()) {
    std::tie(line, ref) = ref.split("\n");
    // Emit each comment line breaking it if it exceeds the emittedLineLength.
    for (;;) {
      indent();
      os << "// ";

      // Base case 1: the entire comment fits on one line.
      if (line.size() <= lineLength) {
        os << line << "\n";
        break;
      }

      // The comment does NOT fit on one line.  Use a simple algorithm to find
      // a position to break the line:
      //   1) Search backwards for whitespace and break there if you find it.
      //   2) If no whitespace exists in (1), search forward for whitespace
      //      and break there.
      // This algorithm violates the emittedLineLength if (2) ever occurrs,
      // but it's dead simple.
      auto breakPos = line.rfind(' ', lineLength);
      // No whitespace exists looking backwards.
      if (breakPos == StringRef::npos) {
        breakPos = line.find(' ', lineLength);
        // No whitespace exists looking forward (you hit the end of the
        // string).
        if (breakPos == StringRef::npos)
          breakPos = line.size();
      }

      // Emit up to the break position.  Trim any whitespace after the break
      // position.  Exit if nothing is left to emit.  Otherwise, update the
      // comment ref and continue;
      os << line.take_front(breakPos) << "\n";
      breakPos = line.find_first_not_of(' ', breakPos);
      // Base Case 2: nothing left except whitespace.
      if (breakPos == StringRef::npos)
        break;

      line = line.drop_front(breakPos);
    }
  }
}

/// Given an expression that is spilled into a temporary wire, try to synthesize
/// a better name than "_T_42" based on the structure of the expression.
StringAttr EmitterBase::inferStructuralNameForTemporary(Value expr) {
  StringAttr result;
  bool addPrefixUnderScore = true;

  // Look through read_inout.
  if (auto read = expr.getDefiningOp<ReadInOutOp>())
    return inferStructuralNameForTemporary(read.input());

  // Module ports carry names!
  if (auto blockArg = expr.dyn_cast<BlockArgument>()) {
    auto moduleOp = cast<HWModuleOp>(blockArg.getOwner()->getParentOp());
    StringRef name = getPortVerilogName(moduleOp, blockArg.getArgNumber());
    result = StringAttr::get(expr.getContext(), name);

  } else if (auto *op = expr.getDefiningOp()) {
    // Uses of a wire or register can be done inline.
    if (isa<WireOp, RegOp>(op)) {
      StringRef name = getSymOpName(op);
      result = StringAttr::get(expr.getContext(), name);

    } else if (auto nameHint = op->getAttrOfType<StringAttr>("sv.namehint")) {
      // Use a dialect (sv) attribute to get a hint for the name if the op
      // doesn't explicitly specify it. Do this last
      result = nameHint;

      // If there is a namehint, don't add underscores to the name.
      addPrefixUnderScore = false;
    } else {
      TypeSwitch<Operation *>(op)
          // Generate a pretty name for VerbatimExpr's that look macro-like
          // using the same logic that generates the MLIR syntax name.
          .Case([&result](VerbatimExprOp verbatim) {
            verbatim.getAsmResultNames([&](Value, StringRef name) {
              result = StringAttr::get(verbatim.getContext(), name);
            });
          })
          .Case([&result](VerbatimExprSEOp verbatim) {
            verbatim.getAsmResultNames([&](Value, StringRef name) {
              result = StringAttr::get(verbatim.getContext(), name);
            });
          })

          // If this is an extract from a namable object, derive a name from it.
          .Case([&result, this](ExtractOp extract) {
            if (auto operandName =
                    inferStructuralNameForTemporary(extract.input())) {
              unsigned numBits = extract.getType().getWidth();
              if (numBits == 1)
                result = StringAttr::get(extract.getContext(),
                                         operandName.strref() + "_" +
                                             Twine(extract.lowBit()));
              else
                result =
                    StringAttr::get(extract.getContext(),
                                    operandName.strref() + "_" +
                                        Twine(extract.lowBit() + numBits - 1) +
                                        "to" + Twine(extract.lowBit()));
            }
          });
      // TODO: handle other common patterns.
    }
  }

  // Make sure any synthesized name starts with an _.
  if (!result || result.strref().empty())
    return {};

  // Make sure that all temporary names start with an underscore.
  if (addPrefixUnderScore && result.strref().front() != '_')
    result = StringAttr::get(expr.getContext(), "_" + result.strref());

  return result;
}

//===----------------------------------------------------------------------===//
// ModuleEmitter
//===----------------------------------------------------------------------===//

namespace {

class ModuleEmitter : public EmitterBase {
public:
  explicit ModuleEmitter(VerilogEmitterState &state) : EmitterBase(state) {}

  void emitHWModule(HWModuleOp module);
  void emitHWExternModule(HWModuleExternOp module);
  void emitHWGeneratedModule(HWModuleGeneratedOp module);

  // Statements.
  void emitStatement(Operation *op);
  void emitBind(BindOp op);
  void emitBindInterface(BindInterfaceOp op);

  StringRef getNameRemotely(Value value, const ModulePortInfo &modulePorts,
                            HWModuleOp remoteModule);

  /// Legalize the given field name if it is an invalid verilog name.
  StringRef getVerilogStructFieldName(StringAttr field) {
    return fieldNameResolver.getRenamedFieldName(field).getValue();
  }

  //===--------------------------------------------------------------------===//
  // Methods for formatting types.

  /// Emit a type's packed dimensions.
  void emitTypeDims(Type type, Location loc, raw_ostream &os);

  /// Print the specified packed portion of the type to the specified stream,
  ///
  ///  * When `implicitIntType` is false, a "logic" is printed.  This is used in
  ///        struct fields and typedefs.
  ///  * When `singleBitDefaultType` is false, single bit values are printed as
  ///       `[0:0]`.  This is used in parameter lists.
  ///
  /// This returns true if anything was printed.
  bool printPackedType(Type type, raw_ostream &os, Location loc,
                       bool implicitIntType = true,
                       bool singleBitDefaultType = true);

  /// Output the unpacked array dimensions.  This is the part of the type that
  /// is to the right of the name.
  void printUnpackedTypePostfix(Type type, raw_ostream &os);

  //===--------------------------------------------------------------------===//
  // Methods for formatting parameters.

  /// Prints a parameter attribute expression in a Verilog compatible way to the
  /// specified stream.  This returns the precedence of the generated string.
  SubExprInfo printParamValue(Attribute value, raw_ostream &os,
                              function_ref<InFlightDiagnostic()> emitError);

  SubExprInfo printParamValue(Attribute value, raw_ostream &os,
                              VerilogPrecedence parenthesizeIfLooserThan,
                              function_ref<InFlightDiagnostic()> emitError);

  //===--------------------------------------------------------------------===//
  // Mutable state while emitting a module body.

  /// This is the current module being emitted for a HWModuleOp.
  HWModuleOp currentModuleOp;

  /// This set keeps track of all of the expression nodes that need to be
  /// emitted as standalone wire declarations.  This can happen because they are
  /// multiply-used or because the user requires a name to reference.
  SmallPtrSet<Operation *, 16> outOfLineExpressions;

  /// This set keeps track of expressions that were emitted into their
  /// 'automatic logic' or 'localparam' declaration.  This is only used for
  /// expressions in a procedural region, because we otherwise just emit wires
  /// on demand.
  SmallPtrSet<Operation *, 16> expressionsEmittedIntoDecl;

  /// This class keeps track of field name renamings in the module scope.
  FieldNameResolver fieldNameResolver;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Methods for formatting types.

/// Emit a list of dimensions.
static void emitDims(ArrayRef<Attribute> dims, raw_ostream &os, Location loc,
                     ModuleEmitter &emitter) {
  for (Attribute width : dims) {
    if (!width) {
      os << "<<invalid type>>";
      continue;
    }
    if (auto intAttr = width.dyn_cast<IntegerAttr>()) {
      if (intAttr.getValue().isZero())
        os << "/*Zero Width*/";
      else
        os << '[' << (intAttr.getValue().getZExtValue() - 1) << ":0]";
      continue;
    }

    // Otherwise it must be a parameterized dimension.  Shove the "-1" into the
    // attribute so it gets printed in canonical form.
    auto negOne = getInt32Attr(loc.getContext(), -1);
    width = ParamExprAttr::get(PEO::Add, width, negOne);
    os << '[';
    emitter.printParamValue(width, os, [loc]() {
      return mlir::emitError(loc, "invalid parameter in type");
    });
    os << ":0]";
  }
}

/// Emit a type's packed dimensions.
void ModuleEmitter::emitTypeDims(Type type, Location loc, raw_ostream &os) {
  SmallVector<Attribute, 4> dims;
  getTypeDims(dims, type, loc);
  emitDims(dims, os, loc, *this);
}

/// Output the basic type that consists of packed and primitive types.  This is
/// those to the left of the name in verilog. implicitIntType controls whether
/// to print a base type for (logic) for inteters or whether the caller will
/// have handled this (with logic, wire, reg, etc).
///
/// Returns true when anything was printed out.
static bool printPackedTypeImpl(Type type, raw_ostream &os, Location loc,
                                SmallVectorImpl<Attribute> &dims,
                                bool implicitIntType, bool singleBitDefaultType,
                                ModuleEmitter &emitter) {
  return TypeSwitch<Type, bool>(type)
      .Case<IntegerType>([&](IntegerType integerType) {
        if (!implicitIntType)
          os << "logic";
        if (integerType.getWidth() != 1 || !singleBitDefaultType)
          dims.push_back(
              getInt32Attr(type.getContext(), integerType.getWidth()));
        if (!dims.empty() && !implicitIntType)
          os << ' ';

        emitDims(dims, os, loc, emitter);
        return !dims.empty() || !implicitIntType;
      })
      .Case<IntType>([&](IntType intType) {
        if (!implicitIntType)
          os << "logic ";
        dims.push_back(intType.getWidth());
        emitDims(dims, os, loc, emitter);
        return true;
      })
      .Case<ArrayType>([&](ArrayType arrayType) {
        dims.push_back(
            getInt32Attr(arrayType.getContext(), arrayType.getSize()));
        return printPackedTypeImpl(arrayType.getElementType(), os, loc, dims,
                                   implicitIntType, singleBitDefaultType,
                                   emitter);
      })
      .Case<InOutType>([&](InOutType inoutType) {
        return printPackedTypeImpl(inoutType.getElementType(), os, loc, dims,
                                   implicitIntType, singleBitDefaultType,
                                   emitter);
      })
      .Case<StructType>([&](StructType structType) {
        if (structType.getElements().empty()) {
          if (!implicitIntType)
            os << "logic ";
          os << "/*Zero Width*/";
          return true;
        }
        os << "struct packed {";
        for (auto &element : structType.getElements()) {
          SmallVector<Attribute, 8> structDims;
          printPackedTypeImpl(stripUnpackedTypes(element.type), os, loc,
                              structDims, /*implicitIntType=*/false,
                              /*singleBitDefaultType=*/true, emitter);
          os << ' ' << emitter.getVerilogStructFieldName(element.name);
          emitter.printUnpackedTypePostfix(element.type, os);
          os << "; ";
        }
        os << '}';
        emitDims(dims, os, loc, emitter);
        return true;
      })

      .Case<InterfaceType>([](InterfaceType ifaceType) { return false; })
      .Case<UnpackedArrayType>([&](UnpackedArrayType arrayType) {
        os << "<<unexpected unpacked array>>";
        mlir::emitError(loc, "Unexpected unpacked array in packed type ")
            << arrayType;
        return true;
      })
      .Case<TypeAliasType>([&](TypeAliasType typeRef) {
        auto typedecl = typeRef.getTypeDecl(emitter.state.symbolCache);
        if (!typedecl) {
          mlir::emitError(loc, "unresolvable type reference");
          return false;
        }
        if (typedecl.type() != typeRef.getInnerType()) {
          mlir::emitError(loc, "declared type did not match aliased type");
          return false;
        }

        os << typedecl.getPreferredName();
        emitDims(dims, os, typedecl->getLoc(), emitter);
        return true;
      })
      .Default([&](Type type) {
        os << "<<invalid type '" << type << "'>>";
        mlir::emitError(loc, "value has an unsupported verilog type ") << type;
        return true;
      });
}

/// Print the specified packed portion of the type to the specified stream,
///
///  * When `implicitIntType` is false, a "logic" is printed.  This is used in
///        struct fields and typedefs.
///  * When `singleBitDefaultType` is false, single bit values are printed as
///       `[0:0]`.  This is used in parameter lists.
///
/// This returns true if anything was printed.
bool ModuleEmitter::printPackedType(Type type, raw_ostream &os, Location loc,
                                    bool implicitIntType,
                                    bool singleBitDefaultType) {
  SmallVector<Attribute, 8> packedDimensions;
  return printPackedTypeImpl(type, os, loc, packedDimensions, implicitIntType,
                             singleBitDefaultType, *this);
}

/// Output the unpacked array dimensions.  This is the part of the type that is
/// to the right of the name.
void ModuleEmitter::printUnpackedTypePostfix(Type type, raw_ostream &os) {
  TypeSwitch<Type, void>(type)
      .Case<InOutType>([&](InOutType inoutType) {
        printUnpackedTypePostfix(inoutType.getElementType(), os);
      })
      .Case<UnpackedArrayType>([&](UnpackedArrayType arrayType) {
        os << "[0:" << (arrayType.getSize() - 1) << "]";
        printUnpackedTypePostfix(arrayType.getElementType(), os);
      })
      .Case<InterfaceType>([&](auto) {
        // Interface instantiations have parentheses like a module with no
        // ports.
        os << "()";
      });
}

//===----------------------------------------------------------------------===//
// Methods for formatting parameters.

/// Prints a parameter attribute expression in a Verilog compatible way to the
/// specified stream.  This returns the precedence of the generated string.
SubExprInfo
ModuleEmitter::printParamValue(Attribute value, raw_ostream &os,
                               function_ref<InFlightDiagnostic()> emitError) {
  return printParamValue(value, os, VerilogPrecedence::LowestPrecedence,
                         emitError);
}

/// Helper that prints a parameter constant value in a Verilog compatible way.
/// This returns the precedence of the generated string.
SubExprInfo
ModuleEmitter::printParamValue(Attribute value, raw_ostream &os,
                               VerilogPrecedence parenthesizeIfLooserThan,
                               function_ref<InFlightDiagnostic()> emitError) {
  if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
    IntegerType intTy = intAttr.getType().cast<IntegerType>();
    APInt value = intAttr.getValue();

    // We omit the width specifier if the value is <= 32-bits in size, which
    // makes this more compatible with unknown width extmodules.
    if (intTy.getWidth() > 32) {
      // Sign comes out before any width specifier.
      if (value.isNegative() && (intTy.isSigned() || intTy.isSignless())) {
        os << '-';
        value = -value;
      }
      if (intTy.isSigned())
        os << intTy.getWidth() << "'sd";
      else
        os << intTy.getWidth() << "'d";
    }
    value.print(os, intTy.isSigned());
    return {Symbol, intTy.isSigned() ? IsSigned : IsUnsigned};
  }
  if (auto strAttr = value.dyn_cast<StringAttr>()) {
    os << '"';
    os.write_escaped(strAttr.getValue());
    os << '"';
    return {Symbol, IsUnsigned};
  }
  if (auto fpAttr = value.dyn_cast<FloatAttr>()) {
    // TODO: relying on float printing to be precise is not a good idea.
    os << fpAttr.getValueAsDouble();
    return {Symbol, IsUnsigned};
  }
  if (auto verbatimParam = value.dyn_cast<ParamVerbatimAttr>()) {
    os << verbatimParam.getValue().getValue();
    return {Symbol, IsUnsigned};
  }
  if (auto parameterRef = value.dyn_cast<ParamDeclRefAttr>()) {
    // Get the name of this parameter (in case it got renamed).
    os << state.globalNames.getParameterVerilogName(currentModuleOp,
                                                    parameterRef.getName());

    // TODO: Should we support signed parameters?
    return {Symbol, IsUnsigned};
  }

  // Handle nested expressions.
  auto expr = value.dyn_cast<ParamExprAttr>();
  if (!expr) {
    os << "<<UNKNOWN MLIRATTR: " << value << ">>";
    emitError() << " = " << value;
    return {LowestPrecedence, IsUnsigned};
  }

  StringRef operatorStr;
  VerilogPrecedence subprecedence = ForceEmitMultiUse;
  Optional<SubExprSignResult> operandSign;

  switch (expr.getOpcode()) {
  case PEO::Add:
    operatorStr = " + ";
    subprecedence = Addition;
    break;
  case PEO::Mul:
    operatorStr = " * ";
    subprecedence = Multiply;
    break;
  case PEO::And:
    operatorStr = " & ";
    subprecedence = And;
    break;
  case PEO::Or:
    operatorStr = " | ";
    subprecedence = Or;
    break;
  case PEO::Xor:
    operatorStr = " ^ ";
    subprecedence = Xor;
    break;
  case PEO::Shl:
    operatorStr = " << ";
    subprecedence = Shift;
    break;
  case PEO::ShrU:
    // >> in verilog is always a logical shift even if operands are signed.
    operatorStr = " >> ";
    subprecedence = Shift;
    break;
  case PEO::ShrS:
    // >>> in verilog is an arithmetic shift if both operands are signed.
    operatorStr = " >>> ";
    subprecedence = Shift;
    operandSign = IsSigned;
    break;
  case PEO::DivU:
    operatorStr = " / ";
    subprecedence = Multiply;
    operandSign = IsUnsigned;
    break;
  case PEO::DivS:
    operatorStr = " / ";
    subprecedence = Multiply;
    operandSign = IsSigned;
    break;
  case PEO::ModU:
    operatorStr = " % ";
    subprecedence = Multiply;
    operandSign = IsUnsigned;
    break;
  case PEO::ModS:
    operatorStr = " % ";
    subprecedence = Multiply;
    operandSign = IsSigned;
    break;
  }

  // Emit the specified operand with a $signed() or $unsigned() wrapper around
  // it if context requires a specific signedness to compute the right value.
  // This returns true if the operand is signed.
  // TODO: This could try harder to omit redundant casts like the mainline
  // expression emitter.
  auto emitOperand = [&](Attribute operand) -> bool {
    if (operandSign.hasValue())
      os << (operandSign.getValue() == IsSigned ? "$signed(" : "$unsigned(");
    auto signedness =
        printParamValue(operand, os, subprecedence, emitError).signedness;
    if (operandSign.hasValue()) {
      os << ')';
      signedness = operandSign.getValue();
    }
    return signedness == IsSigned;
  };

  if (subprecedence > parenthesizeIfLooserThan)
    os << '(';
  bool allOperandsSigned = emitOperand(expr.getOperands()[0]);
  for (auto op : ArrayRef(expr.getOperands()).drop_front()) {
    // Handle the special case of (a + b + -42) as (a + b - 42).
    // TODO: Also handle (a + b + x*-1).
    if (expr.getOpcode() == PEO::Add) {
      if (auto integer = op.dyn_cast<IntegerAttr>()) {
        const APInt &value = integer.getValue();
        if (value.isNegative() && !value.isMinSignedValue()) {
          os << " - ";
          allOperandsSigned &=
              emitOperand(IntegerAttr::get(op.getType(), -value));
          continue;
        }
      }
    }

    os << operatorStr;
    allOperandsSigned &= emitOperand(op);
  }
  if (subprecedence > parenthesizeIfLooserThan) {
    os << ')';
    subprecedence = Symbol;
  }
  return {subprecedence, allOperandsSigned ? IsSigned : IsUnsigned};
}

//===----------------------------------------------------------------------===//
// Expression Emission
//===----------------------------------------------------------------------===//

namespace {
/// This builds a recursively nested expression from an SSA use-def graph.  This
/// uses a post-order walk, but it needs to obey precedence and signedness
/// constraints that depend on the behavior of the child nodes.  To handle this,
/// we emit the characters to a SmallVector which allows us to emit a bunch of
/// stuff, then pre-insert parentheses and other things if we find out that it
/// was needed later.
class ExprEmitter : public EmitterBase,
                    public TypeOpVisitor<ExprEmitter, SubExprInfo>,
                    public CombinationalVisitor<ExprEmitter, SubExprInfo>,
                    public Visitor<ExprEmitter, SubExprInfo> {
public:
  /// Create an ExprEmitter for the specified module emitter, and keeping track
  /// of any emitted expressions in the specified set.  If any subexpressions
  /// are too large to emit, then they are added into tooLargeSubExpressions to
  /// be emitted independently by the caller.
  ExprEmitter(ModuleEmitter &emitter, SmallVectorImpl<char> &outBuffer,
              SmallPtrSet<Operation *, 8> &emittedExprs,
              SmallVectorImpl<Operation *> &tooLargeSubExpressions,
              ModuleNameManager &names)
      : EmitterBase(emitter.state, os), emitter(emitter),
        emittedExprs(emittedExprs),
        tooLargeSubExpressions(tooLargeSubExpressions), outBuffer(outBuffer),
        os(outBuffer), names(names) {}

  /// Emit the specified value as an expression.  If this is an inline-emitted
  /// expression, we emit that expression, otherwise we emit a reference to the
  /// already computed name.
  ///
  void emitExpression(Value exp, VerilogPrecedence parenthesizeIfLooserThan) {
    // Emit the expression.
    emitSubExpr(exp, parenthesizeIfLooserThan, OOLTopLevel,
                /*signRequirement*/ NoRequirement);

    // Emitted expression might break the line length constraint so align it
    // here.
    formatOutBuffer();
  }

private:
  friend class TypeOpVisitor<ExprEmitter, SubExprInfo>;
  friend class CombinationalVisitor<ExprEmitter, SubExprInfo>;
  friend class Visitor<ExprEmitter, SubExprInfo>;

  enum SubExprSignRequirement { NoRequirement, RequireSigned, RequireUnsigned };
  enum SubExprOutOfLineBehavior {
    OOLTopLevel, //< Top level expressions shouldn't be emitted out of line.
    OOLUnary,    //< Unary expressions are more generous on line lengths.
    OOLBinary    //< Binary expressions split easily.
  };

  /// Emit the specified value `exp` as a subexpression to the stream.  The
  /// `parenthesizeIfLooserThan` parameter indicates when parentheses should be
  /// added aroun the subexpression.  The `signReq` flag can cause emitSubExpr
  /// to emit a subexpression that is guaranteed to be signed or unsigned, and
  /// the `isSelfDeterminedUnsignedValue` flag indicates whether the value is
  /// known to be have "self determined" width, allowing us to omit extensions.
  SubExprInfo emitSubExpr(Value exp, VerilogPrecedence parenthesizeIfLooserThan,
                          SubExprOutOfLineBehavior outOfLineBehavior,
                          SubExprSignRequirement signReq = NoRequirement,
                          bool isSelfDeterminedUnsignedValue = false);

  void retroactivelyEmitExpressionIntoTemporary(Operation *op);
  void formatOutBuffer();

  SubExprInfo visitUnhandledExpr(Operation *op);
  SubExprInfo visitInvalidComb(Operation *op) {
    return dispatchTypeOpVisitor(op);
  }
  SubExprInfo visitUnhandledComb(Operation *op) {
    return visitUnhandledExpr(op);
  }
  SubExprInfo visitInvalidTypeOp(Operation *op) {
    return dispatchSVVisitor(op);
  }
  SubExprInfo visitUnhandledTypeOp(Operation *op) {
    return visitUnhandledExpr(op);
  }
  SubExprInfo visitUnhandledSV(Operation *op) { return visitUnhandledExpr(op); }

  using Visitor::visitSV;

  /// These are flags that control `emitBinary`.
  enum EmitBinaryFlags {
    EB_RequireSignedOperands = RequireSigned,     /* 0x1*/
    EB_RequireUnsignedOperands = RequireUnsigned, /* 0x2*/
    EB_OperandSignRequirementMask = 0x3,

    /// This flag indicates that the RHS operand is an unsigned value that has
    /// "self determined" width.  This means that we can omit explicit zero
    /// extensions from it, and don't impose a sign on it.
    EB_RHS_UnsignedWithSelfDeterminedWidth = 0x4,

    /// This flag indicates that the result should be wrapped in a $signed(x)
    /// expression to force the result to signed.
    EB_ForceResultSigned = 0x8,
  };

  /// Emit a binary expression.  The "emitBinaryFlags" are a bitset from
  /// EmitBinaryFlags.
  SubExprInfo emitBinary(Operation *op, VerilogPrecedence prec,
                         const char *syntax, unsigned emitBinaryFlags = 0);

  SubExprInfo emitUnary(Operation *op, const char *syntax,
                        bool resultAlwaysUnsigned = false);

  SubExprInfo visitSV(GetModportOp op);
  SubExprInfo visitSV(ReadInterfaceSignalOp op);
  SubExprInfo visitSV(XMROp op);
  SubExprInfo visitVerbatimExprOp(Operation *op, ArrayAttr symbols);
  SubExprInfo visitSV(VerbatimExprOp op) {
    return visitVerbatimExprOp(op, op.symbols());
  }
  SubExprInfo visitSV(VerbatimExprSEOp op) {
    return visitVerbatimExprOp(op, op.symbols());
  }
  SubExprInfo visitSV(ConstantXOp op);
  SubExprInfo visitSV(ConstantZOp op);

  // Noop cast operators.
  SubExprInfo visitSV(ReadInOutOp op) {
    return emitSubExpr(op->getOperand(0), LowestPrecedence, OOLUnary);
  }
  SubExprInfo visitSV(ArrayIndexInOutOp op);
  SubExprInfo visitSV(IndexedPartSelectInOutOp op);
  SubExprInfo visitSV(IndexedPartSelectOp op);
  SubExprInfo visitSV(StructFieldInOutOp op);

  // Sampled value functions
  SubExprInfo visitSV(SampledOp op);

  // Other
  using TypeOpVisitor::visitTypeOp;
  SubExprInfo visitTypeOp(ConstantOp op);
  SubExprInfo visitTypeOp(BitcastOp op);
  SubExprInfo visitTypeOp(ParamValueOp op);
  SubExprInfo visitTypeOp(ArraySliceOp op);
  SubExprInfo visitTypeOp(ArrayGetOp op);
  SubExprInfo visitTypeOp(ArrayCreateOp op);
  SubExprInfo visitTypeOp(ArrayConcatOp op);
  SubExprInfo visitTypeOp(StructCreateOp op);
  SubExprInfo visitTypeOp(StructExtractOp op);
  SubExprInfo visitTypeOp(StructInjectOp op);

  // Comb Dialect Operations
  using CombinationalVisitor::visitComb;
  SubExprInfo visitComb(MuxOp op);
  SubExprInfo visitComb(AddOp op) {
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, Addition, "+");
  }
  SubExprInfo visitComb(SubOp op) { return emitBinary(op, Addition, "-"); }
  SubExprInfo visitComb(MulOp op) {
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, Multiply, "*");
  }
  SubExprInfo visitComb(DivUOp op) {
    return emitBinary(op, Multiply, "/", EB_RequireUnsignedOperands);
  }
  SubExprInfo visitComb(DivSOp op) {
    return emitBinary(op, Multiply, "/", EB_RequireSignedOperands);
  }
  SubExprInfo visitComb(ModUOp op) {
    return emitBinary(op, Multiply, "%", EB_RequireUnsignedOperands);
  }
  SubExprInfo visitComb(ModSOp op) {
    return emitBinary(op, Multiply, "%", EB_RequireSignedOperands);
  }
  SubExprInfo visitComb(ShlOp op) {
    return emitBinary(op, Shift, "<<", EB_RHS_UnsignedWithSelfDeterminedWidth);
  }
  SubExprInfo visitComb(ShrUOp op) {
    // >> in Verilog is always an unsigned right shift.
    return emitBinary(op, Shift, ">>", EB_RHS_UnsignedWithSelfDeterminedWidth);
  }
  SubExprInfo visitComb(ShrSOp op) {
    // >>> is only an arithmetic shift right when both operands are signed.
    // Otherwise it does a logical shift.
    return emitBinary(op, Shift, ">>>",
                      EB_RequireSignedOperands | EB_ForceResultSigned |
                          EB_RHS_UnsignedWithSelfDeterminedWidth);
  }
  SubExprInfo visitComb(AndOp op) {
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, And, "&");
  }
  SubExprInfo visitComb(OrOp op) {
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, Or, "|");
  }
  SubExprInfo visitComb(XorOp op) {
    if (op.isBinaryNot())
      return emitUnary(op, "~");
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, Xor, "^");
  }

  // SystemVerilog spec 11.8.1: "Reduction operator results are unsigned,
  // regardless of the operands."
  SubExprInfo visitComb(ParityOp op) { return emitUnary(op, "^", true); }

  SubExprInfo visitComb(ReplicateOp op);
  SubExprInfo visitComb(ConcatOp op);
  SubExprInfo visitComb(ExtractOp op);
  SubExprInfo visitComb(ICmpOp op);

public:
  ModuleEmitter &emitter;

private:
  /// This is set (before a visit method is called) if emitSubExpr would
  /// prefer to get an output of a specific sign.  This is a hint to cause the
  /// visitor to change its emission strategy, but the visit method can ignore
  /// it without a correctness problem.
  SubExprSignRequirement signPreference = NoRequirement;

  /// Keep track of all operations emitted within this subexpression for
  /// location information tracking.
  SmallPtrSet<Operation *, 8> &emittedExprs;

  /// If any subexpressions would result in too large of a line, report it
  /// back to the caller in this vector.
  SmallVectorImpl<Operation *> &tooLargeSubExpressions;
  SmallVectorImpl<char> &outBuffer;
  llvm::raw_svector_ostream os;
  // Track legalized names.
  ModuleNameManager &names;
};
} // end anonymous namespace

SubExprInfo ExprEmitter::emitBinary(Operation *op, VerilogPrecedence prec,
                                    const char *syntax,
                                    unsigned emitBinaryFlags) {
  if (emitBinaryFlags & EB_ForceResultSigned)
    os << "$signed(";
  auto operandSignReq =
      SubExprSignRequirement(emitBinaryFlags & EB_OperandSignRequirementMask);
  auto lhsInfo =
      emitSubExpr(op->getOperand(0), prec, OOLBinary, operandSignReq);
  os << ' ' << syntax << ' ';

  // Right associative operators are already generally variadic, we need to
  // handle things like: (a<4> == b<4>) == (c<3> == d<3>).  When processing the
  // top operation of the tree, the rhs needs parens.  When processing
  // known-reassociative operators like +, ^, etc we don't need parens.
  // TODO: MLIR should have general "Associative" trait.
  auto rhsPrec = prec;
  if (!isa<AddOp, MulOp, AndOp, OrOp, XorOp>(op))
    rhsPrec = VerilogPrecedence(prec - 1);

  // If the RHS operand has self-determined width and always treated as
  // unsigned, inform emitSubExpr of this.  This is true for the shift amount in
  // a shift operation.
  bool rhsIsUnsignedValueWithSelfDeterminedWidth = false;
  if (emitBinaryFlags & EB_RHS_UnsignedWithSelfDeterminedWidth) {
    rhsIsUnsignedValueWithSelfDeterminedWidth = true;
    operandSignReq = NoRequirement;
  }

  auto rhsInfo =
      emitSubExpr(op->getOperand(1), rhsPrec, OOLBinary, operandSignReq,
                  rhsIsUnsignedValueWithSelfDeterminedWidth);

  // SystemVerilog 11.8.1 says that the result of a binary expression is signed
  // only if both operands are signed.
  SubExprSignResult signedness = IsUnsigned;
  if (lhsInfo.signedness == IsSigned && rhsInfo.signedness == IsSigned)
    signedness = IsSigned;

  if (emitBinaryFlags & EB_ForceResultSigned) {
    os << ')';
    signedness = IsSigned;
    prec = Selection;
  }

  return {prec, signedness};
}

SubExprInfo ExprEmitter::emitUnary(Operation *op, const char *syntax,
                                   bool resultAlwaysUnsigned) {
  os << syntax;
  auto signedness =
      emitSubExpr(op->getOperand(0), Selection, OOLUnary).signedness;
  return {Unary, resultAlwaysUnsigned ? IsUnsigned : signedness};
}

/// This function split the output buffer into multiple lines if the emitted
/// length is larger than the constraint.
void ExprEmitter::formatOutBuffer() {
  // If the output already satisfies the constraint, skip here.
  if (outBuffer.size() <= state.options.emittedLineLength)
    return;

  SmallVector<char> tmpOutBuffer;
  llvm::raw_svector_ostream tmpOs(tmpOutBuffer);
  auto it = outBuffer.begin();
  unsigned currentIndex = 0;

  while (it != outBuffer.end()) {
    // Split by a white space.
    auto next = std::find(it, outBuffer.end(), ' ');
    unsigned tokenLength = std::distance(it, next);

    if (!tmpOutBuffer.empty() &&
        currentIndex + tokenLength > state.options.emittedLineLength) {
      // It breaks the line constraint, so insert a newline and indent.
      tmpOs << '\n';
      tmpOs.indent(state.currentIndent *
                   SPACE_PER_INDENT_IN_EXPRESSION_FORMATTING);
      currentIndex = tokenLength;
      tmpOutBuffer.insert(tmpOutBuffer.end(), it, next);
    } else {
      currentIndex += tokenLength;
      // If `tmpOutBuffer` is not empty, there exists a token before the
      // current token so insert a white space.
      if (!tmpOutBuffer.empty()) {
        currentIndex += 1;
        tmpOs << ' ';
      }
      tmpOutBuffer.insert(tmpOutBuffer.end(), it, next);
    }

    if (next == outBuffer.end())
      break;
    it = next + 1;
  }
  outBuffer = std::move(tmpOutBuffer);
}

/// We eagerly emit single-use expressions inline into big expression trees...
/// up to the point where they turn into massively long source lines of Verilog.
/// At that point, we retroactively break the huge expression by inserting
/// temporaries.  This handles the bookkeeping.
void ExprEmitter::retroactivelyEmitExpressionIntoTemporary(Operation *op) {
  assert(isVerilogExpression(op) && !emitter.outOfLineExpressions.count(op) &&
         "Should only be called on expressions thought to be inlined");

  emitter.outOfLineExpressions.insert(op);
  if (auto name = inferStructuralNameForTemporary(op->getResult(0)))
    names.addName(op->getResult(0), name);
  else
    names.addName(op->getResult(0), "_tmp");

  // Remember that this subexpr needs to be emitted independently.
  tooLargeSubExpressions.push_back(op);
}

/// If the specified extension is a zero extended version of another value,
/// return the shorter value, otherwise return null.
static Value isZeroExtension(Value value) {
  auto concat = value.getDefiningOp<ConcatOp>();
  if (!concat || concat.getNumOperands() != 2)
    return {};

  auto constant = concat.getOperand(0).getDefiningOp<ConstantOp>();
  if (constant && constant.getValue().isZero())
    return concat.getOperand(1);
  return {};
}

/// Emit the specified value `exp` as a subexpression to the stream.  The
/// `parenthesizeIfLooserThan` parameter indicates when parentheses should be
/// added aroun the subexpression.  The `signReq` flag can cause emitSubExpr
/// to emit a subexpression that is guaranteed to be signed or unsigned, and
/// the `isSelfDeterminedUnsignedValue` flag indicates whether the value is
/// known to be have "self determined" width, allowing us to omit extensions.
SubExprInfo ExprEmitter::emitSubExpr(Value exp,
                                     VerilogPrecedence parenthesizeIfLooserThan,
                                     SubExprOutOfLineBehavior outOfLineBehavior,
                                     SubExprSignRequirement signRequirement,
                                     bool isSelfDeterminedUnsignedValue) {
  // If this is a self-determined unsigned value, look through any inline zero
  // extensions.  This occurs on the RHS of a shift operation for example.
  if (isSelfDeterminedUnsignedValue && exp.hasOneUse()) {
    if (auto smaller = isZeroExtension(exp))
      exp = smaller;
  }

  auto *op = exp.getDefiningOp();
  bool shouldEmitInlineExpr = op && isVerilogExpression(op);

  // Don't emit this expression inline if it has multiple uses.
  if (shouldEmitInlineExpr && parenthesizeIfLooserThan != ForceEmitMultiUse &&
      emitter.outOfLineExpressions.count(op))
    shouldEmitInlineExpr = false;

  // If this is a non-expr or shouldn't be done inline, just refer to its name.
  if (!shouldEmitInlineExpr) {
    // All wires are declared as unsigned, so if the client needed it signed,
    // emit a conversion.
    if (signRequirement == RequireSigned) {
      os << "$signed(" << names.getName(exp) << ')';
      return {Symbol, IsSigned};
    }

    os << names.getName(exp);
    return {Symbol, IsUnsigned};
  }

  unsigned subExprStartIndex = outBuffer.size();

  // Inform the visit method about the preferred sign we want from the result.
  // It may choose to ignore this, but some emitters can change behavior based
  // on contextual desired sign.
  signPreference = signRequirement;

  // Okay, this is an expression we should emit inline.  Do this through our
  // visitor.
  auto expInfo = dispatchCombinationalVisitor(exp.getDefiningOp());

  // Check cases where we have to insert things before the expression now that
  // we know things about it.
  auto addPrefix = [&](StringRef prefix) {
    outBuffer.insert(outBuffer.begin() + subExprStartIndex, prefix.begin(),
                     prefix.end());
  };
  if (signRequirement == RequireSigned && expInfo.signedness == IsUnsigned) {
    addPrefix("$signed(");
    os << ')';
    expInfo.signedness = IsSigned;
    expInfo.precedence = Selection;
  } else if (signRequirement == RequireUnsigned &&
             expInfo.signedness == IsSigned) {
    addPrefix("$unsigned(");
    os << ')';
    expInfo.signedness = IsUnsigned;
    expInfo.precedence = Selection;
  } else if (expInfo.precedence > parenthesizeIfLooserThan) {
    // If this subexpression would bind looser than the expression it is bound
    // into, then we need to parenthesize it.  Insert the parentheses
    // retroactively.
    addPrefix("(");
    os << ')';
    // Reset the precedence to the () level.
    expInfo.precedence = Selection;
  }

  // If we emitted this subexpression and it resulted in something very large,
  // then we may be in the process of making super huge lines.  Back off to
  // emitting this as its own temporary on its own line.
  unsigned threshold;
  switch (outOfLineBehavior) {
  case OOLTopLevel:
    threshold = ~0U;
    break;
  case OOLUnary:
    threshold = std::max(state.options.emittedLineLength - 20, 10U);
    break;
  case OOLBinary:
    threshold = std::max(state.options.emittedLineLength / 2, 10U);
    break;
  }

  if (outBuffer.size() - subExprStartIndex > threshold &&
      parenthesizeIfLooserThan != ForceEmitMultiUse &&
      !isExpressionAlwaysInline(op)) {
    // Inform the module emitter that this expression needs a temporary
    // wire/logic declaration and set it up so it will be referenced instead of
    // emitted inline.
    auto emitExpressionIntoTemporary = [&]() {
      retroactivelyEmitExpressionIntoTemporary(op);

      // Lop this off the buffer we emitted.
      outBuffer.resize(subExprStartIndex);

      // Try again, now it will get emitted as a out-of-line leaf.
      return emitSubExpr(exp, parenthesizeIfLooserThan, outOfLineBehavior,
                         signRequirement);
    };

    // If op has multiple uses or op is a too large expression, we have to spill
    // the expression.
    if (!op->hasOneUse() ||
        outBuffer.size() - subExprStartIndex >
            state.options.maximumNumberOfTokensPerExpression)
      return emitExpressionIntoTemporary();
  }

  // Remember that we emitted this.
  emittedExprs.insert(exp.getDefiningOp());
  return expInfo;
}

SubExprInfo ExprEmitter::visitComb(ReplicateOp op) {
  os << '{' << op.getMultiple() << '{';

  // If the subexpression is an inline concat, we can emit it as part of the
  // replicate.
  if (auto concatOp = op.getOperand().getDefiningOp<ConcatOp>()) {
    if (op.getOperand().hasOneUse() &&
        !emitter.outOfLineExpressions.count(concatOp)) {
      llvm::interleaveComma(concatOp.getOperands(), os, [&](Value v) {
        emitSubExpr(v, LowestPrecedence, OOLBinary);
      });
      os << "}}";
      return {Symbol, IsUnsigned};
    }
  }

  emitSubExpr(op.getOperand(), LowestPrecedence, OOLUnary);
  os << "}}";
  return {Symbol, IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(ConcatOp op) {
  os << '{';
  llvm::interleaveComma(op.getOperands(), os, [&](Value v) {
    emitSubExpr(v, LowestPrecedence, OOLBinary);
  });

  os << '}';
  return {Symbol, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(BitcastOp op) {
  // NOTE: Bitcasts are emitted out-of-line with their own wire declaration when
  // their dimensions don't match. SystemVerilog uses the wire declaration to
  // know what type this value is being casted to.
  Type toType = op.getType();
  if (!haveMatchingDims(toType, op.input().getType(), op.getLoc())) {
    os << "/*cast(bit";
    emitter.emitTypeDims(toType, op.getLoc(), os);
    os << ")*/";
  }
  return emitSubExpr(op.input(), LowestPrecedence, OOLUnary);
}

SubExprInfo ExprEmitter::visitComb(ICmpOp op) {
  const char *symop[] = {"==", "!=", "<",  "<=", ">",
                         ">=", "<",  "<=", ">",  ">="};
  SubExprSignRequirement signop[] = {
      // Equality
      NoRequirement, NoRequirement,
      // Signed Comparisons
      RequireSigned, RequireSigned, RequireSigned, RequireSigned,
      // Unsigned Comparisons
      RequireUnsigned, RequireUnsigned, RequireUnsigned, RequireUnsigned};

  auto pred = static_cast<uint64_t>(op.predicate());
  assert(pred < sizeof(symop) / sizeof(symop[0]));

  // Lower "== -1" to Reduction And.
  if (op.isEqualAllOnes())
    return emitUnary(op, "&", true);

  // Lower "!= 0" to Reduction Or.
  if (op.isNotEqualZero())
    return emitUnary(op, "|", true);

  auto result = emitBinary(op, Comparison, symop[pred], signop[pred]);

  // SystemVerilog 11.8.1: "Comparison... operator results are unsigned,
  // regardless of the operands".
  result.signedness = IsUnsigned;
  return result;
}

SubExprInfo ExprEmitter::visitComb(ExtractOp op) {
  unsigned loBit = op.lowBit();
  unsigned hiBit = loBit + op.getType().getWidth() - 1;

  auto x = emitSubExpr(op.input(), LowestPrecedence, OOLUnary);
  assert((x.precedence == Symbol ||
          (x.precedence == Selection && isOkToBitSelectFrom(op.input()))) &&
         "should be handled by isExpressionUnableToInline");

  // If we're extracting the whole input, just return it.  This is valid but
  // non-canonical IR, and we don't want to generate invalid Verilog.
  if (loBit == 0 && op.input().getType().getIntOrFloatBitWidth() == hiBit + 1)
    return x;

  os << '[' << hiBit;
  if (hiBit != loBit) // Emit x[4] instead of x[4:4].
    os << ':' << loBit;
  os << ']';
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(GetModportOp op) {
  auto decl = op.getReferencedDecl(state.symbolCache);
  os << names.getName(op.iface()) << '.' << getSymOpName(decl);
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(ReadInterfaceSignalOp op) {
  auto decl = op.getReferencedDecl(state.symbolCache);

  os << names.getName(op.iface()) << '.' << getSymOpName(decl);
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(XMROp op) {
  if (op.isRooted())
    os << "$root.";
  for (auto s : op.path())
    os << s.cast<StringAttr>().getValue() << '.';
  os << op.terminal();
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitVerbatimExprOp(Operation *op, ArrayAttr symbols) {
  emitTextWithSubstitutions(
      op->getAttrOfType<StringAttr>("string").getValue(), op,
      [&](Value operand) { emitSubExpr(operand, LowestPrecedence, OOLBinary); },
      symbols, names);

  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(ConstantXOp op) {
  os << op.getWidth() << "'bx";
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(ConstantZOp op) {
  os << op.getWidth() << "'bz";
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(ConstantOp op) {
  bool isNegated = false;
  const APInt &value = op.getValue();
  // If this is a negative signed number and not MININT (e.g. -128), then print
  // it as a negated positive number.
  if (signPreference == RequireSigned && value.isNegative() &&
      !value.isMinSignedValue()) {
    os << '-';
    isNegated = true;
  }

  os << op.getType().getWidth() << '\'';

  // Emit this as a signed constant if the caller would prefer that.
  if (signPreference == RequireSigned)
    os << 's';
  os << 'h';

  // Print negated if required.
  SmallString<32> valueStr;
  if (isNegated) {
    (-value).toStringUnsigned(valueStr, 16);
  } else {
    value.toStringUnsigned(valueStr, 16);
  }
  os << valueStr;
  return {Unary, signPreference == RequireSigned ? IsSigned : IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(ParamValueOp op) {
  return emitter.printParamValue(op.value(), os, [&]() {
    return op->emitOpError("invalid parameter use");
  });
}

// 11.5.1 "Vector bit-select and part-select addressing" allows a '+:' syntax
// for slicing operations.
SubExprInfo ExprEmitter::visitTypeOp(ArraySliceOp op) {
  auto arrayPrec = emitSubExpr(op.input(), Selection, OOLUnary);

  unsigned dstWidth = type_cast<ArrayType>(op.getType()).getSize();
  os << '[';
  emitSubExpr(op.lowIndex(), LowestPrecedence, OOLBinary);
  os << " +: " << dstWidth << ']';
  return {Selection, arrayPrec.signedness};
}

SubExprInfo ExprEmitter::visitTypeOp(ArrayGetOp op) {
  emitSubExpr(op.input(), Selection, OOLUnary);
  os << '[';
  emitSubExpr(op.index(), LowestPrecedence, OOLBinary);
  os << ']';
  return {Selection, IsUnsigned};
}

// Syntax from: section 5.11 "Array literals".
SubExprInfo ExprEmitter::visitTypeOp(ArrayCreateOp op) {
  os << '{';
  llvm::interleaveComma(op.inputs(), os, [&](Value operand) {
    os << "{";
    emitSubExpr(operand, LowestPrecedence, OOLBinary);
    os << "}";
  });
  os << '}';
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(ArrayConcatOp op) {
  os << '{';
  llvm::interleaveComma(op.getOperands(), os, [&](Value v) {
    emitSubExpr(v, LowestPrecedence, OOLBinary);
  });
  os << '}';
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(ArrayIndexInOutOp op) {
  auto arrayPrec = emitSubExpr(op.input(), Selection, OOLUnary);
  os << '[';
  emitSubExpr(op.index(), LowestPrecedence, OOLBinary);
  os << ']';
  return {Selection, arrayPrec.signedness};
}

SubExprInfo ExprEmitter::visitSV(IndexedPartSelectInOutOp op) {
  auto prec = emitSubExpr(op.input(), Selection, OOLUnary);
  os << '[';
  emitSubExpr(op.base(), LowestPrecedence, OOLBinary);
  if (op.decrement())
    os << " -: ";
  else
    os << " +: ";
  os << op.width() << ']';
  return {Selection, prec.signedness};
}

SubExprInfo ExprEmitter::visitSV(IndexedPartSelectOp op) {
  auto info = emitSubExpr(op.input(), LowestPrecedence, OOLUnary);
  os << '[';
  emitSubExpr(op.base(), LowestPrecedence, OOLBinary);
  if (op.decrement())
    os << " -: ";
  else
    os << " +: ";
  os << op.width();
  os << ']';
  return info;
}

SubExprInfo ExprEmitter::visitSV(StructFieldInOutOp op) {
  auto prec = emitSubExpr(op.input(), Selection, OOLUnary);
  os << '.' << emitter.getVerilogStructFieldName(op.fieldAttr());
  return {Selection, prec.signedness};
}

SubExprInfo ExprEmitter::visitSV(SampledOp op) {
  os << "$sampled(";
  auto info = emitSubExpr(op.expression(), LowestPrecedence, OOLTopLevel);
  os << ")";
  return info;
}

SubExprInfo ExprEmitter::visitComb(MuxOp op) {
  // The ?: operator is right associative.
  emitSubExpr(op.cond(), VerilogPrecedence(Conditional - 1), OOLBinary);
  os << " ? ";
  auto lhsInfo = emitSubExpr(op.trueValue(), VerilogPrecedence(Conditional - 1),
                             OOLBinary);
  os << " : ";
  auto rhsInfo = emitSubExpr(op.falseValue(), Conditional, OOLBinary);

  SubExprSignResult signedness = IsUnsigned;
  if (lhsInfo.signedness == IsSigned && rhsInfo.signedness == IsSigned)
    signedness = IsSigned;

  return {Conditional, signedness};
}

SubExprInfo ExprEmitter::visitTypeOp(StructCreateOp op) {
  StructType stype = op.getType();
  os << "'{";
  size_t i = 0;
  llvm::interleaveComma(
      stype.getElements(), os, [&](const StructType::FieldInfo &field) {
        os << emitter.getVerilogStructFieldName(field.name) << ": ";
        emitSubExpr(op.getOperand(i++), Selection, OOLBinary);
      });
  os << '}';
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(StructExtractOp op) {
  emitSubExpr(op.input(), Selection, OOLUnary);
  os << '.' << emitter.getVerilogStructFieldName(op.fieldAttr());
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(StructInjectOp op) {
  StructType stype = op.getType().cast<StructType>();
  os << "'{";
  llvm::interleaveComma(
      stype.getElements(), os, [&](const StructType::FieldInfo &field) {
        os << emitter.getVerilogStructFieldName(field.name) << ": ";
        if (field.name == op.field()) {
          emitSubExpr(op.newValue(), Selection, OOLBinary);
        } else {
          emitSubExpr(op.input(), Selection, OOLBinary);
          os << '.' << field.name.getValue();
        }
      });
  os << '}';
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitUnhandledExpr(Operation *op) {
  emitOpError(op, "cannot emit this expression to Verilog");
  os << "<<unsupported expr: " << op->getName().getStringRef() << ">>";
  return {Symbol, IsUnsigned};
}

//===----------------------------------------------------------------------===//
// NameCollector
//===----------------------------------------------------------------------===//

/// Return true if we are unable to ever inline the specified operation.  This
/// happens because not all Verilog expressions are composable, notably you
/// can only use bit selects like x[4:6] on simple expressions, you cannot use
/// expressions in the sensitivity list of always blocks, etc.
static bool isExpressionUnableToInline(Operation *op) {
  if (auto cast = dyn_cast<BitcastOp>(op))
    if (!haveMatchingDims(cast.input().getType(), cast.result().getType(),
                          op->getLoc()))
      // Bitcasts rely on the type being assigned to, so we cannot inline.
      return true;

  // StructCreateOp needs to be assigning to a named temporary so that types
  // are inferred properly by verilog
  if (isa<StructCreateOp>(op))
    return true;

  // Verbatim with a long string should be emitted as an out-of-line declration.
  if (auto verbatim = dyn_cast<VerbatimExprOp>(op))
    if (verbatim.string().size() > 32)
      return true;

  auto *opBlock = op->getBlock();

  // If the parent op is not a module op, it is defined locally.
  bool isLocalDefinition = !isa_and_nonnull<HWModuleOp>(op->getParentOp());

  bool isDuplicatable = isDuplicatableExpression(op);

  // Scan the users of the operation to see if any of them need this to be
  // emitted out-of-line.
  for (auto user : op->getUsers()) {
    // If the op is defined locally and the user is in a different block, then
    // we emit this as an out-of-line declaration into its block and the user
    // can refer to it unless the operation is duplicatable.
    if (isLocalDefinition && user->getBlock() != opBlock && !isDuplicatable)
      return true;

    // Verilog bit selection is required by the standard to be:
    // "a vector, packed array, packed structure, parameter or concatenation".
    //
    // It cannot be an arbitrary expression, e.g. this is invalid:
    //     assign bar = {{a}, {b}, {c}, {d}}[idx];
    //
    // To handle these, we push the subexpression into a temporary.
    if (isa<ExtractOp, ArraySliceOp, ArrayGetOp, StructExtractOp>(user))
      if (op->getResult(0) == user->getOperand(0) && // ignore index operands.
          !isOkToBitSelectFrom(op->getResult(0)))
        return true;

    // Always blocks must have a name in their sensitivity list, not an expr.
    if (isa<AlwaysOp>(user) || isa<AlwaysFFOp>(user)) {
      // Anything other than a read of a wire must be out of line.
      if (auto read = dyn_cast<ReadInOutOp>(op))
        if (read.input().getDefiningOp<WireOp>() ||
            read.input().getDefiningOp<RegOp>())
          continue;
      return true;
    }
  }
  return false;
}

/// Return true if this expression should be emitted inline into any statement
/// that uses it.
static bool isExpressionEmittedInline(Operation *op) {
  // Never create a temporary which is only going to be assigned to an output
  // port.
  if (op->hasOneUse() && isa<hw::OutputOp>(*op->getUsers().begin()))
    return true;

  // If this operation has multiple uses, we can't generally inline it unless
  // the op is duplicatable.
  if (!op->getResult(0).hasOneUse() && !isDuplicatableExpression(op))
    return false;

  // If it isn't structurally possible to inline this expression, emit it out
  // of line.
  return !isExpressionUnableToInline(op);
}

namespace {
class NameCollector {
public:
  // This is information we keep track of for each wire/reg/interface
  // declaration we're going to emit.
  struct ValuesToEmitRecord {
    Value value;
    SmallString<8> typeString;
  };

  NameCollector(ModuleEmitter &moduleEmitter, ModuleNameManager &names)
      : moduleEmitter(moduleEmitter), names(names) {}

  // Scan operations in the specified block, collecting information about
  // those that need to be emitted out of line.
  void collectNames(Block &block);

  size_t getMaxDeclNameWidth() const { return maxDeclNameWidth; }
  size_t getMaxTypeWidth() const { return maxTypeWidth; }
  const SmallVectorImpl<ValuesToEmitRecord> &getValuesToEmit() const {
    return valuesToEmit;
  }

private:
  size_t maxDeclNameWidth = 0, maxTypeWidth = 0;
  SmallVector<ValuesToEmitRecord, 16> valuesToEmit;
  ModuleEmitter &moduleEmitter;
  ModuleNameManager &names;
};
} // namespace

void NameCollector::collectNames(Block &block) {
  bool isBlockProcedural = block.getParentOp()->hasTrait<ProceduralRegion>();

  SmallString<32> nameTmp;

  // Loop over all of the results of all of the ops.  Anything that defines a
  // value needs to be noticed.
  for (auto &op : block) {
    // Instances have a instance name to recognize but we don't need to look
    // at the result values and don't need to schedule them as valuesToEmit.
    if (auto instance = dyn_cast<InstanceOp>(op)) {
      names.addName(&op, getSymOpName(instance));
      continue;
    }
    if (auto interface = dyn_cast<InterfaceInstanceOp>(op)) {
      names.addName(interface.getResult(), getSymOpName(interface));
      continue;
    }

    bool isExpr = isVerilogExpression(&op);
    for (auto result : op.getResults()) {
      // If this is an expression emitted inline or unused, it doesn't need a
      // name.
      if (isExpr) {
        // If this expression is dead, or can be emitted inline, ignore it.
        if (result.use_empty() || isExpressionEmittedInline(&op))
          continue;

        // Remember that this expression should be emitted out of line.
        moduleEmitter.outOfLineExpressions.insert(&op);

        // Get an explicitly set name or try to infer a name from the structure
        // of the expression.
        names.addName(result,
                      moduleEmitter.inferStructuralNameForTemporary(result));

        // Don't measure or emit wires that are emitted inline (i.e. the wire
        // definition is emitted on the line of the expression instead of a
        // block at the top of the module).
        // Procedural blocks always emit out of line variable declarations,
        // because Verilog requires that they all be at the top of a block.
        if (!isBlockProcedural)
          continue;
      }

      // Measure this name and the length of its type, and ensure it is
      // emitted later.
      valuesToEmit.push_back(ValuesToEmitRecord{result, {}});
      auto &typeString = valuesToEmit.back().typeString;

      StringRef declName = getVerilogDeclWord(&op, moduleEmitter.state.options);
      maxDeclNameWidth = std::max(declName.size(), maxDeclNameWidth);

      // Convert the port's type to a string and measure it.
      {
        llvm::raw_svector_ostream stringStream(typeString);
        moduleEmitter.printPackedType(stripUnpackedTypes(result.getType()),
                                      stringStream, op.getLoc());
      }
      maxTypeWidth = std::max(typeString.size(), maxTypeWidth);
    }

    // Notice and renamify named declarations.
    if (isa<WireOp, RegOp, LocalParamOp>(op))
      names.addName(op.getResult(0), getSymOpName(&op));

    // Notice and renamify the labels on verification statements.
    if (isa<AssertOp, AssumeOp, CoverOp, AssertConcurrentOp, AssumeConcurrentOp,
            CoverConcurrentOp>(op)) {
      if (auto labelAttr = op.getAttrOfType<StringAttr>("label"))
        names.addName(&op, labelAttr);
    }

    // Recursively process any regions under the op iff this is a procedural
    // #ifdef region: we need to emit automatic logic values at the top of the
    // enclosing region.
    if (isa<IfDefProceduralOp>(op)) {
      for (auto &region : op.getRegions()) {
        if (!region.empty())
          collectNames(region.front());
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// StmtEmitter
//===----------------------------------------------------------------------===//

namespace {
/// This emits statement-related operations.
class StmtEmitter : public EmitterBase,
                    public hw::StmtVisitor<StmtEmitter, LogicalResult>,
                    public sv::Visitor<StmtEmitter, LogicalResult> {
public:
  /// Create an ExprEmitter for the specified module emitter, and keeping track
  /// of any emitted expressions in the specified set.
  StmtEmitter(ModuleEmitter &emitter, RearrangableOStream &outStream,
              ModuleNameManager &names)
      : EmitterBase(emitter.state, outStream), emitter(emitter),
        rearrangableStream(outStream), names(names) {}

  void emitStatement(Operation *op);
  void emitStatementBlock(Block &body);
  size_t getNumStatementsEmitted() const { return numStatementsEmitted; }

  /// Emit the declaration for the temporary operation. If the operation is not
  /// a constant, emit no initializer and no semicolon, e.g. `wire foo`, and
  /// return false. If the operation *is* a constant, also emit the initializer
  /// and semicolon, e.g. `localparam K = 1'h0;`, and return true.
  bool emitDeclarationForTemporary(Operation *op);

private:
  void collectNamesEmitDecls(Block &block);

  void
  emitExpression(Value exp, SmallPtrSet<Operation *, 8> &emittedExprs,
                 VerilogPrecedence parenthesizeIfLooserThan = LowestPrecedence);

  using StmtVisitor::visitStmt;
  using Visitor::visitSV;
  friend class hw::StmtVisitor<StmtEmitter, LogicalResult>;
  friend class sv::Visitor<StmtEmitter, LogicalResult>;

  // Visitor methods.
  LogicalResult visitUnhandledStmt(Operation *op) { return failure(); }
  LogicalResult visitInvalidStmt(Operation *op) { return failure(); }
  LogicalResult visitUnhandledSV(Operation *op) { return failure(); }
  LogicalResult visitInvalidSV(Operation *op) { return failure(); }

  LogicalResult emitNoop() {
    --numStatementsEmitted;
    return success();
  }

  LogicalResult visitSV(WireOp op) { return emitNoop(); }
  LogicalResult visitSV(RegOp op) { return emitNoop(); }
  LogicalResult visitSV(LocalParamOp op) { return emitNoop(); }
  LogicalResult visitSV(AssignOp op);
  LogicalResult visitSV(BPAssignOp op);
  LogicalResult visitSV(PAssignOp op);
  LogicalResult visitSV(ForceOp op);
  LogicalResult visitSV(ReleaseOp op);
  LogicalResult visitSV(AliasOp op);
  LogicalResult visitSV(InterfaceInstanceOp op);
  LogicalResult visitStmt(ProbeOp op);
  LogicalResult visitStmt(OutputOp op);
  LogicalResult visitStmt(InstanceOp op);
  LogicalResult visitStmt(TypeScopeOp op);
  LogicalResult visitStmt(TypedeclOp op);

  LogicalResult emitIfDef(Operation *op, StringRef cond);
  LogicalResult visitSV(IfDefOp op) { return emitIfDef(op, op.cond()); }
  LogicalResult visitSV(IfDefProceduralOp op) {
    return emitIfDef(op, op.cond());
  }
  LogicalResult visitSV(IfOp op);
  LogicalResult visitSV(AlwaysOp op);
  LogicalResult visitSV(AlwaysCombOp op);
  LogicalResult visitSV(AlwaysFFOp op);
  LogicalResult visitSV(InitialOp op);
  LogicalResult visitSV(CaseZOp op);
  LogicalResult visitSV(FWriteOp op);
  LogicalResult visitSV(VerbatimOp op);

  LogicalResult emitSimulationControlTask(Operation *op, StringRef taskName,
                                          Optional<unsigned> verbosity);
  LogicalResult visitSV(StopOp op);
  LogicalResult visitSV(FinishOp op);
  LogicalResult visitSV(ExitOp op);

  LogicalResult emitSeverityMessageTask(Operation *op, StringRef taskName,
                                        Optional<unsigned> verbosity,
                                        StringAttr message,
                                        ValueRange operands);
  LogicalResult visitSV(FatalOp op);
  LogicalResult visitSV(ErrorOp op);
  LogicalResult visitSV(WarningOp op);
  LogicalResult visitSV(InfoOp op);

  void emitAssertionLabel(Operation *op, StringRef opName);
  void emitAssertionMessage(StringAttr message, ValueRange args,
                            SmallPtrSet<Operation *, 8> &ops,
                            bool isConcurrent);
  template <typename Op>
  LogicalResult emitImmediateAssertion(Op op, StringRef opName);
  LogicalResult visitSV(AssertOp op);
  LogicalResult visitSV(AssumeOp op);
  LogicalResult visitSV(CoverOp op);
  template <typename Op>
  LogicalResult emitConcurrentAssertion(Op op, StringRef opName);
  LogicalResult visitSV(AssertConcurrentOp op);
  LogicalResult visitSV(AssumeConcurrentOp op);
  LogicalResult visitSV(CoverConcurrentOp op);

  LogicalResult visitSV(BindOp op);
  LogicalResult visitSV(InterfaceOp op);
  LogicalResult visitSV(InterfaceSignalOp op);
  LogicalResult visitSV(InterfaceModportOp op);
  LogicalResult visitSV(AssignInterfaceSignalOp op);
  void emitStatementExpression(Operation *op);

  void emitBlockAsStatement(Block *block,
                            SmallPtrSet<Operation *, 8> &locationOps,
                            StringRef multiLineComment = StringRef());

public:
  ModuleEmitter &emitter;

private:
  /// This is the current ostream we're emiting to, when we know it is a
  /// rearrangableStream.
  RearrangableOStream &rearrangableStream;

  /// Track the legalized names.
  ModuleNameManager &names;

  /// This is the index of the start of the current statement being emitted.
  RearrangableOStream::Cursor statementBeginning;

  /// This is the index of the end of the declaration region of the current
  /// 'begin' block, used to emit variable declarations.
  RearrangableOStream::Cursor blockDeclarationInsertPoint;
  unsigned blockDeclarationIndentLevel = INDENT_AMOUNT;

  /// This keeps track of the number of statements emitted, important for
  /// determining if we need to put out a begin/end marker in a block
  /// declaration.
  size_t numStatementsEmitted = 0;
};

} // end anonymous namespace

/// Emit the specified value as an expression.  If this is an inline-emitted
/// expression, we emit that expression, otherwise we emit a reference to the
/// already computed name.
///
void StmtEmitter::emitExpression(Value exp,
                                 SmallPtrSet<Operation *, 8> &emittedExprs,
                                 VerilogPrecedence parenthesizeIfLooserThan) {
  SmallVector<char, 128> exprBuffer;
  SmallVector<Operation *> tooLargeSubExpressions;
  ExprEmitter(emitter, exprBuffer, emittedExprs, tooLargeSubExpressions, names)
      .emitExpression(exp, parenthesizeIfLooserThan);
  os.write(exprBuffer.data(), exprBuffer.size());

  // It is possible that the emitted expression was too large to fit on a line
  // and needs to be split.  If so, the new subexpressions that need emitting
  // are put out into the the 'tooLargeSubExpressions' list.  Re-emit these at
  // the start of the current statement as their own stmt expressions.
  if (tooLargeSubExpressions.empty())
    return;

  // If we are working on a procedural statement, we need to emit the
  // declarations for each variable separately from the assignments to them.
  // Otherwise we just emit inline 'wire' declarations.
  RearrangableOStream::Cursor declStartCursor, declEndCursor;
  if (tooLargeSubExpressions[0]->getParentOp()->hasTrait<ProceduralRegion>()) {
    // Split the current segment to make sure the cursors we are about to create
    // don't get invalidated by statement reordering.
    rearrangableStream.splitCurrentSegment();

    // We're about to emit new things that we want to rearrange.
    declStartCursor = rearrangableStream.getCursor();

    // Emit the declarations into the stream.
    for (auto *expr : tooLargeSubExpressions) {
      // TODO: This results in a lot of things like this:
      //   automatic logic _tmp;
      //   automatic logic _tmp_0;
      // which could be turned into a comma separated list and properly
      // justified.  We could collect these and emit them all at once when
      // we recurse up to finishing off the procedural statement.  That would
      // also eliminate the need for blockDeclarationInsertPoint to be so
      // 'global'.
      if (!emitDeclarationForTemporary(expr))
        os << ";\n";
      ++numStatementsEmitted;
    }
    declEndCursor = rearrangableStream.getCursor();
  }

  /// Generating new statements will change `statementBeginning`, so make sure
  /// to keep track of what it is.
  auto prevStmtBeginning = statementBeginning;

  // Emit each stmt expression in turn.
  auto stmtStartCursor = rearrangableStream.getCursor();
  for (auto *expr : tooLargeSubExpressions) {
    ++numStatementsEmitted;
    emitStatementExpression(expr);
  }

  // Rearrange all of the generated text for these statements to before the
  // previous statement we were emitting, and restore statementBeginning to the
  // right place.
  statementBeginning = rearrangableStream.moveRangeBefore(
      prevStmtBeginning, stmtStartCursor, rearrangableStream.getCursor());
  if (!declStartCursor.isInvalid()) {
    // Scoop up all of the stuff we just emitted, and move it to the
    // blockDeclarationInsertPoint.
    blockDeclarationInsertPoint = rearrangableStream.moveRangeBefore(
        blockDeclarationInsertPoint, declStartCursor, declEndCursor);
  }
}

void StmtEmitter::emitStatementExpression(Operation *op) {
  // Know where the start of this statement is in case any out-of-band precursor
  // statements need to be emitted.
  statementBeginning = rearrangableStream.getCursor();

  // This is invoked for expressions that have a non-single use.  This could
  // either be because they are dead or because they have multiple uses.
  if (op->getResult(0).use_empty()) {
    indent() << "// Unused: ";
    --numStatementsEmitted;
  } else if (isZeroBitType(op->getResult(0).getType())) {
    indent() << "// Zero width: ";
    --numStatementsEmitted;
  } else if (op->getParentOp()->hasTrait<ProceduralRegion>()) {
    // Some expressions in procedural regions can be emitted inline into their
    // "automatic logic" or "localparam" definitions.  Don't redundantly emit
    // them.
    if (emitter.expressionsEmittedIntoDecl.count(op)) {
      --numStatementsEmitted;
      return;
    }
    indent() << names.getName(op->getResult(0)) << " = ";
  } else {
    if (emitDeclarationForTemporary(op))
      return;
    os << " = ";
  }

  // Emit the expression with a special precedence level so it knows to do a
  // "deep" emission even though there are multiple uses, not just emitting the
  // name.
  SmallPtrSet<Operation *, 8> emittedExprs;
  emitExpression(op->getResult(0), emittedExprs, ForceEmitMultiUse);
  os << ';';
  emitLocationInfoAndNewLine(emittedExprs);
}

LogicalResult StmtEmitter::visitSV(AssignOp op) {
  // prepare assigns wires to instance outputs, but these are logically handled
  // in the port binding list when outputing an instance.
  if (dyn_cast_or_null<InstanceOp>(op.src().getDefiningOp()))
    return success();

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "assign ";
  emitExpression(op.dest(), ops);
  os << " = ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(BPAssignOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent();
  emitExpression(op.dest(), ops);
  os << " = ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(PAssignOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent();
  emitExpression(op.dest(), ops);
  os << " <= ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(ForceOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "force ";
  emitExpression(op.dest(), ops);
  os << " = ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(ReleaseOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "release ";
  emitExpression(op.dest(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(AliasOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "alias ";
  llvm::interleave(
      op.getOperands(), os, [&](Value v) { emitExpression(v, ops); }, " = ");
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(InterfaceInstanceOp op) {
  StringRef prefix = "";
  if (op->hasAttr("doNotPrint")) {
    prefix = "// ";
    indent() << "// This interface is elsewhere emitted as a bind statement.\n";
  }

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  auto *interfaceOp = op.getReferencedInterface(&state.symbolCache);
  assert(interfaceOp && "InterfaceInstanceOp has invalid symbol that does not "
                        "point to an interface");

  auto verilogName = getSymOpName(interfaceOp);
  indent() << prefix << verilogName << " " << op.name() << "();";

  emitLocationInfoAndNewLine(ops);

  return success();
}

/// For OutputOp we put "assign" statements at the end of the Verilog module to
/// assign the module outputs to intermediate wires.
LogicalResult StmtEmitter::visitStmt(OutputOp op) {
  --numStatementsEmitted; // Count emitted statements manually.

  SmallPtrSet<Operation *, 8> ops;
  HWModuleOp parent = op->getParentOfType<HWModuleOp>();

  size_t operandIndex = 0;
  for (PortInfo port : parent.getPorts().outputs) {
    auto operand = op.getOperand(operandIndex);
    // Outputs that are set by the output port of an instance are handled
    // directly when the instance is emitted.
    if (operand.hasOneUse() &&
        dyn_cast_or_null<InstanceOp>(operand.getDefiningOp())) {
      ++operandIndex;
      continue;
    }

    ops.clear();
    ops.insert(op);
    indent();
    if (isZeroBitType(port.type))
      os << "// Zero width: ";
    os << "assign " << getPortVerilogName(parent, port) << " = ";
    emitExpression(operand, ops);
    os << ';';
    emitLocationInfoAndNewLine(ops);
    ++operandIndex;
    ++numStatementsEmitted;
  }
  return success();
}

LogicalResult StmtEmitter::visitStmt(TypeScopeOp op) {
  emitStatementBlock(*op.getBodyBlock());
  return success();
}

LogicalResult StmtEmitter::visitStmt(TypedeclOp op) {
  os << "typedef ";
  emitter.printPackedType(stripUnpackedTypes(op.type()), os, op.getLoc(),
                          false);
  os << ' ' << op.getPreferredName();
  emitter.printUnpackedTypePostfix(op.type(), os);
  os << ";\n";
  return success();
}

LogicalResult StmtEmitter::visitSV(FWriteOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "$fwrite(32'h80000002, \"";
  os.write_escaped(op.string());
  os << '"';

  for (auto operand : op.operands()) {
    os << ", ";
    emitExpression(operand, ops);
  }
  os << ");";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(VerbatimOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // Drop an extraneous \n off the end of the string if present.
  StringRef string = op.string();
  if (string.endswith("\n"))
    string = string.drop_back();

  // Emit each \n separated piece of the string with each piece properly
  // indented.  The convention is to not emit the \n so
  // emitLocationInfoAndNewLine can do that for the last line.
  bool isFirst = true;
  indent();

  // Emit each line of the string at a time.
  while (!string.empty()) {
    auto lhsRhs = string.split('\n');
    if (isFirst)
      isFirst = false;
    else {
      os << '\n';
      indent();
    }

    // Emit each chunk of the line.
    emitTextWithSubstitutions(
        lhsRhs.first, op, [&](Value operand) { emitExpression(operand, ops); },
        op.symbols(), names);
    string = lhsRhs.second;
  }

  emitLocationInfoAndNewLine(ops);

  // We don't know how many statements we emitted, so assume conservatively
  // that a lot got put out. This will make sure we get a begin/end block around
  // this.
  numStatementsEmitted += 2;
  return success();
}

/// Emit one of the simulation control tasks `$stop`, `$finish`, or `$exit`.
LogicalResult
StmtEmitter::emitSimulationControlTask(Operation *op, StringRef taskName,
                                       Optional<unsigned> verbosity) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << taskName;
  if (verbosity && *verbosity != 1)
    os << "(" << *verbosity << ")";
  os << ";";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(StopOp op) {
  return emitSimulationControlTask(op, "$stop", op.verbosity());
}

LogicalResult StmtEmitter::visitSV(FinishOp op) {
  return emitSimulationControlTask(op, "$finish", op.verbosity());
}

LogicalResult StmtEmitter::visitSV(ExitOp op) {
  return emitSimulationControlTask(op, "$exit", {});
}

/// Emit one of the severity message tasks `$fatal`, `$error`, `$warning`, or
/// `$info`.
LogicalResult StmtEmitter::emitSeverityMessageTask(Operation *op,
                                                   StringRef taskName,
                                                   Optional<unsigned> verbosity,
                                                   StringAttr message,
                                                   ValueRange operands) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << taskName;

  // In case we have a message to print, or the operation has an optional
  // verbosity and that verbosity is present, print the parenthesized parameter
  // list.
  if ((verbosity && *verbosity != 1) || message) {
    os << "(";

    // If the operation takes a verbosity, print it if it is set, or print the
    // default "1".
    if (verbosity)
      os << *verbosity;

    // Print the message and interpolation operands if present.
    if (message) {
      if (verbosity)
        os << ", ";
      os << "\"";
      os.write_escaped(message.getValue());
      os << "\"";
      for (auto operand : operands) {
        os << ", ";
        emitExpression(operand, ops);
      }
    }

    os << ")";
  }

  os << ";";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(FatalOp op) {
  return emitSeverityMessageTask(op, "$fatal", op.verbosity(), op.messageAttr(),
                                 op.operands());
}

LogicalResult StmtEmitter::visitSV(ErrorOp op) {
  return emitSeverityMessageTask(op, "$error", {}, op.messageAttr(),
                                 op.operands());
}

LogicalResult StmtEmitter::visitSV(WarningOp op) {
  return emitSeverityMessageTask(op, "$warning", {}, op.messageAttr(),
                                 op.operands());
}

LogicalResult StmtEmitter::visitSV(InfoOp op) {
  return emitSeverityMessageTask(op, "$info", {}, op.messageAttr(),
                                 op.operands());
}

/// Emit the `<label>:` portion of an immediate or concurrent verification
/// operation. If a label has been stored for the operation through
/// `addLegalName` in the pre-pass, that label is used. Otherwise, if the
/// `enforceVerifLabels` option is set, a temporary name for the operation is
/// picked and uniquified through `addName`.
void StmtEmitter::emitAssertionLabel(Operation *op, StringRef opName) {
  if (op->getAttrOfType<StringAttr>("label")) {
    os << names.getName(op) << ": ";
  } else if (state.options.enforceVerifLabels) {
    os << names.addName(op, opName) << ": ";
  }
}

/// Emit the optional ` else $error(...)` portion of an immediate or concurrent
/// verification operation.
void StmtEmitter::emitAssertionMessage(StringAttr message, ValueRange args,
                                       SmallPtrSet<Operation *, 8> &ops,
                                       bool isConcurrent = false) {
  if (!message)
    return;
  os << " else $error(\"";
  os.write_escaped(message.getValue());
  os << "\"";
  for (auto arg : args) {
    os << ", ";
    emitExpression(arg, ops);
  }
  os << ")";
}

template <typename Op>
LogicalResult StmtEmitter::emitImmediateAssertion(Op op, StringRef opName) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent();
  emitAssertionLabel(op, opName);
  os << opName;
  switch (op.defer()) {
  case DeferAssert::Immediate:
    break;
  case DeferAssert::Observed:
    os << " #0 ";
    break;
  case DeferAssert::Final:
    os << " final ";
    break;
  }
  os << "(";
  emitExpression(op.expression(), ops);
  os << ")";
  emitAssertionMessage(op.messageAttr(), op.operands(), ops);
  os << ";";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(AssertOp op) {
  return emitImmediateAssertion(op, "assert");
}

LogicalResult StmtEmitter::visitSV(AssumeOp op) {
  return emitImmediateAssertion(op, "assume");
}

LogicalResult StmtEmitter::visitSV(CoverOp op) {
  return emitImmediateAssertion(op, "cover");
}

template <typename Op>
LogicalResult StmtEmitter::emitConcurrentAssertion(Op op, StringRef opName) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent();
  emitAssertionLabel(op, opName);
  os << opName << " property (@(" << stringifyEventControl(op.event()) << " ";
  emitExpression(op.clock(), ops);
  os << ") ";
  emitExpression(op.property(), ops);
  os << ")";
  emitAssertionMessage(op.messageAttr(), op.operands(), ops, true);
  os << ";";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(AssertConcurrentOp op) {
  return emitConcurrentAssertion(op, "assert");
}

LogicalResult StmtEmitter::visitSV(AssumeConcurrentOp op) {
  return emitConcurrentAssertion(op, "assume");
}

LogicalResult StmtEmitter::visitSV(CoverConcurrentOp op) {
  return emitConcurrentAssertion(op, "cover");
}

LogicalResult StmtEmitter::emitIfDef(Operation *op, StringRef cond) {
  bool hasEmptyThen = op->getRegion(0).front().empty();
  if (hasEmptyThen)
    indent() << "`ifndef " << cond;
  else
    indent() << "`ifdef " << cond;

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  emitLocationInfoAndNewLine(ops);

  if (!hasEmptyThen)
    emitStatementBlock(op->getRegion(0).front());

  if (!op->getRegion(1).empty()) {
    if (!hasEmptyThen)
      indent() << "`else\n";
    emitStatementBlock(op->getRegion(1).front());
  }

  indent() << "`endif\n";

  // We don't know how many statements we emitted, so assume conservatively
  // that a lot got put out. This will make sure we get a begin/end block around
  // this.
  numStatementsEmitted += 2;
  return success();
}

/// Emit the body of a control flow statement that is surrounded by begin/end
/// markers if non-singular.  If the control flow construct is multi-line and
/// if multiLineComment is non-null, the string is included in a comment after
/// the 'end' to make it easier to associate.
void StmtEmitter::emitBlockAsStatement(Block *block,
                                       SmallPtrSet<Operation *, 8> &locationOps,
                                       StringRef multiLineComment) {

  // We don't know if we need to emit the begin until after we emit the body of
  // the block.  We can have multiple ops that fold together into one statement
  // (common in nested expressions feeding into a connect) or one apparently
  // simple set of operations that gets broken across multiple lines because
  // they are too long.
  //
  // Solve this by emitting the statements, determining if we need to
  // emit the begin, and if so, emit the begin retroactively.
  RearrangableOStream::Cursor beginInsertPoint = rearrangableStream.getCursor();
  emitLocationInfoAndNewLine(locationOps);

  // Change the blockDeclarationInsertPointIndex for the statements in this
  // block, and restore it back when we move on to code after the block.
  llvm::SaveAndRestore<RearrangableOStream::Cursor> x(
      blockDeclarationInsertPoint, rearrangableStream.getCursor());
  llvm::SaveAndRestore<unsigned> x2(blockDeclarationIndentLevel,
                                    state.currentIndent + INDENT_AMOUNT);

  auto numEmittedBefore = getNumStatementsEmitted();
  emitStatementBlock(*block);

  // If we emitted exactly one statement, then we are done.
  if (getNumStatementsEmitted() - numEmittedBefore == 1)
    return;

  // Otherwise we emit the begin and end logic.
  rearrangableStream.insertLiteral(beginInsertPoint, " begin");

  indent() << "end";
  if (!multiLineComment.empty())
    os << " // " << multiLineComment;
  os << '\n';
}

LogicalResult StmtEmitter::visitSV(IfOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "if (";

  // If we have an else and and empty then block, emit an inverted condition.
  if (!op.hasElse() || !op.getThenBlock()->empty()) {
    // Normal emission.
    emitExpression(op.cond(), ops);
    os << ')';
    emitBlockAsStatement(op.getThenBlock(), ops);
    if (op.hasElse()) {
      indent() << "else";
      emitBlockAsStatement(op.getElseBlock(), ops);
    }
  } else {
    // inverted condition.
    os << '!';
    emitExpression(op.cond(), ops, Unary);
    os << ')';
    emitBlockAsStatement(op.getElseBlock(), ops);
  }

  // We count if as multiple statements to make sure it is always surrounded by
  // a begin/end so we don't get if/else confusion in cases like this:
  // if (cond)
  //   if (otherCond)    // This should force a begin!
  //     stmt
  // else                // Goes with the outer if!
  //   thing;
  ++numStatementsEmitted;
  return success();
}

LogicalResult StmtEmitter::visitSV(AlwaysOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  auto printEvent = [&](AlwaysOp::Condition cond) {
    os << stringifyEventControl(cond.event) << ' ';
    emitExpression(cond.value, ops);
  };

  switch (op.getNumConditions()) {
  case 0:
    indent() << "always @*";
    break;
  case 1:
    indent() << "always @(";
    printEvent(op.getCondition(0));
    os << ')';
    break;
  default:
    indent() << "always @(";
    printEvent(op.getCondition(0));
    for (size_t i = 1, e = op.getNumConditions(); i != e; ++i) {
      os << " or ";
      printEvent(op.getCondition(i));
    }
    os << ')';
    break;
  }

  // Build the comment string, leave out the signal expressions (since they
  // can be large).
  std::string comment;
  if (op.getNumConditions() == 0) {
    comment = "always @*";
  } else {
    comment = "always @(";
    llvm::interleave(
        op.events(),
        [&](Attribute eventAttr) {
          auto event = EventControl(eventAttr.cast<IntegerAttr>().getInt());
          comment += stringifyEventControl(event);
        },
        [&]() { comment += ", "; });
    comment += ')';
  }

  emitBlockAsStatement(op.getBodyBlock(), ops, comment);
  return success();
}

LogicalResult StmtEmitter::visitSV(AlwaysCombOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  StringRef opString = "always_comb";
  if (state.options.noAlwaysComb)
    opString = "always @(*)";

  indent() << opString;
  emitBlockAsStatement(op.getBodyBlock(), ops, opString);
  return success();
}

LogicalResult StmtEmitter::visitSV(AlwaysFFOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "always_ff @(" << stringifyEventControl(op.clockEdge()) << " ";
  emitExpression(op.clock(), ops);
  if (op.resetStyle() == ResetType::AsyncReset) {
    os << " or " << stringifyEventControl(*op.resetEdge()) << " ";
    emitExpression(op.reset(), ops);
  }
  os << ')';

  // Build the comment string, leave out the signal expressions (since they
  // can be large).
  std::string comment;
  comment += "always_ff @(";
  comment += stringifyEventControl(op.clockEdge());
  if (op.resetStyle() == ResetType::AsyncReset) {
    comment += " or ";
    comment += stringifyEventControl(*op.resetEdge());
  }
  comment += ')';

  if (op.resetStyle() == ResetType::NoReset)
    emitBlockAsStatement(op.getBodyBlock(), ops, comment);
  else {
    os << " begin";
    emitLocationInfoAndNewLine(ops);
    addIndent();

    indent() << "if (";
    // Negative edge async resets need to invert the reset condition.  This is
    // noted in the op description.
    if (op.resetStyle() == ResetType::AsyncReset &&
        *op.resetEdge() == EventControl::AtNegEdge)
      os << "!";
    emitExpression(op.reset(), ops);
    os << ')';
    emitBlockAsStatement(op.getResetBlock(), ops);
    indent() << "else";
    emitBlockAsStatement(op.getBodyBlock(), ops);
    reduceIndent();

    indent() << "end";
    os << " // " << comment;
    os << '\n';
  }
  return success();
}

LogicalResult StmtEmitter::visitSV(InitialOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "initial";
  emitBlockAsStatement(op.getBodyBlock(), ops, "initial");
  return success();
}

LogicalResult StmtEmitter::visitSV(CaseZOp op) {
  SmallPtrSet<Operation *, 8> ops, emptyOps;
  ops.insert(op);

  indent() << "casez (";
  emitExpression(op.cond(), ops);
  os << ')';
  emitLocationInfoAndNewLine(ops);

  addIndent();
  for (auto caseInfo : op.getCases()) {
    auto pattern = caseInfo.pattern;

    if (pattern.isDefault())
      indent() << "default";
    else {
      // TODO: We could emit in hex if/when the size is a multiple of 4 and
      // there are no x's crossing nibble boundaries.
      indent() << pattern.getWidth() << "'b";
      for (size_t bit = 0, e = pattern.getWidth(); bit != e; ++bit)
        os << getLetter(pattern.getBit(e - bit - 1), /*isVerilog*/ true);
    }
    os << ":";
    emitBlockAsStatement(caseInfo.block, emptyOps);
  }

  reduceIndent();
  indent() << "endcase";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitStmt(InstanceOp op) {
  bool doNotPrint = op->hasAttr("doNotPrint");
  if (doNotPrint) {
    indent() << "/* This instance is elsewhere emitted as a bind statement.\n";
    addIndent();
  }

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // Use the specified name or the symbol name as appropriate.
  auto *moduleOp = op.getReferencedModule(&state.symbolCache);
  assert(moduleOp && "Invalid IR");
  indent() << getVerilogModuleName(moduleOp);

  // If this is a parameterized module, then emit the parameters.
  if (!op.parameters().empty()) {
    // All the parameters may be defaulted -- don't print out an empty list if
    // so.
    bool printed = false;
    for (auto params :
         llvm::zip(op.parameters(),
                   moduleOp->getAttrOfType<ArrayAttr>("parameters"))) {
      auto param = std::get<0>(params).cast<ParamDeclAttr>();
      auto modParam = std::get<1>(params).cast<ParamDeclAttr>();
      // Ignore values that line up with their default.
      if (param.getValue() == modParam.getValue())
        continue;

      // Handle # if this is the first parameter we're printing.
      if (!printed) {
        os << " #(\n";
        printed = true;
      } else {
        os << ",\n";
      }
      os.indent(state.currentIndent + INDENT_AMOUNT) << '.';
      os << state.globalNames.getParameterVerilogName(moduleOp,
                                                      param.getName());
      os << '(';
      emitter.printParamValue(param.getValue(), os, [&]() {
        return op->emitOpError("invalid instance parameter '")
               << param.getName().getValue() << "' value";
      });
      os << ')';
    }
    if (printed) {
      os << '\n';
      indent() << ')';
    }
  }

  os << ' ' << names.getName(op) << " (";

  SmallVector<PortInfo> portInfo = getAllModulePortInfos(op);

  // Get the max port name length so we can align the '('.
  size_t maxNameLength = 0;
  for (auto &elt : portInfo) {
    maxNameLength = std::max(maxNameLength, elt.getName().size());
  }

  auto getWireForValue = [&](Value result) {
    return result.getUsers().begin()->getOperand(0);
  };

  // Emit the argument and result ports.
  auto opArgs = op.inputs();
  auto opResults = op.getResults();
  bool isFirst = true; // True until we print a port.
  bool isZeroWidth = false;
  SmallVector<Value, 32> portValues;
  for (auto &elt : portInfo) {
    // Figure out which value we are emitting.
    portValues.push_back(elt.isOutput() ? opResults[elt.argNum]
                                        : opArgs[elt.argNum]);
  }

  for (size_t portNum = 0, e = portValues.size(); portNum < e; ++portNum) {
    // Figure out which value we are emitting.
    auto &elt = portInfo[portNum];
    Value portVal = portValues[portNum];
    isZeroWidth = isZeroBitType(portVal.getType());

    // Decide if we should print a comma.  We can't do this if we're the first
    // port or if all the subsequent ports are zero width.
    if (!isFirst) {
      bool shouldPrintComma = true;
      if (isZeroWidth) {
        shouldPrintComma = false;
        for (size_t i = (&elt - portInfo.data()) + 1, e = portInfo.size();
             i != e; ++i)
          if (!isZeroBitType(portValues[i].getType())) {
            shouldPrintComma = true;
            break;
          }
      }

      if (shouldPrintComma)
        os << ',';
    }
    emitLocationInfoAndNewLine(ops);

    // Emit the port's name.
    indent();
    if (!isZeroWidth) {
      // If this is a real port we're printing, then it isn't the first one. Any
      // subsequent ones will need a comma.
      isFirst = false;
      os << "  ";
    } else {
      // We comment out zero width ports, so their presence and initializer
      // expressions are still emitted textually.
      os << "//";
    }

    os << '.' << getPortVerilogName(moduleOp, elt);
    os.indent(maxNameLength - elt.getName().size()) << " (";

    // Emit the value as an expression.
    ops.clear();

    // Output ports that are not connected to single use output ports were
    // lowered to wire.
    OutputOp output;
    if (!elt.isOutput()) {
      emitExpression(portVal, ops);
    } else if (portVal.hasOneUse() &&
               (output = dyn_cast_or_null<OutputOp>(
                    portVal.getUses().begin()->getOwner()))) {
      // If this is directly using the output port of the containing module,
      // just specify that directly so we avoid a temporary wire.
      size_t outputPortNo = portVal.getUses().begin()->getOperandNumber();
      auto containingModule = emitter.currentModuleOp;
      os << getPortVerilogName(containingModule,
                               containingModule.getOutputPort(outputPortNo));
    } else {
      portVal = getWireForValue(portVal);
      emitExpression(portVal, ops);
    }
    os << ')';
  }
  if (!isFirst || isZeroWidth) {
    emitLocationInfoAndNewLine(ops);
    ops.clear();
    indent();
  }
  os << ");";
  emitLocationInfoAndNewLine(ops);
  if (doNotPrint) {
    reduceIndent();
    indent() << "*/\n";
  }
  return success();
}

// Probes only exist to provide naming to values.  They are handled in
// the naming prepass.
LogicalResult StmtEmitter::visitStmt(ProbeOp op) { return success(); }

// This may be called in the top-level, not just in an hw.module.  Thus we can't
// use the name map to find expression names for arguments to the instance, nor
// do we need to emit subexpressions.  Prepare pass, which has run for all
// modules prior to this, has ensured that all arguments are bound to wires,
// regs, or ports, with legalized names, so we can lookup up the names through
// the IR.
LogicalResult StmtEmitter::visitSV(BindOp op) {
  emitter.emitBind(op);
  return success();
}

LogicalResult StmtEmitter::visitSV(InterfaceOp op) {
  os << "interface " << getSymOpName(op) << ";\n";
  // FIXME: Don't emit the body of this as general statements, they aren't!
  emitStatementBlock(*op.getBodyBlock());
  os << "endinterface\n\n";
  return success();
}

LogicalResult StmtEmitter::visitSV(InterfaceSignalOp op) {
  indent();
  emitter.printPackedType(stripUnpackedTypes(op.type()), os, op->getLoc(),
                          false);
  os << ' ' << getSymOpName(op);
  emitter.printUnpackedTypePostfix(op.type(), os);
  os << ";\n";
  return success();
}

LogicalResult StmtEmitter::visitSV(InterfaceModportOp op) {
  indent() << "modport " << getSymOpName(op) << '(';

  llvm::interleaveComma(op.ports(), os, [&](const Attribute &portAttr) {
    auto port = portAttr.cast<ModportStructAttr>();
    os << port.direction().getValue() << ' ';
    auto signalDecl = state.symbolCache.getDefinition(port.signal());
    os << getSymOpName(signalDecl);
  });

  os << ");\n";
  return success();
}

LogicalResult StmtEmitter::visitSV(AssignInterfaceSignalOp op) {
  SmallPtrSet<Operation *, 8> emitted;
  indent() << "assign ";
  emitExpression(op.iface(), emitted);
  os << '.' << op.signalName() << " = ";
  emitExpression(op.rhs(), emitted);
  os << ";\n";
  return success();
}

void StmtEmitter::emitStatement(Operation *op) {
  // Expressions may either be ignored or emitted as an expression statements.
  if (isVerilogExpression(op)) {
    if (emitter.outOfLineExpressions.count(op)) {
      ++numStatementsEmitted;
      emitStatementExpression(op);
    }
    return;
  }

  ++numStatementsEmitted;

  // Know where the start of this statement is in case any out-of-band precursor
  // statements need to be emitted.
  statementBeginning = rearrangableStream.getCursor();

  // Handle HW statements.
  if (succeeded(dispatchStmtVisitor(op)))
    return;

  // Handle SV Statements.
  if (succeeded(dispatchSVVisitor(op)))
    return;

  emitOpError(op, "cannot emit this operation to Verilog");
  indent() << "unknown MLIR operation " << op->getName().getStringRef() << "\n";
}

/// Given an operation corresponding to a VerilogExpression, determine whether
/// it is safe to emit inline into a 'localparam' or 'automatic logic' varaible
/// initializer in a procedural region.
///
/// We can't emit exprs inline when they refer to something else that can't be
/// emitted inline, when they're in a general #ifdef region,
static bool
isExpressionEmittedInlineIntoProceduralDeclaration(Operation *op,
                                                   StmtEmitter &stmtEmitter) {
  if (!isVerilogExpression(op))
    return false;

  // If the expression exists in an #ifdef region, then bail.  Emitting it
  // inline would cause it to be executed unconditionally, because the
  // declarations are outside the #ifdef.
  if (isa<IfDefProceduralOp>(op->getParentOp()))
    return false;

  // This expression tree can be emitted into the initializer if all leaf
  // references are safe to refer to from here.  They are only safe if they are
  // defined in an enclosing scope (guaranteed to already be live by now) or if
  // they are defined in this block and already emitted to an inline automatic
  // logic variable.
  SmallVector<Value, 8> exprsToScan(op->getOperands());

  // This loop is guaranteed to terminate because we're only scanning up
  // single-use expressions and other things that 'isExpressionEmittedInline'
  // returns success for.  Cycles won't get in here.
  while (!exprsToScan.empty()) {
    Operation *expr = exprsToScan.pop_back_val().getDefiningOp();
    if (!expr)
      continue; // Ports are always safe to reference.

    // If this is an internal node in the expression tree, process its operands.
    if (isExpressionEmittedInline(expr)) {
      exprsToScan.append(expr->getOperands().begin(),
                         expr->getOperands().end());
      continue;
    }

    // Otherwise, this isn't an inlinable expression.  If it is defined outside
    // this block, then it is live-in.
    if (expr->getBlock() != op->getBlock())
      continue;

    // Otherwise, if it is defined in this block then it is only ok to reference
    // if it has already been emitted into an automatic logic.
    if (!stmtEmitter.emitter.expressionsEmittedIntoDecl.count(expr))
      return false;
  }

  return true;
}

/// Emit the declaration for the temporary operation. If the operation is not
/// a constant, emit no initializer and no semicolon, e.g. `wire foo`, and
/// return false. If the operation *is* a constant, also emit the initializer
/// and semicolon, e.g. `localparam K = 1'h0`, and return true.
bool StmtEmitter::emitDeclarationForTemporary(Operation *op) {
  StringRef declWord = getVerilogDeclWord(op, state.options);

  os.indent(blockDeclarationIndentLevel) << declWord;
  if (!declWord.empty())
    os << ' ';
  if (emitter.printPackedType(stripUnpackedTypes(op->getResult(0).getType()),
                              os, op->getLoc()))
    os << ' ';
  os << names.getName(op->getResult(0));

  // Emit the initializer expression for this declaration inline if safe.
  if (!isExpressionEmittedInlineIntoProceduralDeclaration(op, *this))
    return false;

  // Keep track that we emitted this.
  emitter.expressionsEmittedIntoDecl.insert(op);

  os << " = ";
  SmallPtrSet<Operation *, 8> emittedExprs;
  emitExpression(op->getResult(0), emittedExprs, ForceEmitMultiUse);
  os << ';';
  emitLocationInfoAndNewLine(emittedExprs);
  return true;
}

void StmtEmitter::collectNamesEmitDecls(Block &block) {
  // In the first pass, we fill in the symbol table, calculate the max width
  // of the declaration words and the max type width.
  NameCollector collector(emitter, names);
  collector.collectNames(block);

  auto &valuesToEmit = collector.getValuesToEmit();
  if (valuesToEmit.empty())
    return;

  size_t maxDeclNameWidth = collector.getMaxDeclNameWidth();
  size_t maxTypeWidth = collector.getMaxTypeWidth();

  if (maxTypeWidth > 0) // add a space if any type exists
    maxTypeWidth += 1;

  SmallPtrSet<Operation *, 8> opsForLocation;

  // Okay, now that we have measured the things to emit, emit the things.
  for (const auto &record : valuesToEmit) {
    statementBeginning = rearrangableStream.getCursor();

    // We have two different sorts of things that we proactively emit:
    // declarations (wires, regs, localpamarams, etc) and expressions that
    // cannot be emitted inline (e.g. because of limitations around subscripts).
    auto *op = record.value.getDefiningOp();
    opsForLocation.clear();
    opsForLocation.insert(op);

    // Emit the leading word, like 'wire' or 'reg'.
    auto type = record.value.getType();
    auto word = getVerilogDeclWord(op, state.options);
    if (!isZeroBitType(type)) {
      indent() << word;
      auto extraIndent = word.empty() ? 0 : 1;
      os.indent(maxDeclNameWidth - word.size() + extraIndent);
    } else {
      indent() << "// Zero width: " << word << ' ';
    }

    // Emit the type.
    os << record.typeString;
    if (record.typeString.size() < maxTypeWidth)
      os.indent(maxTypeWidth - record.typeString.size());

    // Emit the name.
    os << names.getName(record.value);

    // Print out any array subscripts or other post-name stuff.
    emitter.printUnpackedTypePostfix(type, os);

    if (auto localparam = dyn_cast<LocalParamOp>(op)) {
      os << " = ";
      emitter.printParamValue(localparam.value(), os, [&]() {
        return op->emitOpError("invalid localparam value");
      });
    }

    // Constants carry their assignment directly in the declaration.
    if (isExpressionEmittedInlineIntoProceduralDeclaration(op, *this)) {
      os << " = ";
      emitExpression(op->getResult(0), opsForLocation, ForceEmitMultiUse);

      // Remember that we emitted this inline into the declaration so we don't
      // emit it and we know the value is available for other declaration
      // expressions who might want to reference it.
      emitter.expressionsEmittedIntoDecl.insert(op);
    }

    os << ';';
    emitLocationInfoAndNewLine(opsForLocation);
    ++numStatementsEmitted;

    // If any sub-expressions are too large to fit on a line and need a
    // temporary declaration, put it after the already-emitted declarations.
    // This is important to maintain incrementally after each statement, because
    // each statement can generate spills when they are overly-long.
    blockDeclarationInsertPoint = rearrangableStream.getCursor();
    blockDeclarationIndentLevel = state.currentIndent;
  }

  os << '\n';
}

void StmtEmitter::emitStatementBlock(Block &body) {
  addIndent();

  // Build up the symbol table for all of the values that need names in the
  // module.  #ifdef's in procedural regions are special because local variables
  // are all emitted at the top of their enclosing blocks.
  if (!isa<IfDefProceduralOp>(body.getParentOp()))
    collectNamesEmitDecls(body);

  // Emit the body.
  for (auto &op : body) {
    emitStatement(&op);
  }

  reduceIndent();
}

void ModuleEmitter::emitStatement(Operation *op) {
  RearrangableOStream outputBuffer;
  ModuleNameManager names;
  StmtEmitter(*this, outputBuffer, names).emitStatement(op);
  outputBuffer.print(os);
}

//===----------------------------------------------------------------------===//
// Module Driver
//===----------------------------------------------------------------------===//

void ModuleEmitter::emitHWExternModule(HWModuleExternOp module) {
  auto verilogName = module.getVerilogModuleNameAttr();
  os << "// external module " << verilogName.getValue() << "\n\n";
}

void ModuleEmitter::emitHWGeneratedModule(HWModuleGeneratedOp module) {
  auto verilogName = module.getVerilogModuleNameAttr();
  os << "// external generated module " << verilogName.getValue() << "\n\n";
}

// This may be called in the top-level, not just in an hw.module.  Thus we can't
// use the name map to find expression names for arguments to the instance, nor
// do we need to emit subexpressions.  Prepare pass, which has run for all
// modules prior to this, has ensured that all arguments are bound to wires,
// regs, or ports, with legalized names, so we can lookup up the names through
// the IR.
void ModuleEmitter::emitBind(BindOp op) {
  InstanceOp inst = op.getReferencedInstance(&state.symbolCache);

  HWModuleOp parentMod = inst->getParentOfType<hw::HWModuleOp>();
  auto parentVerilogName = getVerilogModuleNameAttr(parentMod);

  Operation *childMod = inst.getReferencedModule(&state.symbolCache);
  auto childVerilogName = getVerilogModuleNameAttr(childMod);

  indent() << "bind " << parentVerilogName.getValue() << " "
           << childVerilogName.getValue() << ' ' << getSymOpName(inst) << " (";

  ModulePortInfo parentPortInfo = parentMod.getPorts();
  SmallVector<PortInfo> childPortInfo = getAllModulePortInfos(inst);

  // Get the max port name length so we can align the '('.
  size_t maxNameLength = 0;
  for (auto &elt : childPortInfo) {
    auto portName = getPortVerilogName(childMod, elt);
    elt.name = Builder(inst.getContext()).getStringAttr(portName);
    maxNameLength = std::max(maxNameLength, elt.getName().size());
  }

  // Emit the argument and result ports.
  auto opArgs = inst.inputs();
  auto opResults = inst.getResults();
  bool isFirst = true; // True until we print a port.
  for (auto &elt : childPortInfo) {
    // Figure out which value we are emitting.
    Value portVal = elt.isOutput() ? opResults[elt.argNum] : opArgs[elt.argNum];
    bool isZeroWidth = isZeroBitType(elt.type);

    // Decide if we should print a comma.  We can't do this if we're the first
    // port or if all the subsequent ports are zero width.
    if (!isFirst) {
      bool shouldPrintComma = true;
      if (isZeroWidth) {
        shouldPrintComma = false;
        for (size_t i = (&elt - childPortInfo.data()) + 1,
                    e = childPortInfo.size();
             i != e; ++i)
          if (!isZeroBitType(childPortInfo[i].type)) {
            shouldPrintComma = true;
            break;
          }
      }

      if (shouldPrintComma)
        os << ',';
    }
    os << '\n';

    // Emit the port's name.
    indent();
    if (!isZeroWidth) {
      // If this is a real port we're printing, then it isn't the first one. Any
      // subsequent ones will need a comma.
      isFirst = false;
      os << "  ";
    } else {
      // We comment out zero width ports, so their presence and initializer
      // expressions are still emitted textually.
      os << "//";
    }

    os << '.' << elt.getName();
    os.indent(maxNameLength - elt.getName().size()) << " (";

    // Emit the value as an expression.
    auto name = getNameRemotely(portVal, parentPortInfo, parentMod);
    assert(!name.empty() && "bind port connection must have a name");
    os << name << ')';
  }
  if (!isFirst) {
    os << '\n';
    indent();
  }
  os << ");\n";
}

/// Return the name of a value in a remote module to be used in a `bind`
/// statement. This function examines the remote module `remoteModule` and looks
/// up the corresponding name in the provide `GlobalNameTable`. This requires
/// that all names this function may be asked to lookup have been legalized and
/// added to that name table.
StringRef ModuleEmitter::getNameRemotely(Value value,
                                         const ModulePortInfo &modulePorts,
                                         HWModuleOp remoteModule) {
  if (auto barg = value.dyn_cast<BlockArgument>())
    return getPortVerilogName(remoteModule,
                              modulePorts.inputs[barg.getArgNumber()]);

  Operation *valueOp = value.getDefiningOp();

  // Handle wires/registers, likely as instance inputs.
  if (auto readinout = dyn_cast<ReadInOutOp>(valueOp)) {
    auto *wireInput = readinout.input().getDefiningOp();
    if (!wireInput)
      return {};
    if (isa<WireOp, RegOp>(wireInput))
      return getSymOpName(wireInput);
  }

  // Handle values being driven onto wires, likely as instance outputs.
  if (isa<InstanceOp>(valueOp)) {
    for (auto &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (!isa<AssignOp>(user) || use.getOperandNumber() != 1)
        continue;
      Value drivenOnto = user->getOperand(0);
      Operation *drivenOntoOp = drivenOnto.getDefiningOp();
      if (isa<WireOp, RegOp>(drivenOntoOp))
        return getSymOpName(drivenOntoOp);
    }
  }

  // Handle local parameters.
  if (isa<LocalParamOp>(valueOp))
    return getSymOpName(valueOp);
  return {};
}

void ModuleEmitter::emitBindInterface(BindInterfaceOp bind) {
  auto instance = bind.getReferencedInstance(&state.symbolCache);
  auto instantiator = instance->getParentOfType<HWModuleOp>().getName();
  auto *interface = bind->getParentOfType<ModuleOp>().lookupSymbol(
      instance.getInterfaceType().getInterface());
  os << "bind " << instantiator << " "
     << cast<InterfaceOp>(*interface).sym_name() << " "
     << getSymOpName(instance) << " (.*);\n\n";
}

void ModuleEmitter::emitHWModule(HWModuleOp module) {
  currentModuleOp = module;

  ModuleNameManager names;

  // Add all the ports to the name table so wires etc don't reuse the name.
  SmallVector<PortInfo> portInfo = module.getAllPorts();
  for (auto &port : portInfo) {
    StringRef name = getPortVerilogName(module, port);
    Value value;
    if (!port.isOutput())
      value = module.getArgument(port.argNum);
    names.addName(value, name);
  }

  // Add all parameters to the name table.
  for (auto param : module.parameters()) {
    // Add the name to the name table so any conflicting wires are renamed.
    StringRef verilogName = state.globalNames.getParameterVerilogName(
        module, param.cast<ParamDeclAttr>().getName());
    names.addName(nullptr, verilogName);
  }

  SmallPtrSet<Operation *, 8> moduleOpSet;
  moduleOpSet.insert(module);

  emitComment(module.commentAttr());

  os << "module " << getVerilogModuleName(module);

  // If we have any parameters, print them on their own line.
  if (!module.parameters().empty()) {
    os << "\n  #(";

    auto printParamType = [&](Type type, Attribute defaultValue,
                              SmallString<8> &result) {
      result.clear();
      llvm::raw_svector_ostream sstream(result);

      // If there is a default value like "32" then just print without type at
      // all.
      if (defaultValue) {
        if (auto intAttr = defaultValue.dyn_cast<IntegerAttr>())
          if (intAttr.getValue().getBitWidth() == 32)
            return;
        if (auto fpAttr = defaultValue.dyn_cast<FloatAttr>())
          if (fpAttr.getType().isF64())
            return;
        if (defaultValue.isa<StringAttr>())
          return;
      }

      // Classic Verilog parser don't allow a type in the parameter declaration.
      // For compatibility with them, we omit the type when it is implicit based
      // on its initializer value, and print the type commented out when it is
      // a 32-bit "integer" parameter.
      if (auto intType = type_dyn_cast<IntegerType>(type))
        if (intType.getWidth() == 32) {
          sstream << "/*integer*/";
          return;
        }

      printPackedType(type, sstream, module->getLoc(),
                      /*implicitIntType=*/true,
                      // Print single-bit values as explicit `[0:0]` type.
                      /*singleBitDefaultType=*/false);
    };

    // Determine the max width of the parameter types so things are lined up.
    size_t maxTypeWidth = 0;
    SmallString<8> scratch;
    for (auto param : module.parameters()) {
      auto paramAttr = param.cast<ParamDeclAttr>();
      // Measure the type length by printing it to a temporary string.
      printParamType(paramAttr.getType().getValue(), paramAttr.getValue(),
                     scratch);
      maxTypeWidth = std::max(scratch.size(), maxTypeWidth);
    }

    if (maxTypeWidth > 0) // add a space if any type exists.
      maxTypeWidth += 1;

    llvm::interleave(
        module.parameters(), os,
        [&](Attribute param) {
          auto paramAttr = param.cast<ParamDeclAttr>();
          auto defaultValue = paramAttr.getValue(); // may be null if absent.
          os << "parameter ";
          printParamType(paramAttr.getType().getValue(), defaultValue, scratch);
          os << scratch;
          if (scratch.size() < maxTypeWidth)
            os.indent(maxTypeWidth - scratch.size());

          os << state.globalNames.getParameterVerilogName(module,
                                                          paramAttr.getName());

          if (defaultValue) {
            os << " = ";
            printParamValue(defaultValue, os, [&]() {
              return module->emitError("parameter '")
                     << paramAttr.getName().getValue() << "' has invalid value";
            });
          }
        },
        ",\n    ");
    os << ") ";
  }

  os << '(';
  if (!portInfo.empty())
    emitLocationInfoAndNewLine(moduleOpSet);

  // Determine the width of the widest type we have to print so everything
  // lines up nicely.
  bool hasOutputs = false, hasZeroWidth = false;
  size_t maxTypeWidth = 0, lastNonZeroPort = -1;
  SmallVector<SmallString<8>, 16> portTypeStrings;

  for (size_t i = 0, e = portInfo.size(); i < e; ++i) {
    auto port = portInfo[i];
    hasOutputs |= port.isOutput();
    hasZeroWidth |= isZeroBitType(port.type);
    if (!isZeroBitType(port.type))
      lastNonZeroPort = i;

    // Convert the port's type to a string and measure it.
    portTypeStrings.push_back({});
    {
      llvm::raw_svector_ostream stringStream(portTypeStrings.back());
      printPackedType(stripUnpackedTypes(port.type), stringStream,
                      module->getLoc());
    }

    maxTypeWidth = std::max(portTypeStrings.back().size(), maxTypeWidth);
  }

  if (maxTypeWidth > 0) // add a space if any type exists
    maxTypeWidth += 1;

  addIndent();

  for (size_t portIdx = 0, e = portInfo.size(); portIdx != e;) {
    size_t startOfLinePos = os.tell();

    indent();
    // Emit the arguments.
    auto portType = portInfo[portIdx].type;
    bool isZeroWidth = false;
    if (hasZeroWidth) {
      isZeroWidth = isZeroBitType(portType);
      os << (isZeroWidth ? "// " : "   ");
    }

    PortDirection thisPortDirection = portInfo[portIdx].direction;
    switch (thisPortDirection) {
    case PortDirection::OUTPUT:
      os << "output ";
      break;
    case PortDirection::INPUT:
      os << (hasOutputs ? "input  " : "input ");
      break;
    case PortDirection::INOUT:
      os << (hasOutputs ? "inout  " : "inout ");
      break;
    }

    // Emit the type.
    os << portTypeStrings[portIdx];
    if (portTypeStrings[portIdx].size() < maxTypeWidth)
      os.indent(maxTypeWidth - portTypeStrings[portIdx].size());

    // Emit the name.
    os << getPortVerilogName(module, portInfo[portIdx]);
    printUnpackedTypePostfix(portType, os);
    ++portIdx;

    auto lineLength = state.options.emittedLineLength;

    // If we have any more ports with the same types and the same direction,
    // emit them in a list on the same line.
    while (portIdx != e && portInfo[portIdx].direction == thisPortDirection &&
           stripUnpackedTypes(portType) ==
               stripUnpackedTypes(portInfo[portIdx].type)) {
      // Don't exceed our preferred line length.
      StringRef name = getPortVerilogName(module, portInfo[portIdx]);
      if (os.tell() + 2 + name.size() - startOfLinePos >
          // We use "-2" here because we need a trailing comma or ); for the
          // decl.
          lineLength - 2)
        break;

      // Append this to the running port decl.
      os << ", " << name;
      printUnpackedTypePostfix(portInfo[portIdx].type, os);
      ++portIdx;
    }

    if (portIdx != e) {
      if (portIdx <= lastNonZeroPort)
        os << ',';
    } else if (isZeroWidth)
      os << "\n   );\n";
    else
      os << ");\n";
    os << '\n';
  }

  if (portInfo.empty()) {
    os << ");";
    emitLocationInfoAndNewLine(moduleOpSet);
  }
  reduceIndent();

  // Emit the body of the module.
  RearrangableOStream outputBuffer;
  StmtEmitter(*this, outputBuffer, names)
      .emitStatementBlock(*module.getBodyBlock());
  outputBuffer.print(os);
  os << "endmodule\n\n";

  currentModuleOp = HWModuleOp();
}

//===----------------------------------------------------------------------===//
// Top level "file" emitter logic
//===----------------------------------------------------------------------===//

/// Organize the operations in the root MLIR module into output files to be
/// generated. If `separateModules` is true, a handful of top-level
/// declarations will be split into separate output files even in the absence
/// of an explicit output file attribute.
void SharedEmitterState::gatherFiles(bool separateModules) {

  /// Collect all the inner names from the specified module and add them to the
  /// IRCache.  Declarations (named things) only exist at the top level of the
  /// module.  Also keep track of any modules that contain bind operations.
  /// These are non-hierarchical references which we need to be careful about
  /// during emission.
  auto collectInstanceSymbolsAndBinds = [&](HWModuleOp moduleOp) {
    moduleOp.walk([&](Operation *op) {
      // Populate the symbolCache with all operations that can define a symbol.
      if (auto name = op->getAttrOfType<StringAttr>(
              hw::InnerName::getInnerNameAttrName()))
        symbolCache.addDefinition(moduleOp.getNameAttr(), name.getValue(), op);
      // HACK: This is to make interface-related operations work as they are at
      // the moment, with names being stored in `sym_name` instead of
      // `inner_sym`.
      if (auto instOp = dyn_cast<InterfaceInstanceOp>(op))
        if (auto attr = instOp.sym_nameAttr())
          symbolCache.addDefinition(moduleOp.getNameAttr(), attr.getValue(),
                                    op);
      if (isa<BindOp>(op))
        modulesContainingBinds.insert(moduleOp);
    });
  };
  /// Collect any port marked as being referenced via symbol.
  auto collectPorts = [&](auto moduleOp) {
    auto numArgs = moduleOp.getNumArguments();
    for (size_t p = 0; p != numArgs; ++p)
      for (NamedAttribute argAttr : moduleOp.getArgAttrs(p))
        if (auto sym = argAttr.getValue().dyn_cast<FlatSymbolRefAttr>())
          symbolCache.addDefinition(moduleOp.getNameAttr(), sym.getValue(),
                                    moduleOp, p);
    for (size_t p = 0, e = moduleOp.getNumResults(); p != e; ++p)
      for (NamedAttribute resultAttr : moduleOp.getResultAttrs(p))
        if (auto sym = resultAttr.getValue().dyn_cast<FlatSymbolRefAttr>())
          symbolCache.addDefinition(moduleOp.getNameAttr(), sym.getValue(),
                                    moduleOp, p + numArgs);
  };

  SmallString<32> outputPath;
  for (auto &op : *designOp.getBody()) {
    auto info = OpFileInfo{&op, replicatedOps.size()};

    bool hasFileName = false;
    bool emitReplicatedOps = true;
    bool addToFilelist = true;

    outputPath.clear();

    // Check if the operation has an explicit `output_file` attribute set. If
    // it does, extract the information from the attribute.
    auto attr = op.getAttrOfType<hw::OutputFileAttr>("output_file");
    if (attr) {
      LLVM_DEBUG(llvm::dbgs() << "Found output_file attribute " << attr
                              << " on " << op << "\n";);
      if (!attr.isDirectory())
        hasFileName = true;
      appendPossiblyAbsolutePath(outputPath, attr.getFilename().getValue());
      emitReplicatedOps = attr.getIncludeReplicatedOps().getValue();
      addToFilelist = !attr.getExcludeFromFilelist().getValue();
    }

    // Collect extra file lists to output the file to.
    SmallVector<StringAttr> opFileList;
    if (auto fl = op.getAttrOfType<hw::FileListAttr>("output_filelist"))
      opFileList.push_back(fl.getFilename());
    if (auto fla = op.getAttrOfType<ArrayAttr>("output_filelist"))
      for (auto fl : fla)
        opFileList.push_back(fl.cast<hw::FileListAttr>().getFilename());

    auto separateFile = [&](Operation *op, Twine defaultFileName = "") {
      // If we're emitting to a separate file and the output_file attribute
      // didn't specify a filename, take the default one if present or emit an
      // error if not.
      if (!hasFileName) {
        if (!defaultFileName.isTriviallyEmpty()) {
          llvm::sys::path::append(outputPath, defaultFileName);
        } else {
          op->emitError("file name unspecified");
          encounteredError = true;
          llvm::sys::path::append(outputPath, "error.out");
        }
      }

      auto destFile = StringAttr::get(op->getContext(), outputPath);
      auto &file = files[destFile];
      file.ops.push_back(info);
      file.emitReplicatedOps = emitReplicatedOps;
      file.addToFilelist = addToFilelist;
      for (auto fl : opFileList)
        fileLists[fl.getValue()].push_back(destFile);
    };

    // Separate the operation into dedicated output file, or emit into the
    // root file, or replicate in all output files.
    TypeSwitch<Operation *>(&op)
        .Case<HWModuleOp>([&](auto mod) {
          // Build the IR cache.
          symbolCache.addDefinition(mod.getNameAttr(), mod);
          collectPorts(mod);
          collectInstanceSymbolsAndBinds(mod);

          // Emit into a separate file named after the module.
          if (attr || separateModules)
            separateFile(mod, getVerilogModuleName(mod) + ".sv");
          else
            rootFile.ops.push_back(info);
        })
        .Case<InterfaceOp>([&](InterfaceOp intf) {
          // Build the IR cache.
          symbolCache.addDefinition(intf.getNameAttr(), intf);
          // Populate the symbolCache with all operations that can define a
          // symbol.
          for (auto &op : *intf.getBodyBlock())
            if (auto symOp = dyn_cast<mlir::SymbolOpInterface>(op))
              if (auto name = symOp.getNameAttr())
                symbolCache.addDefinition(name, symOp);

          // Emit into a separate file named after the interface.
          if (attr || separateModules)
            separateFile(intf, intf.sym_name() + ".sv");
          else
            rootFile.ops.push_back(info);
        })
        .Case<HWModuleExternOp>([&](HWModuleExternOp op) {
          // Build the IR cache.
          symbolCache.addDefinition(op.getNameAttr(), op);
          collectPorts(op);
          if (separateModules)
            separateFile(op, "extern_modules.sv");
          else
            rootFile.ops.push_back(info);
        })
        .Case<VerbatimOp, IfDefOp>([&](Operation *op) {
          // Emit into a separate file using the specified file name or
          // replicate the operation in each outputfile.
          if (!attr) {
            replicatedOps.push_back(op);
          } else
            separateFile(op, "");
        })
        .Case<HWGeneratorSchemaOp>([&](HWGeneratorSchemaOp schemaOp) {
          symbolCache.addDefinition(schemaOp.getNameAttr(), schemaOp);
        })
        .Case<TypeScopeOp>([&](TypeScopeOp op) {
          symbolCache.addDefinition(op.getNameAttr(), op);
          // TODO: How do we want to handle typedefs in a split output?
          if (!attr) {
            replicatedOps.push_back(op);
          } else
            separateFile(op, "");
        })
        .Case<BindOp, BindInterfaceOp>([&](auto op) {
          if (!attr) {
            separateFile(op, "bindfile");
          } else {
            separateFile(op);
          }
        })
        .Default([&](auto *) {
          op.emitError("unknown operation");
          encounteredError = true;
        });
  }

  // We've built the whole symbol cache.  Freeze it so things can start
  // querying it (potentially concurrently).
  symbolCache.freeze();
}

/// Given a FileInfo, collect all the replicated and designated operations
/// that go into it and append them to "thingsToEmit".
void SharedEmitterState::collectOpsForFile(const FileInfo &file,
                                           EmissionList &thingsToEmit) {
  // If we're emitting replicated ops, keep track of where we are in the list.
  size_t lastReplicatedOp = 0;
  size_t numReplicatedOps = file.emitReplicatedOps ? replicatedOps.size() : 0;

  thingsToEmit.reserve(thingsToEmit.size() + numReplicatedOps +
                       file.ops.size());

  // Emit each operation in the file preceded by the replicated ops not yet
  // printed.
  for (const auto &opInfo : file.ops) {
    // Emit the replicated per-file operations before the main operation's
    // position (if enabled).
    for (; lastReplicatedOp < std::min(opInfo.position, numReplicatedOps);
         ++lastReplicatedOp)
      thingsToEmit.emplace_back(replicatedOps[lastReplicatedOp]);

    // Emit the operation itself.
    thingsToEmit.emplace_back(opInfo.op);
  }

  // Emit the replicated per-file operations after the last operation (if
  // enabled).
  for (; lastReplicatedOp < numReplicatedOps; lastReplicatedOp++)
    thingsToEmit.emplace_back(replicatedOps[lastReplicatedOp]);
}

static void emitOperation(VerilogEmitterState &state, Operation *op) {
  TypeSwitch<Operation *>(op)
      .Case<HWModuleOp>([&](auto op) { ModuleEmitter(state).emitHWModule(op); })
      .Case<HWModuleExternOp>(
          [&](auto op) { ModuleEmitter(state).emitHWExternModule(op); })
      .Case<HWModuleGeneratedOp>(
          [&](auto op) { ModuleEmitter(state).emitHWGeneratedModule(op); })
      .Case<HWGeneratorSchemaOp>([&](auto op) { /* Empty */ })
      .Case<BindOp>([&](auto op) { ModuleEmitter(state).emitBind(op); })
      .Case<BindInterfaceOp>(
          [&](auto op) { ModuleEmitter(state).emitBindInterface(op); })
      .Case<InterfaceOp, VerbatimOp, IfDefOp>(
          [&](auto op) { ModuleEmitter(state).emitStatement(op); })
      .Case<TypeScopeOp>([&](auto typedecls) {
        ModuleEmitter(state).emitStatement(typedecls);
      })
      .Default([&](auto *op) {
        state.encounteredError = true;
        op->emitError("unknown operation");
      });
}

/// Actually emit the collected list of operations and strings to the
/// specified file.
void SharedEmitterState::emitOps(EmissionList &thingsToEmit, raw_ostream &os,
                                 bool parallelize) {
  MLIRContext *context = designOp->getContext();

  // Disable parallelization overhead if MLIR threading is disabled.
  if (parallelize)
    parallelize &= context->isMultithreadingEnabled();

  // If we aren't parallelizing output, directly output each operation to the
  // specified stream.
  if (!parallelize) {
    VerilogEmitterState state(designOp, *this, options, symbolCache,
                              globalNames, os);
    for (auto &entry : thingsToEmit) {
      if (auto *op = entry.getOperation())
        emitOperation(state, op);
      else
        os << entry.getStringData();
    }

    if (state.encounteredError)
      encounteredError = true;
    return;
  }

  // If we are parallelizing emission, we emit each independent operation to a
  // string buffer in parallel, then concat at the end.
  parallelForEach(context, thingsToEmit, [&](StringOrOpToEmit &stringOrOp) {
    auto *op = stringOrOp.getOperation();
    if (!op)
      return; // Ignore things that are already strings.

    // BindOp emission reaches into the hw.module of the instance, and that
    // body may be being transformed by its own emission.  Defer their
    // emission to the serial phase.  They are speedy to emit anyway.
    if (isa<BindOp>(op) || modulesContainingBinds.count(op))
      return;

    SmallString<256> buffer;
    llvm::raw_svector_ostream tmpStream(buffer);
    VerilogEmitterState state(designOp, *this, options, symbolCache,
                              globalNames, tmpStream);
    emitOperation(state, op);
    stringOrOp.setString(buffer);
  });

  // Finally emit each entry now that we know it is a string.
  for (auto &entry : thingsToEmit) {
    // Almost everything is lowered to a string, just concat the strings onto
    // the output stream.
    auto *op = entry.getOperation();
    if (!op) {
      os << entry.getStringData();
      continue;
    }

    // If this wasn't emitted to a string (e.g. it is a bind) do so now.
    VerilogEmitterState state(designOp, *this, options, symbolCache,
                              globalNames, os);
    emitOperation(state, op);
  }
}

/// Prepare the given MLIR module for emission.
static void prepareForEmission(ModuleOp module,
                               const LoweringOptions &options) {
  SmallVector<HWModuleOp> modulesToPrepare;
  module.walk([&](HWModuleOp op) { modulesToPrepare.push_back(op); });
  parallelForEach(module->getContext(), modulesToPrepare, [&](auto op) {
    prepareHWModule(*op.getBodyBlock(), options);
  });
}

//===----------------------------------------------------------------------===//
// Unified Emitter
//===----------------------------------------------------------------------===//

LogicalResult circt::exportVerilog(ModuleOp module, llvm::raw_ostream &os) {
  // Prepare the ops in the module for emission and legalize the names that will
  // end up in the output.
  LoweringOptions options(module);
  prepareForEmission(module, options);
  GlobalNameTable globalNames = legalizeGlobalNames(module);

  SharedEmitterState emitter(module, options, std::move(globalNames));
  emitter.gatherFiles(false);

  SharedEmitterState::EmissionList list;

  // Collect the contents of the main file. This is a container for anything
  // not explicitly split out into a separate file.
  emitter.collectOpsForFile(emitter.rootFile, list);

  // Emit the separate files.
  for (const auto &it : emitter.files) {
    list.emplace_back("\n// ----- 8< ----- FILE \"" + it.first.str() +
                      "\" ----- 8< -----\n\n");
    emitter.collectOpsForFile(it.second, list);
  }

  // Emit the filelists.
  for (auto &it : emitter.fileLists) {
    std::string contents("\n// ----- 8< ----- FILE \"" + it.first().str() +
                         "\" ----- 8< -----\n\n");
    for (auto &name : it.second)
      contents += name.str() + "\n";
    list.emplace_back(contents);
  }

  // Finally, emit all the ops we collected.
  emitter.emitOps(list, os, /*parallelize=*/true);
  return failure(emitter.encounteredError);
}

namespace {

struct ExportVerilogPass : public ExportVerilogBase<ExportVerilogPass> {
  ExportVerilogPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override {
    // Make sure LoweringOptions are applied to the module if it was overridden
    // on the command line.
    // TODO: This should be moved up to circt-opt and circt-translate.
    applyLoweringCLOptions(getOperation());

    if (failed(exportVerilog(getOperation(), os)))
      signalPassFailure();
  }

private:
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::createExportVerilogPass(llvm::raw_ostream &os) {
  return std::make_unique<ExportVerilogPass>(os);
}

std::unique_ptr<mlir::Pass> circt::createExportVerilogPass() {
  return createExportVerilogPass(llvm::outs());
}

//===----------------------------------------------------------------------===//
// Split Emitter
//===----------------------------------------------------------------------===//

static std::unique_ptr<llvm::ToolOutputFile>
createOutputFile(StringRef fileName, StringRef dirname,
                 SharedEmitterState &emitter) {
  // Determine the output path from the output directory and filename.
  SmallString<128> outputFilename(dirname);
  appendPossiblyAbsolutePath(outputFilename, fileName);
  auto outputDir = llvm::sys::path::parent_path(outputFilename);

  // Create the output directory if needed.
  std::error_code error = llvm::sys::fs::create_directories(outputDir);
  if (error) {
    emitter.designOp.emitError("cannot create output directory \"")
        << outputDir << "\": " << error.message();
    emitter.encounteredError = true;
    return {};
  }

  // Open the output file.
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    emitter.designOp.emitError(errorMessage);
    emitter.encounteredError = true;
  }
  return output;
}

static void createSplitOutputFile(StringAttr fileName, FileInfo &file,
                                  StringRef dirname,
                                  SharedEmitterState &emitter) {
  auto output = createOutputFile(fileName, dirname, emitter);
  if (!output)
    return;

  SharedEmitterState::EmissionList list;
  emitter.collectOpsForFile(file, list);

  // Emit the file, copying the global options into the individual module
  // state.  Don't parallelize emission of the ops within this file - we
  // already parallelize per-file emission and we pay a string copy overhead
  // for parallelization.
  emitter.emitOps(list, output->os(), /*parallelize=*/false);
  output->keep();
}

LogicalResult circt::exportSplitVerilog(ModuleOp module, StringRef dirname) {
  // Prepare the ops in the module for emission and legalize the names that will
  // end up in the output.
  LoweringOptions options(module);
  prepareForEmission(module, options);
  GlobalNameTable globalNames = legalizeGlobalNames(module);

  SharedEmitterState emitter(module, options, std::move(globalNames));
  emitter.gatherFiles(true);

  // Emit each file in parallel if context enables it.
  parallelForEach(module->getContext(), emitter.files.begin(),
                  emitter.files.end(), [&](auto &it) {
                    createSplitOutputFile(it.first, it.second, dirname,
                                          emitter);
                  });

  // Write the file list.
  SmallString<128> filelistPath(dirname);
  llvm::sys::path::append(filelistPath, "filelist.f");

  std::string errorMessage;
  auto output = mlir::openOutputFile(filelistPath, &errorMessage);
  if (!output) {
    module->emitError(errorMessage);
    return failure();
  }

  for (const auto &it : emitter.files) {
    if (it.second.addToFilelist)
      output->os() << it.first << "\n";
  }
  output->keep();

  // Emit the filelists.
  for (auto &it : emitter.fileLists) {
    auto output = createOutputFile(it.first(), dirname, emitter);
    if (!output)
      continue;
    for (auto &name : it.second)
      output->os() << name.str() << "\n";
    output->keep();
  }

  return failure(emitter.encounteredError);
}

namespace {

struct ExportSplitVerilogPass
    : public ExportSplitVerilogBase<ExportSplitVerilogPass> {
  ExportSplitVerilogPass(StringRef directory) {
    directoryName = directory.str();
  }
  void runOnOperation() override {
    // Make sure LoweringOptions are applied to the module if it was overridden
    // on the command line.
    // TODO: This should be moved up to circt-opt and circt-translate.
    applyLoweringCLOptions(getOperation());
    if (failed(exportSplitVerilog(getOperation(), directoryName)))
      signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::createExportSplitVerilogPass(StringRef directory) {
  return std::make_unique<ExportSplitVerilogPass>(directory);
}
