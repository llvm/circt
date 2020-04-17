//===- Dialect.cpp - Implement the FIRRTL dialect -------------------------===//
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"

using namespace spt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// If the specified attribute set contains the firrtl.name attribute, return it.
StringAttr firrtl::getFIRRTLNameAttr(ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs) {
    // FIXME: We currently use firrtl.name instead of name because this makes
    // the FunctionLike handling in MLIR core happier.  It otherwise doesn't
    // allow attributes on module parameters.
    if (argAttr.first != "firrtl.name")
      continue;

    return argAttr.second.dyn_cast<StringAttr>();
  }

  return StringAttr();
}

namespace {

// We implement the OpAsmDialectInterface so that FIRRTL dialect operations
// automatically interpret the name attribute on function arguments and
// on operations as their SSA name.
struct FIRRTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {
    // Many firrtl dialect operations have an optional 'name' attribute.  If
    // present, use it.
    if (op->getNumResults() > 0)
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        setNameFn(op->getResult(0), nameAttr.getValue());

    // For constants in particular, propagate the value into the result name to
    // make it easier to read the IR.
    if (auto constant = dyn_cast<ConstantOp>(op)) {
      auto intTy = constant.getType().dyn_cast<IntType>();

      // Otherwise, build a complex name with the value and type.
      SmallString<32> specialNameBuffer;
      llvm::raw_svector_ostream specialName(specialNameBuffer);
      specialName << 'c';
      if (intTy) {
        if (!intTy.isSigned() || !constant.value().isNegative())
          constant.value().print(specialName, /*isSigned:*/ false);
        else {
          specialName << 'm';
          (-constant.value()).print(specialName, /*isSigned:*/ false);
        }

        specialName << (intTy.isSigned() ? "_si" : "_ui");
        auto width = intTy.getWidthOrSentinel();
        if (width != -1)
          specialName << width;
      } else {
        constant.value().print(specialName, /*isSigned:*/ false);
      }
      setNameFn(constant.getResult(), specialName.str());
    }
  }

  /// Get a special name to use when printing the entry block arguments of the
  /// region contained by an operation in this dialect.
  void getAsmBlockArgumentNames(Block *block,
                                OpAsmSetValueNameFn setNameFn) const override {
    // Check to see if the operation containing the arguments has 'firrtl.name'
    // attributes for them.  If so, use that as the name.
    auto *parentOp = block->getParentOp();

    for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
      // Scan for a 'firrtl.name' attribute.
      if (auto str = getFIRRTLNameAttr(impl::getArgAttrs(parentOp, i)))
        setNameFn(block->getArgument(i), str.getValue());
    }
  }
};
} // end anonymous namespace

FIRRTLDialect::FIRRTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {

  // Register types.
  addTypes<SIntType, UIntType, ClockType, ResetType, AsyncResetType, AnalogType,
           // Derived Types
           FlipType, BundleType, FVectorType>();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<FIRRTLOpAsmDialectInterface>();
}

FIRRTLDialect::~FIRRTLDialect() {}

void FIRRTLDialect::printType(Type type, DialectAsmPrinter &os) const {
  type.cast<FIRRTLType>().print(os.getStream());
}

// Provide implementations for the enums we use.
#include "spt/Dialect/FIRRTL/IR/FIRRTLEnums.cpp.inc"
