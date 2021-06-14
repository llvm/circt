//===- LowerBundleVectorTypes.cpp - Expand WhenOps into muxed operations ---*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerBundleVectorTypes pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Parallel.h"

using namespace circt;
using namespace firrtl;

// Helper to peel off the outer most flip type from an aggregate type that has
// all flips canonicalized to the outer level, or just return the bundle
// directly. For any ground type, returns null.
static FIRRTLType getCanonicalAggregateType(Type originalType) {
  FIRRTLType unflipped = originalType.dyn_cast<FIRRTLType>();
  if (auto flipType = originalType.dyn_cast<FlipType>())
    unflipped = flipType.getElementType();

  return TypeSwitch<FIRRTLType, FIRRTLType>(unflipped)
      .Case<BundleType, FVectorType>([](auto a) { return a; })
      .Default([](auto) { return nullptr; });
}
// TODO: check all argument types
namespace {
// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  // This is the underlying ground type of the field.
  FIRRTLType type;
  // This is a suffix to add to the field name to make it unique.
  SmallString<16> suffix;
  // This indicates whether the field was flipped to be an output.
  bool isOutput;

  // Helper to determine if a fully flattened type needs to be flipped.
  FIRRTLType getPortType() { return isOutput ? FlipType::get(type) : type; }
  FlatBundleFieldEntry(const FIRRTLType& type, StringRef suffix, bool isOutput)
  :type(type), suffix(suffix), isOutput(isOutput) {}
};
} // end anonymous namespace

static SmallVector<FlatBundleFieldEntry> peelType(FIRRTLType type) {
  // Field ID is the index of each field when walking aggregate types in a
  // depth-first order. See FieldRef for more information about how this
  // works.
  bool isFlipped = false;
  SmallVector<FlatBundleFieldEntry> retval;
        // Strip off an outer flip type. Flip types don't don't have a field ID,
        // and so there is no need to increment the field ID.
        if (auto flip = type.dyn_cast<FlipType>()) {
          type = flip.getElementType();
          isFlipped = !isFlipped;
        }
        TypeSwitch<FIRRTLType>(type)
            .Case<BundleType>([&](auto bundle) {
              SmallString<16> tmpSuffix;
              // Otherwise, we have a bundle type.  Break it down.
              for (auto &elt : bundle.getElements()) {
                // Construct the suffix to pass down.
                tmpSuffix.resize(0);
                tmpSuffix.push_back('_');
                tmpSuffix.append(elt.name.getValue());
                if (auto flip = elt.type.template dyn_cast<FlipType>()) {
                  retval.emplace_back(flip.getElementType(), tmpSuffix, !isFlipped);
                } else {
                  retval.emplace_back(elt.type, tmpSuffix, isFlipped);
                }
                                          }
            })
            .Case<FVectorType>([&](auto vector) {
              // Increment the field ID to point to the first element.
              for (size_t i = 0, e = vector.getNumElements(); i != e; ++i) {
                retval.emplace_back(vector.getElementType(),
                        "_" + std::to_string(i),
                        isFlipped);
              }
            })
            .Default([&](auto) {
              retval.emplace_back(type, "", isFlipped);
            });
return retval;
}


/// Copy annotations from \p annotations to \p loweredAttrs, except annotations
/// with "target" key, that do not match the field suffix.
static void filterAnnotations(ArrayAttr annotations,
                              SmallVector<Attribute> &loweredAttrs,
                              StringRef suffix) {
  if (!annotations || annotations.empty())
    return;

  for (auto opAttr : annotations) {
    auto di = opAttr.dyn_cast<DictionaryAttr>();
    if (!di) {
      loweredAttrs.push_back(opAttr);
      continue;
    }
    auto targetAttr = di.get("target");
    if (!targetAttr) {
      loweredAttrs.push_back(opAttr);
      continue;
    }

    ArrayAttr subFieldTarget = targetAttr.cast<ArrayAttr>();
    SmallString<16> targetStr;
    for (auto fName : subFieldTarget) {
      std::string fNameStr = fName.cast<StringAttr>().getValue().str();
      // The fNameStr will begin with either '[' or '.', replace it with an
      // '_' to construct the suffix.
      fNameStr[0] = '_';
      // If it ends with ']', then just remove it.
      if (fNameStr.back() == ']')
        fNameStr.erase(fNameStr.size() - 1);

      targetStr += fNameStr;
    }
    // If no subfield attribute, then copy the annotation.
    if (targetStr.empty()) {
      loweredAttrs.push_back(opAttr);
      continue;
    }
    // If the subfield suffix doesn't match, then ignore the annotation.
    if (suffix.find(targetStr.str().str()) != 0)
      continue;

    NamedAttrList modAttr;
    for (auto attr : di.getValue()) {
      // Ignore the actual target annotation, but copy the rest of annotations.
      if (attr.first.str() == "target")
        continue;
      modAttr.push_back(attr);
    }
    loweredAttrs.push_back(
        DictionaryAttr::get(annotations.getContext(), modAttr));
  }
}

/// Copy annotations from \p annotations into a new AnnotationSet and return it.
/// This removes annotations with "target" key that does not match the field
/// suffix.
static AnnotationSet filterAnnotations(AnnotationSet annotations,
                                       StringRef suffix) {
  if (annotations.empty())
    return annotations;
  SmallVector<Attribute> loweredAttrs;
  filterAnnotations(annotations.getArrayAttr(), loweredAttrs, suffix);
  return AnnotationSet(ArrayAttr::get(annotations.getContext(), loweredAttrs));
}


namespace {
class AggregateUserVisitor
    : public FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>> {
public:
  AggregateUserVisitor(MLIRContext *context, ImplicitLocOpBuilder *builder)
      : context(context), builder(builder) {}
  using FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>>::visitDecl;
  using FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>>::visitExpr;
  using FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>>::visitStmt;

  void visitExpr(SubfieldOp op, ArrayRef<Value> mapping);
  void visitExpr(SubindexOp op, ArrayRef<Value> mapping);
  void visitExpr(SubaccessOp op, ArrayRef<Value> mapping);

private:
  MLIRContext *context;

  // The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;
};
}

void AggregateUserVisitor::visitExpr(SubindexOp op, ArrayRef<Value> mapping) {
  // Get the input bundle type.
  Value input = op.input();
  auto inputType = input.getType();
  if (auto flipType = inputType.dyn_cast<FlipType>())
    inputType = flipType.getElementType();

  op.replaceAllUsesWith(mapping[op.index()]);

  // Remember to remove the original op.
  op.erase();
}

void AggregateUserVisitor::visitExpr(SubaccessOp op, ArrayRef<Value> mapping) {
  // Get the input bundle type.
  Value input = op.input();
  auto inputType = input.getType();
  if (auto flipType = inputType.dyn_cast<FlipType>())
    inputType = flipType.getElementType();

  auto selectWidth =
      op.index().getType().cast<FIRRTLType>().getBitWidthOrSentinel();

  // if we were given a mapping, this is a write and the mapping is the write
  // Value
  for (size_t index = 0, e = inputType.cast<FVectorType>().getNumElements();
       index < e; ++index) {
    auto cond = builder->create<EQPrimOp>(
        op.index(), builder->createOrFold<ConstantOp>(
                        UIntType::get(op.getContext(), selectWidth),
                        APInt(selectWidth, index)));
    auto whenCond = builder->create<WhenOp>(cond, false, [&]() {
      builder->create<ConnectOp>(builder->create<SubindexOp>(input, index),
                                 mapping[0]);
    });
  }
}

void AggregateUserVisitor::visitExpr(SubfieldOp op, ArrayRef<Value> mapping) {
  // Get the input bundle type.
  Value input = op.input();
  auto inputType = input.getType();
  if (auto flipType = inputType.dyn_cast<FlipType>())
    inputType = flipType.getElementType();
  auto bundleType = inputType.cast<BundleType>();

  op.replaceAllUsesWith(mapping[*bundleType.getElementIndex(op.fieldname())]);

  // Remember to remove the original op.
  op.erase();
}

  //===----------------------------------------------------------------------===//
  // Module Type Lowering
  //===----------------------------------------------------------------------===//
  namespace {
  class TypeLoweringVisitor : public FIRRTLVisitor<TypeLoweringVisitor> {
  public:
    TypeLoweringVisitor(MLIRContext *context) : context(context) {}
    using FIRRTLVisitor<TypeLoweringVisitor>::visitDecl;
    using FIRRTLVisitor<TypeLoweringVisitor>::visitExpr;
    using FIRRTLVisitor<TypeLoweringVisitor>::visitStmt;

    // If the referenced operation is a FModuleOp or an FExtModuleOp, perform
    // type lowering on all operations.
    void lowerModule(Operation *op);

    void visitDecl(FExtModuleOp op);
    void visitDecl(FModuleOp op);
//    void visitDecl(InstanceOp op, ArrayRef<Value> mapping);
//    void visitDecl(MemOp op, ArrayRef<Value> mapping);
    void visitDecl(NodeOp op);
    void visitDecl(RegOp op);
    void visitDecl(WireOp op);
    void visitDecl(RegResetOp op);
//    void visitExpr(InvalidValueOp op, ArrayRef<Value> mapping);
//    void visitExpr(SubfieldOp op);
//    void visitExpr(SubindexOp op);
    void visitExpr(SubaccessOp op);
    void visitExpr(MuxPrimOp op);
    void visitStmt(ConnectOp op);
//    void visitStmt(WhenOp op, ArrayRef<Value> mapping);
//    void visitStmt(PartialConnectOp op, ArrayRef<Value> mapping);

  private:

  void processUsers(Operation* op, ArrayRef<Value> mapping);

  MLIRContext* context;

  // The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;

  // State to keep track of arguments and operations to clean up at the end.
  SmallVector<Operation *, 16> opsToRemove;
};
}

void TypeLoweringVisitor::processUsers(Operation *op, ArrayRef<Value> mapping) {
  AggregateUserVisitor aggV(context, builder);
  for (auto user : llvm::make_early_inc_range(op->getUsers())) {
    aggV.dispatchVisitor(user, mapping);
  }
}

void TypeLoweringVisitor::lowerModule(Operation *op) {
  if (auto module = dyn_cast<FModuleOp>(op))
    return visitDecl(module);
  if (auto extModule = dyn_cast<FExtModuleOp>(op))
    return visitDecl(extModule);
}

// Expand connects of aggregates
void TypeLoweringVisitor::visitStmt(ConnectOp op) {
  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(op.src().getType());

  // Ground Type
  if (!resultType) {
    if (SubaccessOp sao = dyn_cast_or_null<SubaccessOp>(op.dest().getDefiningOp())) {
      AggregateUserVisitor(context, builder)
          .visitExpr(sao, ArrayRef<Value>(op.src()));
      opsToRemove.push_back(op);
    }
    return;
  }

  SmallVector<FlatBundleFieldEntry> fields = peelType(resultType);

  // Loop over the leaf aggregates.
  for (auto field : llvm::enumerate(fields)) {
    Value src, dest;
    if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
    src = builder->create<SubfieldOp>(op.src(), bundle.getElement(field.index()).name);
    dest = builder->create<SubfieldOp>(op.dest(),
                                       bundle.getElement(field.index()).name);
    } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
      src = builder->create<SubindexOp>(op.src(), field.index());
      dest = builder->create<SubindexOp>(op.dest(), field.index());
    } else {
      op->emitError("Unknown aggregate type");
    }
    if (field.value().isOutput)
    std::swap(src,dest);
    builder->create<ConnectOp>(dest, src);
  }
opsToRemove.push_back(op);
}

// Expand muxes of aggregates
void TypeLoweringVisitor::visitExpr(MuxPrimOp op) {
  Value result = op.result();

  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(result.getType());

  // If the wire is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
  SmallVector<Value> lowered;

  // Loop over the leaf aggregates.
  for (auto field : llvm::enumerate(fieldTypes)) {
    Value high, low;
    if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
      high = builder->create<SubfieldOp>(op.high(),
                                        bundle.getElement(field.index()).name);
      low = builder->create<SubfieldOp>(op.low(),
                                         bundle.getElement(field.index()).name);
    } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
      high = builder->create<SubindexOp>(op.high(), field.index());
      low = builder->create<SubindexOp>(op.low(), field.index());
    } else {
      op->emitError("Unknown aggregate type");
    }

    auto mux = builder->create<MuxPrimOp>(op.sel(), high, low);
    lowered.push_back(mux.getResult());
  }
  processUsers(op, lowered);
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitDecl(FExtModuleOp module) {
}

  void TypeLoweringVisitor::visitDecl(FModuleOp module) {
    auto *body = module.getBodyBlock();

    ImplicitLocOpBuilder theBuilder(module.getLoc(), context);
    builder = &theBuilder;

    // Lower the operations.
    for (auto iop = body->rbegin(), iep = body->rend(); iop != iep; ++iop) {
      // We erase old ops eagerly so we don't have dangling uses we've already lowered.
      for (auto *op : opsToRemove)
        op->erase();
      opsToRemove.clear();

      builder->setInsertionPoint(&*iop);
      builder->setLoc(iop->getLoc());
      dispatchVisitor(&*iop);
    }

    for (auto *op : opsToRemove)
      op->erase();
      opsToRemove.clear();

      // Lower the module block arguments.
      // SmallVector<BlockArgument, 8> args(body->args_begin(),
      // body->args_end()); originalNumModuleArgs = args.size(); for (auto arg :
      // args)
      //   if (auto type = arg.getType().dyn_cast<FIRRTLType>())
      //     lowerArg(module, arg, type);
  }

  /// Lower a wire op with a bundle to multiple non-bundled wires.
  void TypeLoweringVisitor::visitDecl(WireOp op) {
    Value result = op.result();

    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());

    // If the wire is not a bundle, there is nothing to do.
    if (!resultType)
      return;

    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
    SmallVector<Value> lowered;

    // Loop over the leaf aggregates.
    auto name = op.name().str();
    for (auto field : fieldTypes) {
      SmallString<16> loweredName;
      if (!name.empty())
        loweredName = (name + field.suffix).str();
      SmallVector<Attribute> loweredAttrs;
      // For all annotations on the parent op, filter them based on the target
      // attribute.
      filterAnnotations(op.annotations(), loweredAttrs, field.suffix);
      auto wire =
          builder->create<WireOp>(field.type, loweredName, loweredAttrs);
      lowered.push_back(wire.getResult());
    }
    processUsers(op, lowered);
    opsToRemove.push_back(op);
  }

  /// Lower a reg op with a bundle to multiple non-bundled regs.
  void TypeLoweringVisitor::visitDecl(RegOp op) {
    Value result = op.result();

    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());

    // If the wire is not a bundle, there is nothing to do.
    if (!resultType)
      return;

    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
    SmallVector<Value> lowered;

    // Loop over the leaf aggregates.
    auto name = op.name().str();
    for (auto field : fieldTypes) {
      SmallString<16> loweredName;
      if (!name.empty())
        loweredName = (name + field.suffix).str();
      SmallVector<Attribute> loweredAttrs;
      // For all annotations on the parent op, filter them based on the target
      // attribute.
      filterAnnotations(op.annotations(), loweredAttrs, field.suffix);
      auto reg =
          builder->create<RegOp>(field.getPortType(), op.clockVal(), loweredName, loweredAttrs);
      lowered.push_back(reg.getResult());
    }
    processUsers(op, lowered);
    opsToRemove.push_back(op);
  }

  /// Lower a reg op with a bundle to multiple non-bundled regs.
  void TypeLoweringVisitor::visitDecl(RegResetOp op) {
    Value result = op.result();

    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());

    // If the wire is not a bundle, there is nothing to do.
    if (!resultType)
      return;

    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
    SmallVector<Value> lowered;

    // Loop over the leaf aggregates.
    auto name = op.name().str();
    for (auto field : llvm::enumerate(fieldTypes)) {
      SmallString<16> loweredName;
      if (!name.empty())
        loweredName = (name + field.value().suffix).str();
      SmallVector<Attribute> loweredAttrs;
      // For all annotations on the parent op, filter them based on the target
      // attribute.
      filterAnnotations(op.annotations(), loweredAttrs, field.value().suffix);
      Value resetVal;
      if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
        resetVal = builder->create<SubfieldOp>(
            op.resetValue(), bundle.getElement(field.index()).name);
      } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
        resetVal = builder->create<SubindexOp>(op.resetValue(), field.index());
      } else {
        op->emitError("Unknown aggregate type");
      }

      auto reg = builder->create<RegResetOp>(field.value().getPortType(), op.clockVal(), op.resetSignal(),
                                        resetVal, loweredName, loweredAttrs);
      lowered.push_back(reg.getResult());
    }
    processUsers(op, lowered);
    opsToRemove.push_back(op);
  }

  /// Lower a wire op with a bundle to multiple non-bundled wires.
  void TypeLoweringVisitor::visitDecl(NodeOp op) {
    Value result = op.result();

    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());

    // If the wire is not a bundle, there is nothing to do.
    if (!resultType)
      return;

    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
    SmallVector<Value> lowered;

    // Loop over the leaf aggregates.
    auto name = op.name().str();
    for (auto field : llvm::enumerate(fieldTypes)) {
      SmallString<16> loweredName;
      if (!name.empty())
        loweredName = (name + field.value().suffix).str();
      SmallVector<Attribute> loweredAttrs;
      // For all annotations on the parent op, filter them based on the target
      // attribute.
      filterAnnotations(op.annotations(), loweredAttrs, field.value().suffix);
      Value input;
      if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
        input = builder->create<SubfieldOp>(
            op.input(), bundle.getElement(field.index()).name);
      } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
        input = builder->create<SubindexOp>(op.input(), field.index());
      } else {
        op->emitError("Unknown aggregate type");
      }

      auto node =
          builder->create<NodeOp>(field.value().type, input, loweredName, loweredAttrs);
      lowered.push_back(node.getResult());
    }
    processUsers(op, lowered);
    opsToRemove.push_back(op);
  }


  void TypeLoweringVisitor::visitExpr(SubaccessOp op) {
    // Get the input bundle type.
    Value input = op.input();
    auto inputType = input.getType();
    if (auto flipType = inputType.dyn_cast<FlipType>())
      inputType = flipType.getElementType();

    auto selectWidth =
        op.index().getType().cast<FIRRTLType>().getBitWidthOrSentinel();

      //Reads.  All writes have been eliminated before now
      
      auto vType = inputType.cast<FVectorType>();
       Value mux =
          builder->create<InvalidValueOp>(vType.getElementType());
       
          for (size_t index = vType.getNumElements(); index > 0; --index) {
            auto cond = builder->create<EQPrimOp>(
            op.index(), builder->createOrFold<ConstantOp>(
                            UIntType::get(op.getContext(), selectWidth),
                            APInt(selectWidth, index-1)));
                            auto access = builder->create<SubindexOp>(input, index-1);
            mux = builder->create<MuxPrimOp>(cond, access, mux);
      }
      op.replaceAllUsesWith(mux);
      opsToRemove.push_back(op);


        return;
  }

  //
  //
  //===----------------------------------------------------------------------===//
  // Pass Infrastructure
  //===----------------------------------------------------------------------===//

namespace {
  struct LowerBundleVectorPass
      : public LowerBundleVectorBase<LowerBundleVectorPass> {
    void runOnOperation() override;

  private:
    void runAsync();
    void runSync();
  };
  } // end anonymous namespace

  void LowerBundleVectorPass::runSync() {
    auto circuit = getOperation();
    for (auto &op : circuit.getBody()->getOperations()) {
      TypeLoweringVisitor(&getContext()).lowerModule(&op);
    }
  }

  // This is the main entrypoint for the lowering pass.
  void LowerBundleVectorPass::runOnOperation() {
      runSync();
  }

  /// This is the pass constructor.
  std::unique_ptr<mlir::Pass>
  circt::firrtl::createLowerBundleVectorTypesPass() {
    return std::make_unique<LowerBundleVectorPass>();
  }
