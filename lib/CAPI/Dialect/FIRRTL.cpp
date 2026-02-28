//===- FIRRTL.cpp - C interface for the FIRRTL dialect --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the C-API for FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/FIRRTL.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Import/FIRAnnotations.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace firrtl;

namespace json = llvm::json;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl,
                                      circt::firrtl::FIRRTLDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

bool firrtlTypeIsConst(MlirType type) { return isConst(unwrap(type)); }

MlirType firrtlTypeGetConstType(MlirType type, bool isConst) {
  return wrap(cast<FIRRTLBaseType>(unwrap(type)).getConstType(isConst));
}

int64_t firrtlTypeGetBitWidth(MlirType type, bool ignoreFlip) {
  return getBitWidth(cast<FIRRTLBaseType>(unwrap(type)), ignoreFlip)
      .value_or(-1);
}

bool firrtlTypeIsAUInt(MlirType type) { return isa<UIntType>(unwrap(type)); }

MlirType firrtlTypeGetUInt(MlirContext ctx, int32_t width) {
  return wrap(UIntType::get(unwrap(ctx), width));
}

bool firrtlTypeIsASInt(MlirType type) { return isa<SIntType>(unwrap(type)); }

MlirType firrtlTypeGetSInt(MlirContext ctx, int32_t width) {
  return wrap(SIntType::get(unwrap(ctx), width));
}

bool firrtlTypeIsAClock(MlirType type) { return isa<ClockType>(unwrap(type)); }

MlirType firrtlTypeGetClock(MlirContext ctx) {
  return wrap(ClockType::get(unwrap(ctx)));
}

bool firrtlTypeIsAReset(MlirType type) { return isa<ResetType>(unwrap(type)); }

MlirType firrtlTypeGetReset(MlirContext ctx) {
  return wrap(ResetType::get(unwrap(ctx)));
}

bool firrtlTypeIsAAsyncReset(MlirType type) {
  return isa<AsyncResetType>(unwrap(type));
}

MlirType firrtlTypeGetAsyncReset(MlirContext ctx) {
  return wrap(AsyncResetType::get(unwrap(ctx)));
}

bool firrtlTypeIsAAnalog(MlirType type) {
  return isa<AnalogType>(unwrap(type));
}

MlirType firrtlTypeGetAnalog(MlirContext ctx, int32_t width) {
  return wrap(AnalogType::get(unwrap(ctx), width));
}

bool firrtlTypeIsAVector(MlirType type) {
  return isa<FVectorType>(unwrap(type));
}

MlirType firrtlTypeGetVector(MlirContext ctx, MlirType element, size_t count) {
  auto baseType = cast<FIRRTLBaseType>(unwrap(element));
  assert(baseType && "element must be base type");

  return wrap(FVectorType::get(baseType, count));
}

MlirType firrtlTypeGetVectorElement(MlirType vec) {
  return wrap(cast<FVectorType>(unwrap(vec)).getElementType());
}

size_t firrtlTypeGetVectorNumElements(MlirType vec) {
  return cast<FVectorType>(unwrap(vec)).getNumElements();
}

bool firrtlTypeIsABundle(MlirType type) {
  return isa<BundleType>(unwrap(type));
}

bool firrtlTypeIsAOpenBundle(MlirType type) {
  return isa<OpenBundleType>(unwrap(type));
}

MlirType firrtlTypeGetBundle(MlirContext ctx, size_t count,
                             const FIRRTLBundleField *fields) {
  bool bundleCompatible = true;
  SmallVector<OpenBundleType::BundleElement, 4> bundleFields;

  bundleFields.reserve(count);

  for (size_t i = 0; i < count; i++) {
    auto field = fields[i];
    auto type = cast<FIRRTLType>(unwrap(field.type));
    bundleFields.emplace_back(unwrap(field.name), field.isFlip, type);
    bundleCompatible &= isa<BundleType::ElementType>(type);
  }

  // Try to emit base-only bundle.
  if (bundleCompatible) {
    auto bundleFieldsMapped = llvm::map_range(bundleFields, [](auto field) {
      return BundleType::BundleElement{
          field.name, field.isFlip, cast<BundleType::ElementType>(field.type)};
    });
    return wrap(
        BundleType::get(unwrap(ctx), llvm::to_vector(bundleFieldsMapped)));
  }
  return wrap(OpenBundleType::get(unwrap(ctx), bundleFields));
}

size_t firrtlTypeGetBundleNumFields(MlirType bundle) {
  if (auto bundleType = dyn_cast<BundleType>(unwrap(bundle))) {
    return bundleType.getNumElements();
  } else if (auto bundleType = dyn_cast<OpenBundleType>(unwrap(bundle))) {
    return bundleType.getNumElements();
  } else {
    llvm_unreachable("must be a bundle type");
  }
}

bool firrtlTypeGetBundleFieldByIndex(MlirType type, size_t index,
                                     FIRRTLBundleField *field) {
  auto unwrapped = unwrap(type);

  auto cvt = [field](auto element) {
    field->name = wrap(element.name);
    field->isFlip = element.isFlip;
    field->type = wrap(element.type);
  };
  if (auto bundleType = dyn_cast<BundleType>(unwrapped)) {
    cvt(bundleType.getElement(index));
    return true;
  } else if (auto bundleType = dyn_cast<OpenBundleType>(unwrapped)) {
    cvt(bundleType.getElement(index));
    return true;
  }
  return false;
}

unsigned firrtlTypeGetBundleFieldIndex(MlirType type, MlirStringRef fieldName) {
  std::optional<unsigned> fieldIndex;
  if (auto bundleType = dyn_cast<BundleType>(unwrap(type))) {
    fieldIndex = bundleType.getElementIndex(unwrap(fieldName));
  } else if (auto bundleType = dyn_cast<OpenBundleType>(unwrap(type))) {
    fieldIndex = bundleType.getElementIndex(unwrap(fieldName));
  } else {
    llvm_unreachable("must be a bundle type");
  }
  assert(fieldIndex.has_value() && "unknown field");
  return fieldIndex.value();
}

bool firrtlTypeIsARef(MlirType type) { return isa<RefType>(unwrap(type)); }

MlirType firrtlTypeGetRef(MlirType target, bool forceable) {
  auto baseType = dyn_cast<FIRRTLBaseType>(unwrap(target));
  assert(baseType && "target must be base type");

  return wrap(RefType::get(baseType, forceable));
}

MlirType firrtlTypeGetColoredRef(MlirType target, bool forceable,
                                 MlirAttribute layer) {
  auto baseType = dyn_cast<FIRRTLBaseType>(unwrap(target));
  assert(baseType && "target must be base type");
  auto layerAttr = dyn_cast<SymbolRefAttr>(unwrap(layer));
  assert(layerAttr && "layer must be symbol ref attribute");

  return wrap(RefType::get(baseType, forceable, layerAttr));
}

bool firrtlTypeIsAAnyRef(MlirType type) {
  return isa<AnyRefType>(unwrap(type));
}

MlirType firrtlTypeGetAnyRef(MlirContext ctx) {
  return wrap(AnyRefType::get(unwrap(ctx)));
}

bool firrtlTypeIsAInteger(MlirType type) {
  return isa<FIntegerType>(unwrap(type));
}

MlirType firrtlTypeGetInteger(MlirContext ctx) {
  return wrap(FIntegerType::get(unwrap(ctx)));
}

bool firrtlTypeIsADouble(MlirType type) {
  return isa<DoubleType>(unwrap(type));
}

MlirType firrtlTypeGetDouble(MlirContext ctx) {
  return wrap(DoubleType::get(unwrap(ctx)));
}

bool firrtlTypeIsAString(MlirType type) {
  return isa<StringType>(unwrap(type));
}

MlirType firrtlTypeGetString(MlirContext ctx) {
  return wrap(StringType::get(unwrap(ctx)));
}

bool firrtlTypeIsABoolean(MlirType type) { return isa<BoolType>(unwrap(type)); }

MlirType firrtlTypeGetBoolean(MlirContext ctx) {
  return wrap(BoolType::get(unwrap(ctx)));
}

bool firrtlTypeIsAPath(MlirType type) { return isa<PathType>(unwrap(type)); }

MlirType firrtlTypeGetPath(MlirContext ctx) {
  return wrap(PathType::get(unwrap(ctx)));
}

bool firrtlTypeIsAList(MlirType type) { return isa<ListType>(unwrap(type)); }

MlirType firrtlTypeGetList(MlirContext ctx, MlirType elementType) {
  auto type = dyn_cast<PropertyType>(unwrap(elementType));
  assert(type && "element must be property type");

  return wrap(ListType::get(unwrap(ctx), type));
}

bool firrtlTypeIsAClass(MlirType type) { return isa<ClassType>(unwrap(type)); }

MlirType firrtlTypeGetClass(MlirContext ctx, MlirAttribute name,
                            size_t numberOfElements,
                            const FIRRTLClassElement *elements) {
  auto nameSymbol = dyn_cast<FlatSymbolRefAttr>(unwrap(name));
  assert(nameSymbol && "name must be FlatSymbolRefAttr");

  SmallVector<ClassElement, 4> classElements;
  classElements.reserve(numberOfElements);

  for (size_t i = 0; i < numberOfElements; i++) {
    auto element = elements[i];
    auto dir = element.direction == FIRRTL_DIRECTION_IN ? Direction::In
                                                        : Direction::Out;
    classElements.emplace_back(unwrap(element.name), unwrap(element.type), dir);
  }
  return wrap(ClassType::get(unwrap(ctx), nameSymbol, classElements));
}

MlirType firrtlTypeGetMaskType(MlirType type) {
  auto baseType = type_dyn_cast<FIRRTLBaseType>(unwrap(type));
  assert(baseType && "unexpected type, must be base type");
  return wrap(baseType.getMaskType());
}

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

MlirAttribute firrtlAttrGetConvention(MlirContext ctx,
                                      FIRRTLConvention convention) {
  Convention value;

  switch (convention) {
  case FIRRTL_CONVENTION_INTERNAL:
    value = Convention::Internal;
    break;
  case FIRRTL_CONVENTION_SCALARIZED:
    value = Convention::Scalarized;
    break;
  }

  return wrap(ConventionAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetLayerConvention(MlirContext ctx,
                                           FIRRTLLayerConvention convention) {
  LayerConvention value;

  switch (convention) {
  case FIRRTL_LAYER_CONVENTION_BIND:
    value = LayerConvention::Bind;
    break;
  case FIRRTL_LAYER_CONVENTION_INLINE:
    value = LayerConvention::Inline;
    break;
  }

  return wrap(LayerConventionAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetPortDirs(MlirContext ctx, size_t count,
                                    const FIRRTLDirection *dirs) {
  static_assert(FIRRTL_DIRECTION_IN ==
                static_cast<std::underlying_type_t<Direction>>(Direction::In));
  static_assert(FIRRTL_DIRECTION_OUT ==
                static_cast<std::underlying_type_t<Direction>>(Direction::Out));

  // FIXME: The `reinterpret_cast` here may voilate strict aliasing rule. Is
  // there a better way?
  return wrap(direction::packAttribute(
      unwrap(ctx), ArrayRef(reinterpret_cast<const Direction *>(dirs), count)));
}

MlirAttribute firrtlAttrGetParamDecl(MlirContext ctx, MlirIdentifier name,
                                     MlirType type, MlirAttribute value) {
  return wrap(ParamDeclAttr::get(unwrap(ctx), unwrap(name), unwrap(type),
                                 unwrap(value)));
}

MlirAttribute firrtlAttrGetNameKind(MlirContext ctx, FIRRTLNameKind nameKind) {
  NameKindEnum value;

  switch (nameKind) {
  case FIRRTL_NAME_KIND_DROPPABLE_NAME:
    value = NameKindEnum::DroppableName;
    break;
  case FIRRTL_NAME_KIND_INTERESTING_NAME:
    value = NameKindEnum::InterestingName;
    break;
  }

  return wrap(NameKindEnumAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetRUW(MlirContext ctx, FIRRTLRUW ruw) {
  RUWBehavior value;

  switch (ruw) {
  case FIRRTL_RUW_UNDEFINED:
    value = RUWBehavior::Undefined;
    break;
  case FIRRTL_RUW_OLD:
    value = RUWBehavior::Old;
    break;
  case FIRRTL_RUW_NEW:
    value = RUWBehavior::New;
    break;
  }

  return wrap(RUWBehaviorAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetMemInit(MlirContext ctx, MlirIdentifier filename,
                                   bool isBinary, bool isInline) {
  return wrap(
      MemoryInitAttr::get(unwrap(ctx), unwrap(filename), isBinary, isInline));
}

MlirAttribute firrtlAttrGetMemDir(MlirContext ctx, FIRRTLMemDir dir) {
  MemDirAttr value;

  switch (dir) {
  case FIRRTL_MEM_DIR_INFER:
    value = MemDirAttr::Infer;
    break;
  case FIRRTL_MEM_DIR_READ:
    value = MemDirAttr::Read;
    break;
  case FIRRTL_MEM_DIR_WRITE:
    value = MemDirAttr::Write;
    break;
  case FIRRTL_MEM_DIR_READ_WRITE:
    value = MemDirAttr::ReadWrite;
    break;
  }

  return wrap(MemDirAttrAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetEventControl(MlirContext ctx,
                                        FIRRTLEventControl eventControl) {
  EventControl value;

  switch (eventControl) {
  case FIRRTL_EVENT_CONTROL_AT_POS_EDGE:
    value = EventControl::AtPosEdge;
    break;
  case FIRRTL_EVENT_CONTROL_AT_NEG_EDGE:
    value = EventControl::AtNegEdge;
    break;
  case FIRRTL_EVENT_CONTROL_AT_EDGE:
    value = EventControl::AtEdge;
    break;
  }

  return wrap(EventControlAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetIntegerFromString(MlirType type, unsigned numBits,
                                             MlirStringRef str, uint8_t radix) {
  auto value = APInt{numBits, unwrap(str), radix};
  return wrap(IntegerAttr::get(unwrap(type), value));
}

FIRRTLValueFlow firrtlValueFoldFlow(MlirValue value, FIRRTLValueFlow flow) {
  Flow flowValue;

  switch (flow) {
  case FIRRTL_VALUE_FLOW_NONE:
    flowValue = Flow::None;
    break;
  case FIRRTL_VALUE_FLOW_SOURCE:
    flowValue = Flow::Source;
    break;
  case FIRRTL_VALUE_FLOW_SINK:
    flowValue = Flow::Sink;
    break;
  case FIRRTL_VALUE_FLOW_DUPLEX:
    flowValue = Flow::Duplex;
    break;
  }

  auto flowResult = firrtl::foldFlow(unwrap(value), flowValue);

  switch (flowResult) {
  case Flow::None:
    return FIRRTL_VALUE_FLOW_NONE;
  case Flow::Source:
    return FIRRTL_VALUE_FLOW_SOURCE;
  case Flow::Sink:
    return FIRRTL_VALUE_FLOW_SINK;
  case Flow::Duplex:
    return FIRRTL_VALUE_FLOW_DUPLEX;
  }
  llvm_unreachable("invalid flow");
}

bool firrtlImportAnnotationsFromJSONRaw(
    MlirContext ctx, MlirStringRef annotationsStr,
    MlirAttribute *importedAnnotationsArray) {
  auto annotations = json::parse(unwrap(annotationsStr));
  if (!annotations) {
    return false;
  }

  auto *ctxUnwrapped = unwrap(ctx);

  json::Path::Root root;
  SmallVector<Attribute> annos;
  if (!importAnnotationsFromJSONRaw(annotations.get(), annos, root,
                                    ctxUnwrapped)) {
    return false;
  }

  *importedAnnotationsArray = wrap(ArrayAttr::get(ctxUnwrapped, annos));
  return true;
}
