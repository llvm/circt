//===- GenStatesStruct.cpp - hw generate struct Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass collects elements: output reg of module, local reg / mem  in
// current module and signal with attribute of trigger. Form a struct with these
// elements and their clone member which will be used to represent previous one
// of the member state. The struct will be stored in the global variable and the
// pointer of the struct will be passed to the module. The module will be
// overrided with the new struct pointer.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LLVM.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/Casting.h"

using namespace circt;
using namespace mlir;
using namespace sv;
using namespace hw;

//===----------------------------------------------------------------------===//
// StateStructGenerate Pass
//===----------------------------------------------------------------------===//
/// The Structure stores the information of the port of the module.
struct PortInfoCollect {
  StringRef portName;
  Type portType;
  int direction;
  bool isTrigger;
};

/// The Structure stores the information of the reg of the module.
struct RegInfoCollect {
  StringRef regName;
  Type regType;
  bool isOutputReg;
};

// Declare the class of GenStateStruct Pass.
namespace {
class GenStateStruct {
  // The flag to decide whether to generate the information of the ports which
  // will be packed as structs that stored in the global variable.
  bool emitNoneInfo;

  /// The function to check for the existence of the pair to be searched for
  /// given the input.
  template <typename T>
  bool isIntInPairPart(SmallVector<std::pair<T, unsigned>> &vec, T target);

  /// The function to find the type of the operation in the container if the
  /// type is found by another value in the pair-vector.
  template <typename T>
  T findOperationType(SmallVector<std::pair<mlir::Operation *, T>> &opTypePairs,
                      mlir::Operation *op);

  /// The function to check whether the operation is a leaf module.
  /// The leaf module is the module which does not have any instance that comes
  /// from other modules. The leaf module will be the end of the instance graph.
  bool isleafModule(Operation *op);

  /// The function to convert the port direction to the integer data.
  int convertPortDirection(PortInfo &pInfo);

  /// The function to check whether this register is connected to the output
  /// port, which we call it output register.
  bool isOutputReg(sv::RegOp op);

  /// The function to override current module with the new struct pointer.
  void overrideModule(
      InstanceGraph &instanceGraph, HWModuleOp op, LLVM::LLVMStructType type,
      SmallVector<std::pair<mlir::Operation *, LLVM::LLVMPointerType>>
          &externTypeCollect);

  /// The function to override extern module with the new struct pointer.
  /// Its struct pointer is Opaque type which is used to solve cross-translation
  /// unit problem
  void overrideExternModule(
      MLIRContext *ctx, HWModuleExternOp op,
      SmallVector<std::pair<mlir::Operation *, LLVM::LLVMPointerType>>
          &externTypeCollect);

  /// The function to override instance relationship with the newly created
  /// struct pointer for submodule of current module.
  void overrideInstance(
      OpBuilder builder, Value operand, InstanceGraph &instanceGraph,
      SmallVector<std::pair<mlir::Operation *, LLVM::LLVMPointerType>>
          &externTypeCollect,
      hw::InstanceOp inst, LLVM::GEPOp gep, LLVM::ConstantOp zeroC);

  /// The function to convert hw::InOutType to LLVM compatible type.
  /// Attention: we support InOutType / nested ArrayType /nested
  /// UnpackedArrayType to convert to LLVM compatible Type. Other nested
  /// InOutType will hit the assertions in program and stop execution of
  /// this pass.
  Type convertInOutType(hw::InOutType type);
  LLVM::LLVMStructType getExternModulePortTy(MLIRContext *ctx, bool noOpaqueTy,
                                             HWModuleExternOp op);

public:
  GenStateStruct(bool emitNoneInfo) : emitNoneInfo(emitNoneInfo) {}
  /// The function to add prefix to the given name and return the new name.
  std::string addPrefixToName(StringRef name);

  /// This function is used to generate the struct type based on the given
  /// infomation of current module. The struct type is packed defaultly.
  /// If the module is an extern module which only have a declaration, generate
  /// opaque struct type for it.
  LLVM::LLVMStructType createStructType(MLIRContext *ctx, HWModuleOp op,
                                        InstanceGraph &instGraph,
                                        SmallVector<PortInfoCollect> &pInfo,
                                        SmallVector<RegInfoCollect> &rInfo);

  /// The function to create the global struct that is used to store all state
  /// values and ports infomation.
  LLVM::GlobalOp createGlobalStruct(Location loc, OpBuilder &builder,
                                    ModuleOp module, StringRef name,
                                    LLVM::LLVMStructType type);

  /// The function to initialize the global struct with default value 0, and
  /// return initialized struct object.
  void initializeGlobalStruct(OpBuilder &builder, HWModuleOp op,
                              LLVM::GlobalOp globalOp,
                              LLVM::LLVMStructType type,
                              InstanceGraph &instGraph,
                              SmallVector<PortInfoCollect> &pInfo,
                              SmallVector<RegInfoCollect> &rInfo);

  /// The function is used to collect all ports info above the given module.
  /// The collected info: port name, port type, port direction and whether it is
  /// trigger signal.
  void collectPortInfo(HWModuleOp op, SmallVector<PortInfoCollect> &pInfo);

  /// The function is used to collect all regs info in the given module.
  /// The collected info: reg name, reg type and whether it is output reg.
  void collectRegInfo(HWModuleOp op, SmallVector<RegInfoCollect> &rInfo);

  /// The function to override all modules in InstanceGraph based on a
  /// depth-first algorithm.
  void dfsOverrideModules(
      MLIRContext *ctx,
      SmallVector<std::pair<mlir::Operation *, LLVM::LLVMStructType>>
          &typeCollect,
      SmallVector<std::pair<mlir::Operation *, LLVM::LLVMPointerType>>
          &externTypeCollect,
      InstanceGraph &instanceGraph, DenseSet<Operation *> &overrided,
      Operation *topOp);

  /// The function to get the number of secondary trigger signal.
  unsigned getNumOfSecondaryTriggerNum(HWModuleOp op);
};

struct StateStructGeneratePass
    : public StateStructGenerateBase<StateStructGeneratePass> {
  void runOnOperation() override;

  using StateStructGenerateBase<StateStructGeneratePass>::emitNoneInfo;
};
} // namespace

bool GenStateStruct::isOutputReg(sv::RegOp op) {
  bool outputReg = false;
  for (auto *user : op->getUsers()) {
    if (isa<hw::OutputOp>(user)) {
      outputReg = true;
      break;
    }
  }
  return outputReg;
}

int GenStateStruct::convertPortDirection(PortInfo &pInfo) {
  // The value 0 is used to represent the port direction is unknown.
  // when other function handle the value 0, it will hit the assertion.
  if (pInfo.isInput())
    return 1;

  if (pInfo.isOutput())
    return 2;

  if (pInfo.isInOut())
    return 3;

  return 0;
}

// Check the value is stored in the Container or not.
template <typename T>
bool GenStateStruct::isIntInPairPart(SmallVector<std::pair<T, unsigned>> &vec,
                                     T target) {
  // Traverse all elements in contanier.
  return std::any_of(vec.begin(), vec.end(),
                     [&](const std::pair<T, unsigned> &p) {
                       // Check if there is the target value, return result.
                       return p.first == target;
                     });
}

template <typename T>
T GenStateStruct::findOperationType(
    SmallVector<std::pair<mlir::Operation *, T>> &opTypePairs,
    mlir::Operation *op) {
  auto it = llvm::find_if(opTypePairs,
                          [op](const std::pair<mlir::Operation *, T> &pair) {
                            return pair.first == op;
                          });
  if (it != opTypePairs.end())
    return it->second;
  // throw an exception, if not find the type.
  assert(it == opTypePairs.end() && "Can't find the type of the operation");
  return nullptr;
}

bool GenStateStruct::isleafModule(Operation *op) {
  bool isLeaf = false;
  if (isa<hw::HWModuleOp>(op))
    isLeaf = dyn_cast<hw::HWModuleOp>(op).getOps<hw::InstanceOp>().empty();
  if (isa<hw::HWModuleExternOp>(op))
    isLeaf = true;

  return isLeaf;
}

std::string GenStateStruct::addPrefixToName(StringRef name) {
  std::string prefix = "_Struct_";
  prefix.append(name.data(), name.size());
  return prefix;
}

Type GenStateStruct::convertInOutType(hw::InOutType type) {
  auto elementTy = getInOutElementType(type);
  // Convert the element type to LLVM compatible type.
  if (elementTy.isa<IntegerType>()) {
    return IntegerType::get(type.getContext(),
                            elementTy.getIntOrFloatBitWidth());
  }
  if (elementTy.isa<hw::ArrayType>()) {
    return LLVM::LLVMArrayType::get(
        getAnyHWArrayElementType(elementTy),
        dyn_cast<hw::ArrayType>(elementTy).getSize());
  }
  if (elementTy.isa<hw::UnpackedArrayType>()) {
    return LLVM::LLVMArrayType::get(
        getAnyHWArrayElementType(elementTy),
        dyn_cast<hw::UnpackedArrayType>(elementTy).getSize());
  }
  return elementTy;
}

LLVM::LLVMStructType
GenStateStruct::getExternModulePortTy(MLIRContext *ctx, bool noOpaqueTy,
                                      HWModuleExternOp op) {
  SmallVector<Type> extModuleTy;
  ModulePortInfo externPInfo = op.getPortList();

  // The noOpaqueTy decides whether to create a opaque type or not.
  if (noOpaqueTy) {
    for (auto pInfo : externPInfo) {
      extModuleTy.push_back(pInfo.type);
    }
    return LLVM::LLVMStructType::getLiteral(ctx, extModuleTy,
                                            /*isPacked=*/true);
  }

  return LLVM::LLVMStructType::getOpaque(addPrefixToName(op.getName()), ctx);
}

LLVM::LLVMStructType GenStateStruct::createStructType(
    MLIRContext *ctx, HWModuleOp op, InstanceGraph &instGraph,
    SmallVector<PortInfoCollect> &pInfo, SmallVector<RegInfoCollect> &rInfo) {
  auto i1Ty = IntegerType::get(ctx, 1);
  auto i2Ty = IntegerType::get(ctx, 2);
  auto i8Ty = IntegerType::get(ctx, 8);

  // Create three container to store the type of 1. ports & regs, 2. triiger
  // signal & delay regs. 3. packed struct for storing all critical information,
  // the last struct pointer that is used to create instance relationships.
  SmallVector<Type> clonedPortTy;
  SmallVector<Type> delayedTy;
  SmallVector<Type> packedStructTy;

  LLVM::LLVMStructType innerStructTy;

  // Generate the type of the port part.
  for (PortInfoCollect pstruct : pInfo) {
    if (pstruct.portType.isa<InOutType>())
      pstruct.portType =
          convertInOutType(dyn_cast<hw::InOutType>(pstruct.portType));

    // Add emitNoneInfo to control generation of port-info type.
    if (!emitNoneInfo) {
      auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, pstruct.portName.size());
      if (pstruct.direction != 2)
        innerStructTy = LLVM::LLVMStructType::getLiteral(
            ctx, {arrayTy, pstruct.portType, i2Ty, i1Ty},
            /*isPacked=*/true);
      else
        innerStructTy = LLVM::LLVMStructType::getLiteral(
            ctx, {arrayTy, pstruct.portType, i2Ty},
            /*isPacked=*/true);
      packedStructTy.push_back(innerStructTy);
    }
    clonedPortTy.push_back(pstruct.portType);

    if (pstruct.isTrigger)
      delayedTy.push_back(pstruct.portType);
  }

  // Generate the type for the secondary trigger signal part.
  for (size_t i = 0, e = getNumOfSecondaryTriggerNum(op); i < e; i++)
    delayedTy.push_back(i1Ty);

  // Generate the type for register part.
  for (RegInfoCollect rstruct : rInfo) {
    if (rstruct.regType.isa<InOutType>())
      rstruct.regType =
          convertInOutType(dyn_cast<hw::InOutType>(rstruct.regType));

    // Add emitNoneInfo to control generation of reg-info type.
    if (!emitNoneInfo) {
      auto arrayTy = LLVM::LLVMArrayType::get(i8Ty, rstruct.regName.size());
      innerStructTy = LLVM::LLVMStructType::getLiteral(
          ctx, {arrayTy, rstruct.regType, i1Ty},
          /*isPacked=*/true);
      packedStructTy.push_back(innerStructTy);
    }

    clonedPortTy.push_back(rstruct.regType);
    delayedTy.push_back(rstruct.regType);
  }

  // Generate the pointer type for the submodule part.
  for (auto inst : llvm::make_early_inc_range(op.getOps<hw::InstanceOp>())) {
    if (auto refedModule =
            dyn_cast<HWModuleOp>(*instGraph.getReferencedModule(inst)))
      innerStructTy = LLVM::LLVMStructType::getIdentified(
          ctx, addPrefixToName(refedModule.getName()));
    else
      innerStructTy = getExternModulePortTy(
          ctx, false,
          dyn_cast<HWModuleExternOp>(*instGraph.getReferencedModule(inst)));
    packedStructTy.push_back(LLVM::LLVMPointerType::get(innerStructTy));
  }

  // Combine all container to create the final struct type.
  clonedPortTy.insert(clonedPortTy.end(), delayedTy.begin(), delayedTy.end());
  clonedPortTy.insert(clonedPortTy.end(), packedStructTy.begin(),
                      packedStructTy.end());

  // The NewIdentified struct type has an explicit name and type. If we want to
  // change the struct type to add or remove some members,we need to remove the
  // whole struct definition beacuse the Indentified Struct does not support
  // modify structure directly.
  return LLVM::LLVMStructType::getNewIdentified(
      ctx, addPrefixToName(op.getName()), clonedPortTy,
      /*isPacked=*/true);
}

LLVM::GlobalOp GenStateStruct::createGlobalStruct(Location loc,
                                                  OpBuilder &builder,
                                                  ModuleOp module,
                                                  StringRef name,
                                                  LLVM::LLVMStructType type) {
  LLVM::GlobalOp global;

  // Create an empty global struct which is uninitialized.
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/false,
                                            LLVM::Linkage::Internal, name,
                                            Attribute(),
                                            /*alignment=*/0, /*addrspace=*/0);
  }
  return global;
}

// TODO: initialize the global struct with completely random value. Maybe it is
// a new option for this pass.
void GenStateStruct::initializeGlobalStruct(
    OpBuilder &builder, HWModuleOp op, LLVM::GlobalOp globalOp,
    LLVM::LLVMStructType type, InstanceGraph &instGraph,
    SmallVector<PortInfoCollect> &pInfo, SmallVector<RegInfoCollect> &rInfo) {
  // Initialize the global struct with specified information.
  Location loc = globalOp.getLoc();
  MLIRContext *ctx = builder.getContext();
  Region &region = globalOp.getInitializerRegion();
  Block *block = builder.createBlock(&region);
  builder.setInsertionPoint(block, block->begin());

  LLVM::LLVMStructType innerStructTy;
  IntegerType i1Ty = IntegerType::get(ctx, 1);
  IntegerType i2Ty = IntegerType::get(ctx, 2);
  IntegerType i8Ty = IntegerType::get(builder.getContext(), 8);

  SmallVector<Value> allInfo;
  SmallVector<Value> allDelayed;
  SmallVector<Value> allPortReg;

  // Create operations which initialize the port part.
  for (PortInfoCollect pstruct : pInfo) {
    if (pstruct.portType.isa<InOutType>())
      pstruct.portType =
          convertInOutType(dyn_cast<hw::InOutType>(pstruct.portType));

    if (!emitNoneInfo) {
      SmallVector<LLVM::ConstantOp> op;
      SmallVector<Value> innerPacked;
      LLVM::LLVMArrayType arrayTy =
          LLVM::LLVMArrayType::get(i8Ty, pstruct.portName.size());

      // Convert the port name to char array.
      for (char c : pstruct.portName)
        op.push_back(builder.create<LLVM::ConstantOp>(loc, i8Ty, int(c)));

      Value array = builder.create<LLVM::UndefOp>(loc, arrayTy);
      for (size_t i = 0, e = arrayTy.getNumElements(); i < e; ++i)
        array = builder.create<LLVM::InsertValueOp>(loc, array, op[i], i);

      innerPacked.push_back(array);

      if (pstruct.direction != 2 && pstruct.direction != 0) {
        innerStructTy = LLVM::LLVMStructType::getLiteral(
            ctx, {arrayTy, pstruct.portType, i2Ty, i1Ty},
            /*isPacked=*/true);

        // If the data of port direction is invalid, throw an error and stop
        // initialization.
      } else if (pstruct.direction == 0) {
        emitError(loc, "Failed to initialize this global struct beacuse of "
                       "invalid port direction!");
      } else {
        innerStructTy = LLVM::LLVMStructType::getLiteral(
            ctx, {arrayTy, pstruct.portType, i2Ty},
            /*isPacked=*/true);
      }

      innerPacked.push_back(
          builder.create<LLVM::ConstantOp>(loc, pstruct.portType, 0));
      innerPacked.push_back(
          builder.create<LLVM::ConstantOp>(loc, i2Ty, pstruct.direction));
      innerPacked.push_back(
          builder.create<LLVM::ConstantOp>(loc, i1Ty, pstruct.isTrigger));
      Value packedStruct = builder.create<LLVM::UndefOp>(loc, innerStructTy);

      for (size_t j = 0, e = innerStructTy.getBody().size(); j < e; ++j) {
        packedStruct = builder.create<LLVM::InsertValueOp>(loc, packedStruct,
                                                           innerPacked[j], j);
      }
      allInfo.push_back(packedStruct);
    }

    // If the port is a trigger, add it to the allDelayed vector.
    allPortReg.push_back(
        builder.create<LLVM::ConstantOp>(loc, pstruct.portType, 0));
    if (pstruct.isTrigger)
      allDelayed.push_back(
          builder.create<LLVM::ConstantOp>(loc, pstruct.portType, 0));
  }

  // Create operations which initialize the register part.
  for (size_t i = 0, e = getNumOfSecondaryTriggerNum(op); i < e; i++)
    allDelayed.push_back(builder.create<LLVM::ConstantOp>(loc, i1Ty, 0));

  // Create operations which initialize the register part.
  for (RegInfoCollect rstruct : rInfo) {
    if (rstruct.regType.isa<InOutType>())
      rstruct.regType =
          convertInOutType(dyn_cast<hw::InOutType>(rstruct.regType));

    if (!emitNoneInfo) {
      SmallVector<LLVM::ConstantOp> op;
      SmallVector<Value> innerPacked;
      LLVM::LLVMArrayType arrayTy =
          LLVM::LLVMArrayType::get(i8Ty, rstruct.regName.size());

      for (char c : rstruct.regName)
        op.push_back(builder.create<LLVM::ConstantOp>(loc, i8Ty, int(c)));

      Value array = builder.create<LLVM::UndefOp>(loc, arrayTy);
      for (size_t i = 0, e = arrayTy.getNumElements(); i < e; ++i)
        array = builder.create<LLVM::InsertValueOp>(loc, array, op[i], i);

      innerPacked.push_back(array);

      if (isa<IntegerType>(rstruct.regType))
        innerPacked.push_back(
            builder.create<LLVM::ConstantOp>(loc, rstruct.regType, 0));
      else if (isa<LLVM::LLVMArrayType>(rstruct.regType)) {
        auto regTy = dyn_cast<LLVM::LLVMArrayType>(rstruct.regType);
        auto zeroC =
            builder.create<LLVM::ConstantOp>(loc, regTy.getElementType(), 0);
        Value array = builder.create<LLVM::UndefOp>(loc, regTy);

        for (size_t i = 0; i < regTy.getNumElements(); i++)
          array = builder.create<LLVM::InsertValueOp>(loc, array, zeroC, i);

        innerPacked.push_back(array);
      } else
        emitError(loc, "Failed to initialize this global struct beacuse of "
                       "invalid register type!");

      innerPacked.push_back(
          builder.create<LLVM::ConstantOp>(loc, i1Ty, rstruct.isOutputReg));

      innerStructTy = LLVM::LLVMStructType::getLiteral(
          ctx, {arrayTy, rstruct.regType, i1Ty},
          /*isPacked=*/true);
      Value packedStruct = builder.create<LLVM::UndefOp>(loc, innerStructTy);

      for (size_t j = 0, e = innerStructTy.getBody().size(); j < e; ++j) {
        packedStruct = builder.create<LLVM::InsertValueOp>(loc, packedStruct,
                                                           innerPacked[j], j);
      }
      allInfo.push_back(packedStruct);
    }

    // If the register is an output register, add it to the allDelayed vector.
    if (isa<IntegerType>(rstruct.regType)) {
      allPortReg.push_back(
          builder.create<LLVM::ConstantOp>(loc, rstruct.regType, 0));
      allDelayed.push_back(
          builder.create<LLVM::ConstantOp>(loc, rstruct.regType, 0));

      // if the register is an array (one dimension or more), initialize
      // it with inserting value 0 to every elements in it.
    } else if (isa<LLVM::LLVMArrayType>(rstruct.regType)) {
      auto regTy = dyn_cast<LLVM::LLVMArrayType>(rstruct.regType);
      auto zeroC =
          builder.create<LLVM::ConstantOp>(loc, regTy.getElementType(), 0);
      Value array = builder.create<LLVM::UndefOp>(loc, regTy);

      for (size_t k = 0, e = regTy.getNumElements(); k < e; k++)
        array = builder.create<LLVM::InsertValueOp>(loc, array, zeroC, k);

      allPortReg.push_back(array);
      allDelayed.push_back(array);

    } else {
      emitError(loc, "Failed to initialize this global struct beacuse of "
                     "invalid register type!");
    }
  }
  // Create operations which initialize the submodule part.
  for (auto inst : llvm::make_early_inc_range(op.getOps<hw::InstanceOp>())) {
    if (auto refedModule =
            dyn_cast<HWModuleOp>(*instGraph.getReferencedModule(inst)))
      innerStructTy = LLVM::LLVMStructType::getIdentified(
          ctx, addPrefixToName(refedModule.getName()));
    else
      innerStructTy = getExternModulePortTy(
          ctx, false,
          dyn_cast<HWModuleExternOp>(*instGraph.getReferencedModule(inst)));

    Value nullPtr = builder.create<LLVM::NullOp>(
        loc, LLVM::LLVMPointerType::get(innerStructTy));
    allInfo.push_back(nullPtr);
  }

  // Combine all the operations that are used for initializing in order.
  Value finalStruct = builder.create<LLVM::UndefOp>(loc, type);
  allPortReg.insert(allPortReg.end(), allDelayed.begin(), allDelayed.end());
  allPortReg.insert(allPortReg.end(), allInfo.begin(), allInfo.end());

  for (size_t i = 0, e = type.getBody().size(); i < e; ++i) {
    finalStruct =
        builder.create<LLVM::InsertValueOp>(loc, finalStruct, allPortReg[i], i);
  }
  builder.create<LLVM::ReturnOp>(loc, finalStruct);
}

void GenStateStruct::collectPortInfo(HWModuleOp op,
                                     SmallVector<PortInfoCollect> &pInfo) {
  SmallVector<unsigned> triIndexCollect;
  auto argSize = op.getNumInOrInoutPorts();
  auto resSize = op.getResultTypes().size();
  auto arrayAttr = cast_if_present<ArrayAttr>(op->getAttr("sv.trigger"));

  if (arrayAttr)
    for (auto triAttr : arrayAttr)
      triIndexCollect.push_back(
          cast<IntegerAttr>(triAttr).getValue().getZExtValue());

  ModulePortInfo port = op.getPortList();
  // Get and store the port information, if it is trigger signal, then duplicate
  // it in the port list.
  for (size_t i = 0; i < argSize; i++) {
    bool isContain = false;
    if (arrayAttr != nullptr)
      isContain = std::find(triIndexCollect.begin(), triIndexCollect.end(),
                            i) != triIndexCollect.end();

    if (isContain)
      pInfo.push_back({port.at(i).name, port.at(i).type,
                       convertPortDirection(port.at(i)), true});
    else
      pInfo.push_back({port.at(i).name, port.at(i).type,
                       convertPortDirection(port.at(i)), false});
  }
  for (size_t j = argSize; j < argSize + resSize; j++) {
    pInfo.push_back({port.at(j).name, port.at(j).type,
                     convertPortDirection(port.at(j)), false});
  }
}

void GenStateStruct::collectRegInfo(HWModuleOp op,
                                    SmallVector<RegInfoCollect> &rInfo) {
  for (auto regOp : llvm::make_early_inc_range(op.getOps<sv::RegOp>())) {
    bool outputReg = isOutputReg(regOp);
    rInfo.push_back({regOp.getName(), regOp.getType(), outputReg});
  }
}

unsigned GenStateStruct::getNumOfSecondaryTriggerNum(HWModuleOp op) {
  unsigned secondaryTriggerNum = 0;
  // First, check if there is any sv.trigger attribute in the operation.
  for (auto &innerOp : op.getOps()) {
    if (innerOp.hasAttr("sv.trigger")) {
      // If there is ,convert the triiger attribute to ArrayAttr and get its
      // size().
      auto arrayAttr =
          dyn_cast_or_null<ArrayAttr>(innerOp.getAttr("sv.trigger"));
      secondaryTriggerNum += arrayAttr.size();
    }
  }
  return secondaryTriggerNum;
}

void GenStateStruct::overrideModule(
    InstanceGraph &instanceGraph, HWModuleOp op, LLVM::LLVMStructType type,
    SmallVector<std::pair<mlir::Operation *, LLVM::LLVMPointerType>>
        &externTypeCollect) {
  auto loc = op.getLoc();

  // Collect numbers of every parts in struct to calculate indexes for GEPOp
  unsigned inputPortNum = op.getNumArguments();
  unsigned outputPortNum = op.getNumResults();
  unsigned regNum = 0;
  unsigned triggerVNum = 0;
  unsigned instanceNum = 0;

  SmallVector<sv::RegOp> allReg;
  ArrayRef<mlir::Type> typeList = type.getBody();

  for (sv::RegOp regOp : llvm::make_early_inc_range(op.getOps<sv::RegOp>())) {
    allReg.push_back(regOp);
    regNum++;
  }

  if (auto arrayAttr = dyn_cast_or_null<ArrayAttr>(op->getAttr("sv.trigger")))
    triggerVNum += arrayAttr.size();
  triggerVNum += getNumOfSecondaryTriggerNum(op);

  OpBuilder builder = OpBuilder::atBlockBegin(op.getBodyBlock());
  // auto saveInPoint = builder.saveInsertionPoint();

  // Prepare basic info for newly added input port in PortInfo form.
  StringAttr strAttr =
      StringAttr::get(builder.getContext(), "ptr_struct_" + op.getName());
  SmallVector<unsigned> arrEraseIn, arrEraseOut;

  // Insert the struct pointer that preserves all ports and states variables in
  // current module, the insertion index is 0, and get the newly added arg as
  // arg0 variable.
  op.prependInput(strAttr, LLVM::LLVMPointerType::get(type));

  // Create constantOp that is used in GEPOp as start index.
  auto i32Ty = IntegerType::get(builder.getContext(), 32);
  auto zeroC = builder.create<LLVM::ConstantOp>(loc, i32Ty,
                                                builder.getI32IntegerAttr(0));
  auto arg0 = op.getArgument(0);
  // Replace uses of operations from original block arguments to newly added
  // struct pointer argument.
  for (size_t i = 1; i < inputPortNum + 1; i++) {
    auto arg = op.getArgument(i);

    // If this block argument has users, do replacement to its uses chain.
    if (!arg.use_empty()) {
      auto indexC = builder.create<LLVM::ConstantOp>(
          loc, i32Ty, builder.getI32IntegerAttr(i - 1));
      auto gep = builder.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(typeList[i - 1]), arg0,
          ArrayRef<Value>({zeroC, indexC}), /*isInbounds=*/true);

      auto load = builder.create<LLVM::LoadOp>(loc, gep);

      // Update the usechain of argument to the member that is geted from newly
      // added arg.
      arg.replaceAllUsesWith(load);
      indexC.erase();
    }
  }

  // Reset Insertion point to the end of block.
  auto output = cast<hw::OutputOp>(op.getBodyBlock()->getTerminator());
  builder.setInsertionPointToEnd(op.getBodyBlock());
  auto oldPoint = builder.saveInsertionPoint();

  // Alter original output dataflow to update states of inner members in struct
  for (unsigned j = 0; j < outputPortNum; j++) {
    auto operand = output->getOperand(j);

    // Create operations to store corresponding result values.
    auto resIndexC = builder.create<LLVM::ConstantOp>(
        loc, i32Ty, builder.getI32IntegerAttr(inputPortNum + j));
    auto gep = builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(typeList[inputPortNum + j]), arg0,
        ArrayRef<Value>({zeroC, resIndexC}), /*isInbounds=*/true);
    builder.create<LLVM::StoreOp>(loc, operand, gep);
    resIndexC.erase();
  }

  // Collect the ports indexes which are needed to be removed
  for (size_t i = 1; i < inputPortNum + 1; i++)
    arrEraseIn.push_back(i);

  for (size_t j = 0; j < outputPortNum; j++)
    arrEraseOut.push_back(j);

  // Modify ports list of the module to make the pointer of struct beacome the
  // only input value in the module.
  modifyModulePorts(op, {}, {}, arrEraseIn, arrEraseOut, op.getBodyBlock());

  // Remove the original outputOp and replace it with outputOp that do not
  // output any value.
  builder.create<hw::OutputOp>(loc, std::nullopt);
  output.erase();

  // Overriding Instances in current module.
  for (auto inst : llvm::make_early_inc_range(op.getOps<hw::InstanceOp>())) {
    Value bitcast;
    unsigned instIndex = 0;
    builder.setInsertionPoint(inst);

    // if disable emitNoneInfo, calculate the index of instance includes info
    // elements members.
    if (!emitNoneInfo)
      instIndex = (inputPortNum + outputPortNum) * 2 + regNum * 3 +
                  triggerVNum + instanceNum;
    else
      instIndex =
          inputPortNum + outputPortNum + regNum * 2 + triggerVNum + instanceNum;

    auto resIndexC = builder.create<LLVM::ConstantOp>(
        loc, i32Ty, builder.getI32IntegerAttr(instIndex));
    auto gep = builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(typeList[instIndex]), arg0,
        ArrayRef<Value>({zeroC, resIndexC}), /*isInbounds=*/true);
    auto load = builder.create<LLVM::LoadOp>(loc, gep);
    resIndexC.erase();

    auto loadTy = dyn_cast<LLVM::LLVMPointerType>(load.getType());
    auto instStructTy = dyn_cast<LLVM::LLVMStructType>(loadTy.getElementType());

    if (!instStructTy.isOpaque()) {
      bitcast = load;
    } else {
      auto exPtrTy = findOperationType(externTypeCollect,
                                       instanceGraph.getReferencedModule(inst));
      bitcast = builder.create<LLVM::BitcastOp>(loc, exPtrTy, load);
    }

    // Add logic of transfering states to instances.
    for (size_t i = 0; i < inst.getNumOperands(); i++) {
      auto operand = inst.getOperand(i);
      auto type = operand.getType();

      if (operand.getType().isa<hw::InOutType>())
        type = convertInOutType(dyn_cast<hw::InOutType>(type));

      auto operandIndexC = builder.create<LLVM::ConstantOp>(
          loc, i32Ty, builder.getI32IntegerAttr(i));
      auto state = builder.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(type), bitcast,
          ArrayRef<Value>({zeroC, operandIndexC}), /*isInbounds=*/true);
      builder.create<LLVM::StoreOp>(loc, operand, state);
      operandIndexC.erase();
    }

    overrideInstance(builder, load, instanceGraph, externTypeCollect, inst, gep,
                     zeroC);

    builder.restoreInsertionPoint(oldPoint);
    instanceNum++;
  }
}

void GenStateStruct::overrideExternModule(
    MLIRContext *ctx, hw::HWModuleExternOp op,
    SmallVector<std::pair<mlir::Operation *, LLVM::LLVMPointerType>>
        &externTypeCollect) {
  auto inputPortNum = op.getNumArguments();
  auto outputPortNum = op.getNumResults();

  // Collect the extern module and its corresponding pointer type that will
  // bitcast to.
  std::pair<mlir::Operation *, LLVM::LLVMPointerType> pair = std::make_pair(
      op.getOperation(),
      LLVM::LLVMPointerType::get(getExternModulePortTy(ctx, true, op)));
  externTypeCollect.push_back(pair);

  // Prepare basic info for newly added input port in PortInfo form.
  hw::PortInfo structPInfo = PortInfo{
      {/*name*/ StringAttr::get(ctx, "ptr_struct_" + op.getName()),
       /*type*/
       LLVM::LLVMPointerType::get(getExternModulePortTy(ctx, false, op)),
       /*direction*/ ModulePort::Direction::Input},
      /*argNum*/ 1,
      /*sym*/ {},
      /*attr*/ {},
      /*location*/ op.getLoc()};

  std::pair<unsigned, PortInfo> arrInsertIn = {0, structPInfo};
  SmallVector<unsigned> arrEraseIn, arrEraseOut;

  for (size_t i = 0; i < inputPortNum; i++)
    arrEraseIn.push_back(i);

  for (size_t j = 0; j < outputPortNum; j++)
    arrEraseOut.push_back(j);

  op.modifyPorts(/*insertInputs*/ arrInsertIn, /*insertOutputs*/ {},
                 /*eraseInputs*/ arrEraseIn, /*eraseOutputs*/ arrEraseOut);
}

void GenStateStruct::overrideInstance(
    OpBuilder builder, Value operand, InstanceGraph &instanceGraph,
    SmallVector<std::pair<mlir::Operation *, LLVM::LLVMPointerType>>
        &externTypeCollect,
    hw::InstanceOp inst, LLVM::GEPOp gep, LLVM::ConstantOp zeroC) {

  // Get necessary info from the original instance
  auto loc = inst.getLoc();
  auto *ctx = builder.getContext();
  auto subModule = instanceGraph.getReferencedModule(inst);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto instName = StringAttr::get(ctx, inst.getInstanceName());
  auto instArgNums = inst.getNumOperands();
  auto nameAttr = StringAttr::get(ctx, "sv.trigger");

  auto gepType = dyn_cast<LLVM::LLVMPointerType>(gep.getType());
  auto refedPointerType =
      dyn_cast<LLVM::LLVMPointerType>(gepType.getElementType());
  auto refedStructType =
      dyn_cast<LLVM::LLVMStructType>(refedPointerType.getElementType());
  auto refedTypeList = refedStructType.getBody();

  // Create new instance with new prototype
  auto newInst = builder.create<hw::InstanceOp>(loc, subModule, instName,
                                                operand, inst.getParameters());

  if (auto triggerAttr = inst->getAttrOfType<ArrayAttr>(nameAttr))
    newInst->setAttr(nameAttr, triggerAttr);

  Value bitcast, subGep;
  ArrayRef<mlir::Type> exPtrTypeList;
  LLVM::LLVMPointerType ptrTy, exPtrTy;

  if (refedStructType.isOpaque()) {
    exPtrTy = findOperationType(externTypeCollect, subModule);
    bitcast = builder.create<LLVM::BitcastOp>(loc, exPtrTy, operand);
    exPtrTypeList =
        dyn_cast<LLVM::LLVMStructType>(exPtrTy.getElementType()).getBody();
  }

  for (size_t i = 0; i < inst.getNumResults(); i++) {
    auto resIndexC = builder.create<LLVM::ConstantOp>(
        loc, i32Ty, builder.getI32IntegerAttr(instArgNums + i));

    if (!refedStructType.isOpaque()) {
      ptrTy = LLVM::LLVMPointerType::get(refedTypeList[instArgNums + i]);
      subGep = builder.create<LLVM::GEPOp>(loc, ptrTy, operand,
                                           ArrayRef<Value>({zeroC, resIndexC}),
                                           /*isInbounds=*/true);
    } else {
      ptrTy = LLVM::LLVMPointerType::get(exPtrTypeList[instArgNums + i]);
      subGep = builder.create<LLVM::GEPOp>(loc, ptrTy, bitcast,
                                           ArrayRef<Value>({zeroC, resIndexC}),
                                           /*isInbounds=*/true);
    }

    auto subLoad = builder.create<LLVM::LoadOp>(loc, subGep);
    inst.getResult(i).replaceAllUsesWith(subLoad);
  }

  // Replace original instance with new one
  instanceGraph.replaceInstance(inst, newInst);
  inst.erase();
}

void GenStateStruct::dfsOverrideModules(
    MLIRContext *ctx,
    SmallVector<std::pair<mlir::Operation *, LLVM::LLVMStructType>>
        &typeCollect,
    SmallVector<std::pair<mlir::Operation *, LLVM::LLVMPointerType>>
        &externTypeCollect,
    InstanceGraph &instanceGraph, DenseSet<Operation *> &overrided,
    Operation *op) {
  // If the current module has been overrided, return.
  if (overrided.count(op))
    return;

  // If the current module is a leaf module, do overriding and then return.
  if (isleafModule(op)) {

    // Override the current module.
    if (isa<hw::HWModuleOp>(op)) {
      auto leafModuleType = findOperationType(typeCollect, op);
      overrideModule(instanceGraph, cast<hw::HWModuleOp>(op), leafModuleType,
                     externTypeCollect);
    }

    if (isa<hw::HWModuleExternOp>(op)) {
      overrideExternModule(ctx, dyn_cast<hw::HWModuleExternOp>(op),
                           externTypeCollect);
    }

    // Mark the current module as overrided.
    overrided.insert(op);
    return;
  }
  // If the current module is not a leaf module, traverse all the instances in
  // this module to ensure all its child modules have been overrided.
  for (auto inst : llvm::make_early_inc_range(
           cast<HWModuleOp>(op).getOps<hw::InstanceOp>())) {
    // Get the child module of the current instance.
    auto childModule = instanceGraph.getReferencedModule(inst);
    dfsOverrideModules(ctx, typeCollect, externTypeCollect, instanceGraph,
                       overrided, childModule);
  }

  // Process the current module overriding.
  auto nonLeafModuleType = findOperationType(typeCollect, op);
  overrideModule(instanceGraph, cast<hw::HWModuleOp>(op), nonLeafModuleType,
                 externTypeCollect);
  overrided.insert(op);
}

void StateStructGeneratePass::runOnOperation() {
  // Intialize class entry.
  GenStateStruct gstateStruct = {emitNoneInfo};

  // Get the instance Graph for updating modules instances.
  InstanceGraph &instGraph = getAnalysis<InstanceGraph>();

  // Collect all struct type of the modules and store them in paired vector.
  SmallVector<std::pair<mlir::Operation *, LLVM::LLVMStructType>> typeCollect;
  SmallVector<std::pair<mlir::Operation *, LLVM::LLVMPointerType>>
      externTypeCollect;

  // Create a denseSet to store the modules that have been overrided
  DenseSet<Operation *> overridedModules;

  // Get builder and context.
  OpBuilder builder(&getContext());
  MLIRContext *ctx = builder.getContext();

  // Generate struct to store state values.
  getOperation().walk([&](HWModuleOp op) {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<mlir::ModuleOp>();

    SmallVector<PortInfoCollect> pInfo;
    SmallVector<RegInfoCollect> rInfo;

    // Collect all the port and reg information.
    gstateStruct.collectPortInfo(op, pInfo);
    gstateStruct.collectRegInfo(op, rInfo);

    // Create struct type for current module.
    LLVM::LLVMStructType type =
        gstateStruct.createStructType(ctx, op, instGraph, pInfo, rInfo);

    // Deal with current module contents and create corresponding struct
    LLVM::GlobalOp global = gstateStruct.createGlobalStruct(
        loc, builder, module, gstateStruct.addPrefixToName(op.getName()), type);

    // Initialize the global struct.
    gstateStruct.initializeGlobalStruct(builder, op, global, type, instGraph,
                                        pInfo, rInfo);

    // Collect all struct types with their operation pointer.
    std::pair<Operation *, LLVM::LLVMStructType> pairType =
        std::make_pair(op.getOperation(), type);
    typeCollect.push_back(pairType);
  });

  getOperation().walk([&](HWModuleOp op) {
    // Execute DFS overrideModules function to override all modules
    gstateStruct.dfsOverrideModules(ctx, typeCollect, externTypeCollect,
                                    instGraph, overridedModules, op);
  });
}
//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//
namespace circt {
std::unique_ptr<mlir::Pass> createStateStructGeneratePass(bool emitNoneInfo) {
  auto pass = std::make_unique<StateStructGeneratePass>();
  pass->emitNoneInfo = emitNoneInfo;
  return pass;
}
} // namespace circt