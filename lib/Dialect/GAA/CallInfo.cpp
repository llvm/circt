//===- CallInfo.h - CallInfo for each GAAModule -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/GAA/CallInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace gaa;

CallInfo::CallInfo(Operation *operation) {
  auto module = llvm::dyn_cast<ModuleOp>(*operation);
  // first walk the InstanceOp and store the corresponding FunctionLikeOp to each instance.
  // LHS is the SymbolRefAttr of the instance, RHS is the methods in the reference module of the instance.
  using MethodInInstance = std::pair<llvm::StringRef, llvm::SmallVector<llvm::StringRef, 4>>;

  /// a cache that stores all functions name which can be call on each instance.
  llvm::SmallVector<MethodInInstance> instanceAndMethod;

  auto instances = circt::gaa::getInstances(module);
  // walk the instance.
  for (auto instance : instances) {
    // get the reference module of an instance.
    auto instanceName = instance.instanceName();
    auto refModule = getReferenceModule(instance);
    // get all GAAFunctionLike Operator of the reference module.
    // ValueOp, MethodOp for GAAModule
    // BindValueOp, BindMethodOp for GAAExtModule
    auto instanceFunction = getFunctions(refModule);
    auto instanceFunctionNames = llvm::SmallVector<llvm::StringRef, 4>();
    llvm::for_each(instanceFunction, [&](GAAFunctionLike function){
      instanceFunctionNames.push_back(function->getName().getStringRef());
    });
    instanceAndMethod.push_back(std::pair(instanceName, instanceFunctionNames));
  }

  // walking Method/Value/RuleOp to show which function has been called.
  // TODO: validate when insertion.
  module.walk([&](Operation *op){
    mlir::TypeSwitch<Operation*>(op)
        // For each MethodOp and ValueOp in the module, check the
        .Case<MethodOp>([&](GAAFunctionLike function){
          function.walk([&](GAACallLike call){
            auto instanceToCall = call.instanceName();
            auto instanceToCallModule = module.lookupSymbol<InstanceOp>(instanceToCall).moduleName();
            auto functionToCall = call.functionName();
            methodsCache[function.functionName()].push_back(std::pair(instanceToCall, functionToCall));
          });
        })
        .Case<ValueOp>([&](GAAFunctionLike function){
          function.walk([&](GAACallLike call){
            auto instanceToCall = call.instanceName();
            auto instanceToCallModule = module.lookupSymbol<InstanceOp>(instanceToCall).moduleName();
            auto functionToCall = call.functionName();
            valuesCache[function.functionName()].push_back(std::pair(instanceToCall, functionToCall));
          });
        })
        .Case<RuleOp>([&](GAARuleLike rule){
          rule.walk([&](GAACallLike call){
            auto instanceToCall = call.instanceName();
            auto instanceToCallModule = module.lookupSymbol<InstanceOp>(instanceToCall).moduleName();
            auto functionToCall = call.functionName();
            rulesCache[rule.ruleName()].push_back(std::pair(instanceToCall, functionToCall));
          });
        });
  });
}
llvm::SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>, 4>
CallInfo::getAllCallee(llvm::StringRef symbolName) {
  llvm::SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>> all{};
  return all;
}
llvm::SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>, 4>
CallInfo::getAllMethodCallee(llvm::StringRef symbolName) {
  return methodsCache[symbolName];
}
llvm::SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>, 4>
CallInfo::getAllValueCallee(llvm::StringRef symbolName) {
  return valuesCache[symbolName];
}
llvm::SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>, 4>
CallInfo::getAllRuleCallee(llvm::StringRef symbolName) {
  return rulesCache[symbolName];
}
llvm::SmallVector<llvm::StringRef>
CallInfo::getAllCaller(llvm::StringRef instanceName,
                       llvm::StringRef functionName) {
  return llvm::SmallVector<llvm::StringRef>();
}
