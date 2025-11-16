//===- domaintool.cpp - Utility for processing domain information ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'domaintool', which is used for generating domain
// information from "final" MLIR compiled with `firtool` (and possibly other
// tools).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Version.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include "Handler.h"

using namespace circt;
using namespace llvm;
using namespace mlir;

namespace {
namespace options {

cl::OptionCategory cat{"domaintool options"};
cl::opt<std::string> inputFilename{cl::Positional, cl::desc("<input file>"),
                                   cl::init("-")};
cl::list<std::string> domains{
    "domain",
    cl::desc("a domain name to instantiate and arguments to instantiate it, "
             "e.g., '--domain ClockDomain,clockA,42'"),
    cl::Prefix, cl::cat(cat)};
cl::list<size_t> assign{
    "assign",
    cl::desc(
        "connect one of the previously declared domains to a port by its "
        "numeric id, e.g., to attach the second domain to the first port and "
        "the first domain to the second port use '--domain 1 --domain 0'"),
    cl::Prefix, cl::cat(cat)};
cl::opt<std::string> moduleName{
    "module", cl::Required,
    cl::desc("the module to process (not the class name)"), cl::cat(cat)};
cl::opt<bool> verifyDiagnostics{
    "verify-diagnostics",
    cl::desc("Check that emitted diagnostics match expected-* lines on the "
             "corresponding line"),
    cl::init(false), cl::Hidden, cl::cat(cat)};

} // namespace options

class Domain {

public:
  Domain(MLIRContext &context, StringRef str) {
    om::evaluator::BasePathValue emptyPath(&context);
    parameters.push_back(
        std::make_shared<om::evaluator::BasePathValue>(emptyPath));

    SmallVector<StringRef> parts;
    str.split(parts, ",");
    for (auto part : parts) {
      // The first part is the class name.
      if (!name) {
        name = StringAttr::get(&context, part);
        continue;
      }

      // All subsequent parts are parameters.  Test each of them to see what
      // type they are and add them to the args.
      int64_t intParam;
      if (!part.getAsInteger(10, intParam)) {
        parameters.push_back(
            om::evaluator::AttributeValue::get(om::IntegerAttr::get(
                &context, mlir::IntegerAttr::get(IntegerType::get(&context, 64),
                                                 intParam))));
        continue;
      }

      parameters.push_back(om::evaluator::AttributeValue::get(
          StringAttr::get(part, om::StringType::get(&context))));
    }
  }

  FailureOr<om::evaluator::EvaluatorValuePtr>
  instantiate(om::Evaluator &evaluator) {
    return evaluator.instantiate(name, parameters);
  }

private:
  StringAttr name;

  SmallVector<om::evaluator::EvaluatorValuePtr> parameters;
};

class DomainTool {

public:
  DomainTool(MLIRContext &context) : context(context) {}

  LogicalResult execute();

private:
  LogicalResult executeWithSource();

  MLIRContext &context;

  SourceMgr sourceMgr;

  SmallVector<Domain> domains;
};

LogicalResult DomainTool::executeWithSource() {

  auto moduleOp = parseSourceFile<ModuleOp>(sourceMgr, &context);

  auto evaluator = om::Evaluator(moduleOp.get());

  // Instantiate all the command-line provided domains and put these in
  // `domainObjects`.
  SmallVector<om::evaluator::EvaluatorValuePtr> domainObjects;
  for (auto &domain : domains) {
    auto maybeDomain = domain.instantiate(evaluator);
    if (failed(maybeDomain))
      return failure();
    domainObjects.push_back(*maybeDomain);
  }

  // Put the parameters necessary to instantiate the class in `parameters`.
  // This consists of an empty base path and all the domains whose order is
  // specified by the command line `-asign` options.
  SmallVector<om::evaluator::EvaluatorValuePtr> parameters;
  om::evaluator::BasePathValue emptyPath(&context);
  parameters.push_back(
      std::make_shared<om::evaluator::BasePathValue>(emptyPath));
  for (auto domainIndex : options::assign) {
    if (domainIndex >= domainObjects.size()) {
      llvm::errs()
          << "unable to assign domain '" << domainIndex
          << "' because it is larger than the number of domains provided, '"
          << parameters.size() << "'";
      return failure();
    }
    parameters.push_back(domainObjects[domainIndex]);
  }

  // The class is the command-line module name with `_Class` appeneded.
  // Instantiate it with the provided parameters.
  //
  // TODO: This is brittle and relies on the lowering of FIRRTL classes to
  // objects.  Is there a better "ABI" here?
  auto className =
      StringAttr::get(&context, Twine(options::moduleName) + "_Class");
  auto evaluatorValue = evaluator.instantiate(className, parameters);
  if (failed(evaluatorValue))
    return failure();

  // Read the ouptut domain ports off the object.  This has a very specific
  // format where we are expecting to find output objects that have the
  // following fields:
  //
  //   - "domainInfo_out": a domain object
  //   - "associations_out": all the ports associated with the domain object
  //
  // The domain object we get back may be a parameter that we passed in or it
  // may have been created internal to the circuit.
  //
  // The end result of this loop is that the `byType` map is created.  This
  // organizes the resulting objects into a map of:
  //
  //     DomainKind -> Domain -> Associations
  //
  // The domain kind is "clock", "reset", or "power".  The domain is the actual
  // domain object with all its fields populate.  The associations are the ports
  // associated with that domain.
  llvm::MapVector<Type, ObjectMap> byType;
  auto *object = cast<om::evaluator::ObjectValue>(evaluatorValue->get());
  for (auto &[name, value] : object->getFields()) {
    auto *object = dyn_cast<om::evaluator::ObjectValue>(value.get());

    // Get "domainInfo_out".
    auto domainInfoValue = object->getField("domainInfo_out");
    if (failed(domainInfoValue)) {
      llvm::errs() << "output object did not contain an 'domainInfo_out' field";
      return failure();
    }
    auto *domainInfoObject =
        dyn_cast<om::evaluator::ObjectValue>(domainInfoValue->get());
    if (!domainInfoObject) {
      llvm::errs() << "unexpected type of 'domainInfo_out'. Must be an object.";
      return failure();
    }

    // Get "associations_out".
    auto associationsValue = object->getField("associations_out");
    if (failed(associationsValue)) {
      llvm::errs()
          << "output object did not contain an 'associations_out' field";
      return failure();
    }
    auto *associationsList =
        dyn_cast<om::evaluator::ListValue>(associationsValue->get());
    if (!associationsList) {
      llvm::errs() << "unexpected type of 'associations_out'. Must be a list.";
      return failure();
    }

    // Update the `byType` map.
    //
    // TODO: Fix the instability here.  The insertion order of
    // `domainInfoObject` isn't being properly preserved.
    byType[domainInfoObject->getType()][domainInfoObject].append(
        associationsList->getElements());
  }

  // Accumulate domain information into registered handlers.  Pass each domain
  // kind to each handler that can handle it.
  for (auto &[type, objectMap] : byType)
    for (auto &handler : HandlerRegistry::get().getHandlers())
      if (handler->shouldHandle(type))
        if (failed(handler->handle(objectMap)))
          return failure();

  // The domains handlers have seen all the domains they care about.  Now have
  // them take some action (likely generating files or writing to stdout).
  for (auto &handler : HandlerRegistry::get().getHandlers())
    if (failed(handler->emit()))
      return failure();

  return success();
}

LogicalResult DomainTool::execute() {
  // Finish parsing options into objects.
  for (auto const &domainStr : options::domains) {
    domains.push_back(Domain(context, domainStr));
  }

  std::string errorMessage;
  auto buffer = openInputFile(options::inputFilename, &errorMessage);
  if (!buffer) {
    WithColor::error() << errorMessage << "\n";
    return failure();
  }

  sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());

  // Call `executeWithSources` with either the regular diagnostic handler, or,
  // if `--verify-diagnostics` is set, with the verifying handler.
  if (options::verifyDiagnostics) {
    SourceMgrDiagnosticVerifierHandler handler(sourceMgr, &context);
    context.printOpOnDiagnostic(false);
    (void)executeWithSource();
    return handler.verify();
  }
  SourceMgrDiagnosticHandler handler(sourceMgr, &context);
  return executeWithSource();
}

} // namespace

int main(int argc, char **argv) {
  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Add CIRCT version information.
  cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });

  cl::ParseCommandLineOptions(argc, argv,
                              "Utility for generating constraint files/formats "
                              "from MLIR containing domain information\n");

  DialectRegistry registry;
  registry.insert<comb::CombDialect, debug::DebugDialect, hw::HWDialect,
                  om::OMDialect, seq::SeqDialect, sv::SVDialect,
                  verif::VerifDialect>();
  MLIRContext context(registry);
  context.loadDialect<comb::CombDialect, debug::DebugDialect, hw::HWDialect,
                      om::OMDialect, seq::SeqDialect, sv::SVDialect,
                      verif::VerifDialect>();

  exit(failed(DomainTool(context).execute()));
}
