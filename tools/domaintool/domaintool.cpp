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

#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "circt/InitAllDialects.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Version.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include "Handler.h"

using namespace circt;
using namespace llvm;
using namespace mlir;

namespace {
namespace options {
cl::opt<std::string> outputFilename{
    "o",
    cl::desc("Output filename, or directory for split output"),
    cl::value_desc("filename"),
    cl::init("-"),
};

cl::OptionCategory cat{"domaintool options"};
cl::opt<std::string> inputFilename{cl::Positional, cl::desc("<input file>"),
                                   cl::init("-")};
cl::list<std::string> domains{
    "domain",
    cl::desc("a domain name to instantiate and arguments to instantiate it, "
             "e.g., '--domain ClockDomain,clockA,42'"),
    cl::Prefix, cl::cat(cat)};
cl::list<std::string> assign{
    "assign",
    cl::desc(
        "connect one of the previously declared domains to a port by its "
        "numeric id, e.g., to attach the second domain to the first domain and "
        "the first domain to the second domain use '--assign 1 --assign 0'.  "
        "Any unassigned parameters will automatically be set to unknown."),
    cl::Prefix, cl::cat(cat)};
cl::opt<std::string> moduleName{
    "module", cl::Required,
    cl::desc("the module to process (not the class name)"), cl::cat(cat)};
cl::opt<bool> verifyDiagnostics{
    "verify-diagnostics",
    cl::desc("Check that emitted diagnostics match expected-* lines on the "
             "corresponding line"),
    cl::init(false), cl::Hidden, cl::cat(cat)};
cl::opt<bool> splitInputFile("split-input-file",
                             cl::desc("Split the input file into pieces and "
                                      "process each chunk independently"),
                             cl::init(false), cl::Hidden, cl::cat(cat));
} // namespace options

/// A representation of a command-line provided domain, parsed into a name and
/// arguments.  This can then be used to instantiate the actual domain, assuming
/// no problems are found.
///
/// The format for a domain is very simple:
///
///     parameter := integer | string
///     domain := name { `,` parameter }
///
/// By example, the following would create a `ClockDomain` domain with
/// parameters "Foo" and 42:
///
///     ClockDomain,Foo,42
///
/// TODO: Improve this format to be something less brittle.
class Domain {

public:
  /// Construct a domain and parse its arguments into internal datastructures.
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

  /// Instantiate a domain based on the arguments provided or fail.
  FailureOr<om::evaluator::EvaluatorValuePtr>
  instantiate(om::Evaluator &evaluator) {
    return evaluator.instantiate(name, parameters);
  }

private:
  /// The name of the domain.
  StringAttr name;

  /// Parameters used to construct the domain.
  SmallVector<om::evaluator::EvaluatorValuePtr> parameters;
};

class DomainTool {

public:
  DomainTool(MLIRContext &context) : context(context) {}

  /// Programmatic (not command line) entry point for running domaintool.
  LogicalResult execute();

private:
  /// This is the main work function.  The buffer in the input `SourceMgr` will
  /// have domain information extracted and processed.
  LogicalResult processSourceMgr(llvm::SourceMgr &sourceMgr);

  /// Process a single buffer containg one input file.  This buffer has already
  /// been split if it is going to be.
  LogicalResult processBufferSplit(std::unique_ptr<llvm::MemoryBuffer> buffer);

  /// Process a single buffer containg a complete input file.  This will
  /// internally create `SourceMgr`s and farm out calls to `processSourceMgr`.
  LogicalResult processBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer);

  /// An MLIR context used when creating operations.
  MLIRContext &context;

  /// The command-line provided domains.
  SmallVector<Domain> domains;

  /// An output file (or directory) where outputs will be written.
  std::optional<std::unique_ptr<ToolOutputFile>> outputFile;
};

LogicalResult DomainTool::processSourceMgr(llvm::SourceMgr &sourceMgr) {
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

  // The class is the command-line module name with `_Class` appeneded.
  //
  // TODO: This is brittle and relies on the lowering of FIRRTL classes to
  // objects.  Is there a better "ABI" here?
  auto className =
      StringAttr::get(&context, Twine(options::moduleName) + "_Class");

  // Look up the class to get formal parameter types for creating unknown values
  auto classOp = moduleOp->lookupSymbol<om::ClassLike>(className);
  if (!classOp) {
    llvm::errs() << "unable to find class '" << className.getValue() << "'\n";
    return failure();
  }

  // Parse the -assign options to build a list of domain indices for each
  // domain type.
  llvm::MapVector<om::ClassType, SmallVector<size_t>> assignmentsByType;
  for (auto &value : options::assign) {
    size_t domainIndex;
    if (!to_integer(value, domainIndex)) {
      llvm::errs() << "illegal assignment value '" << value
                   << "', must be a number\n";
      return failure();
    }

    if (domainIndex >= domainObjects.size()) {
      llvm::errs()
          << "unable to assign domain '" << domainIndex
          << "' because it is larger than the number of domains provided, '"
          << domainObjects.size() << "'\n";
      return failure();
    }

    auto *objValue =
        cast<om::evaluator::ObjectValue>(domainObjects[domainIndex].get());
    om::ClassType domainType = objValue->getObjectType();
    assignmentsByType[domainType].push_back(domainIndex);
  }

  // Build a map from domain type to the number of parameters of that type
  // that we need to assign. This is used to validate that the user has
  // provided enough domain assignments.
  llvm::MapVector<om::ClassType, size_t> paramCountByType;
  ArrayRef<BlockArgument> formalParams = classOp.getBodyBlock()->getArguments();
  for (size_t i = 1; i < formalParams.size(); ++i)
    if (auto classType = dyn_cast<om::ClassType>(formalParams[i].getType()))
      ++paramCountByType[classType];

  // Validate that for each domain type, the number of assignments matches
  // the number of parameters of that type.
  for (auto &[domainType, numParams] : paramCountByType) {
    size_t numAssignments = 0;
    auto *it = assignmentsByType.find(domainType);
    if (it != assignmentsByType.end())
      numAssignments = it->second.size();

    if (numAssignments != numParams) {
      llvm::errs() << "error: class has " << numParams
                   << " parameter(s) of type '"
                   << domainType.getClassName().getValue() << "' but assigned "
                   << numAssignments << "\n";
      return failure();
    }
  }

  // Build the parameter list by iterating through formal parameters and
  // assigning domains by type or unknown values.
  SmallVector<om::evaluator::EvaluatorValuePtr> parameters;
  om::evaluator::BasePathValue emptyPath(&context);
  parameters.push_back(
      std::make_shared<om::evaluator::BasePathValue>(emptyPath));

  // Track the next assignment index for each domain type
  auto unknownLoc = UnknownLoc::get(&context);
  llvm::DenseMap<om::ClassType, size_t> nextAssignmentIndex;
  for (size_t i = 1; i < formalParams.size(); ++i) {
    Type paramType = formalParams[i].getType();

    // Check if this parameter is a domain type (ClassType)
    if (auto classType = dyn_cast<om::ClassType>(paramType)) {
      // Check if we have assignments for this domain type
      auto *it = assignmentsByType.find(classType);
      if (it != assignmentsByType.end()) {
        // Get the next assignment for this domain type.
        size_t &nextIdx = nextAssignmentIndex[classType];
        size_t domainIndex = it->second[nextIdx++];
        parameters.push_back(domainObjects[domainIndex]);
        continue;
      }
    }

    // For non-domain types or unassigned domain types, create an unknown value
    auto unknownValue =
        om::evaluator::AttributeValue::get(paramType, LocationAttr(unknownLoc));
    unknownValue->markUnknown();
    parameters.push_back(unknownValue);
  }

  // Instantiate the class with the provided parameters.
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
  //     Domain Kind -> Domain -> Associations
  //
  // The domain kind is "clock", "reset", or "power".  The domain is the actual
  // domain object with all its fields populated.  The associations are the
  // ports associated with that domain.
  //
  // Note: Care needs to be taken here to ensure the stability of the output.
  // This means that the iteration over the fields must be stable.  (Using
  // `getFields` is unstable, while `getFieldNames` is stable.)  Additionally,
  // everything needs to to be inserted into `byType` and its underlying
  // `ObjectMap`s stably.
  llvm::MapVector<Type, ObjectMap> byType;
  auto *object = cast<om::evaluator::ObjectValue>(evaluatorValue->get());
  for (auto fieldNameAttr : object->getFieldNames().getAsRange<StringAttr>()) {
    // Skip anything which is unknown.
    if (object->getField(fieldNameAttr)->get()->isUnknown())
      continue;

    // Skip anything which is not an object.
    auto *domain = dyn_cast<om::evaluator::ObjectValue>(
        object->getField(fieldNameAttr)->get());
    if (!domain)
      continue;

    // TODO: Improve the structural typing here in favor of something stricter,
    // like an actual ABI that relies on dedicated class names to figure out
    // what an output domain is.

    bool hasDomainInfo = false, hasAssociations = false;
    for (auto attr : domain->getFieldNames()) {
      auto name = cast<StringAttr>(attr);
      hasDomainInfo |= name == "domainInfo_out";
      hasAssociations |= name == "associations_out";
    }
    if (!hasDomainInfo || !hasAssociations)
      continue;

    // Get "domainInfo_out".
    auto domainInfoValue = domain->getField("domainInfo_out");
    auto *domainInfoObject =
        dyn_cast<om::evaluator::ObjectValue>(domainInfoValue->get());
    if (!domainInfoObject)
      continue;

    // Get "associations_out".
    auto associationsValue = domain->getField("associations_out");
    auto *associationsList =
        dyn_cast<om::evaluator::ListValue>(associationsValue->get());
    if (!associationsList)
      continue;

    // Update the `byType` map.
    byType[domainInfoObject->getType()][domainInfoObject].append(
        associationsList->getElements());
  }

  // Accumulate domain information into registered handlers.  Pass each object
  // map to each handler that claims they can support that domain kind.
  for (auto &[type, objectMap] : byType)
    for (auto &handler : HandlerRegistry::get().getHandlers())
      if (handler->shouldHandle(type))
        if (failed(handler->handle(objectMap)))
          return failure();

  // Now that the domain handlers have seen all the domains they care about,
  // tell them to take some action (likely generating files or writing to
  // stdout).  Reset the handler after emission.
  for (auto &handler : HandlerRegistry::get().getHandlers()) {

    if (failed(handler->emit((*outputFile)->os())))
      return failure();
    handler->clear();
  }

  return success();
}

LogicalResult
DomainTool::processBufferSplit(std::unique_ptr<llvm::MemoryBuffer> buffer) {
  // Ensure null termination.
  if (!buffer->getBuffer().ends_with('\0')) {
    buffer = llvm::MemoryBuffer::getMemBufferCopy(
        buffer->getBuffer(), buffer->getBufferIdentifier());
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());

  // Call `executeWithSources` with either the regular diagnostic handler, or,
  // if `--verify-diagnostics` is set, with the verifying handler.
  if (options::verifyDiagnostics) {
    SourceMgrDiagnosticVerifierHandler handler(sourceMgr, &context);
    context.printOpOnDiagnostic(false);
    (void)processSourceMgr(sourceMgr);
    return handler.verify();
  }
  SourceMgrDiagnosticHandler handler(sourceMgr, &context);
  return processSourceMgr(sourceMgr);
}

LogicalResult
DomainTool::processBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer) {
  // If _not_ `-split-input-file`, then just pass the single buffer to
  if (!options::splitInputFile)
    return processBufferSplit(std::move(buffer));

  return splitAndProcessBuffer(
      std::move(buffer),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processBufferSplit(std::move(buffer));
      },
      llvm::outs());
}

LogicalResult DomainTool::execute() {
  // Finish parsing options into objects.
  for (auto const &domainStr : options::domains)
    domains.push_back(Domain(context, domainStr));

  std::string errorMessage;
  auto buffer = openInputFile(options::inputFilename, &errorMessage);
  if (!buffer) {
    WithColor::error() << errorMessage << "\n";
    return failure();
  }

  // TODO: Implement multi-file output.
  outputFile.emplace(openOutputFile(options::outputFilename, &errorMessage));
  if (!(*outputFile)) {
    WithColor::error() << errorMessage << "\n";
    return failure();
  }

  if (failed(processBuffer(std::move(buffer))))
    return failure();

  if (outputFile.has_value())
    (*outputFile)->keep();

  return success();
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
  registerAllDialects(registry);
  MLIRContext context(registry);
  // Load the OM dialect explicitly since we use OM types/attributes before
  // parsing any MLIR.
  context.loadDialect<om::OMDialect>();

  exit(failed(DomainTool(context).execute()));
}
