#include <string>

#include "circt/Query/Query.h"

namespace circt {
namespace query {

Filter parseFilter(std::string &filter) {
  Filter f;
  /*
  for (auto c : filter) {
    // TODO
  }
  */
  return f;
}

std::vector<mlir::Operation *> filterAsVector(Filter &filter, Operation *module) {
  std::vector<mlir::Operation *> filtered;
  std::vector<std::pair<mlir::Operation *, size_t>> opStack;
  opStack.push_back(std::make_pair(module, 0));

  while (opStack.size() != 0) {
    std::pair<Operation *, size_t> pair = opStack[opStack.size() - 1];
    Operation *op = pair.first;
    size_t i = pair.second;

    opStack.pop_back();

    if (i >= filter.nodes.size()) {
      filtered.push_back(op);
    } else {
      TypeSwitch<Operation *>(&*op)
        .Case<hw::HWModuleOp *>([&](hw::HWModuleOp &op) {
          for (auto &child : op.getBody().getOps()) {
            hw::InstanceOp instance;
            if ((instance = dyn_cast_or_null<hw::InstanceOp>(&child))) {
              auto &node = filter.nodes[i];
              bool match = false;
              switch (node.tag) {
                case FilterType::UNSET:
                  break;
                case FilterType::GLOB:
                  match = true;
                  break;
                case FilterType::RECURSIVE_GLOB:
                  match = true;
                  opStack.push_back(std::make_pair(instance, i));
                  break;
                case FilterType::LITERAL:
                  match = node.literal == instance.getNameAttr().getValue().str();
                  break;
                case FilterType::REGEX:
                  match = std::regex_match(instance.getNameAttr().getValue().str(), node.regex);
                  break;
              }

              if (match) {
                  opStack.push_back(std::make_pair(instance, i + 1));
              }
            }
          }
        });
    }
  }

  return filtered;
}

FilterNode::~FilterNode() {
  switch (this->tag) {
    case FilterType::UNSET:
    case FilterType::GLOB:
    case FilterType::RECURSIVE_GLOB:
      break;
    case FilterType::REGEX: {
      std::regex r = regex;
      break;
    }
    case FilterType::LITERAL: {
      std::string l = literal;
      break;
    }
  }
}

} /* namespace query */
} /* namespace circt */
