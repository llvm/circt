#include <string>

#include "circt/Query/Query.h"

namespace circt {
namespace query {

FilterNode::FilterNode(const FilterNode &other) {
  tag = other.tag;
  switch (tag) {
    case FilterType::GLOB:
    case FilterType::RECURSIVE_GLOB:
    case FilterType::UNSET:
      break;

    case FilterType::REGEX:
      regex = other.regex;
      break;

    case FilterType::LITERAL:
      literal = other.literal;
      break;
  }
}

// TODO: Parsing errors.
Filter::Filter(std::string &filter) {
  FilterNode n;
  bool stop = false;
  size_t start = 0;

  for (size_t i = 0; i < filter.size(); i++) {
    char c = filter[i];
    switch (n.tag) {
      case FilterType::UNSET:
        if (c == '*') {
          n.tag = FilterType::GLOB;
        } else if (c == '/') {
          n.tag = FilterType::REGEX;
        } else if (c != ':') {
          n.tag = FilterType::LITERAL;
        } else {
          stop = true;
        }
        break;

      case FilterType::GLOB:
        if (c == '*') {
          n.tag = FilterType::RECURSIVE_GLOB;
        } else {
          stop = true;
        }
        break;

      case FilterType::RECURSIVE_GLOB:
        stop = true;
        break;

      case FilterType::REGEX:
        if (c == '\\') {
          ++i;
        } else if (i - 1 != start && filter[i - 1] == '/'){
          stop = true;
        }
        break;

      case FilterType::LITERAL:
        if (c == ':') {
          stop = true;
        }
        break;
    }

    if (stop) {
      switch (n.tag) {
        case FilterType::LITERAL:
          n.literal = filter.substr(start, i - start);
          break;

        case FilterType::REGEX: {
          auto s = filter.substr(start, i - start);
          n.regex = std::regex(s);
          break;
        }

        case FilterType::GLOB:
        case FilterType::RECURSIVE_GLOB:
        case FilterType::UNSET:
          break;
      }

      if (c == ':' && filter[i + 1] == ':') {
        i++;
      } else {
        break;
      }

      start = i + 1;
      stop = false;
      nodes.push_back(n);
      n.tag = FilterType::UNSET;
    }
  }
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
      TypeSwitch<Operation *>(op)
        .Case<hw::HWModuleOp>([&](hw::HWModuleOp &op) {
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
                  opStack.push_back(std::make_pair(instance.getReferencedModule(), i));
                  break;
                case FilterType::LITERAL:
                  match = node.literal == instance.getNameAttr().getValue().str();
                  break;
                case FilterType::REGEX:
                  match = std::regex_match(instance.getNameAttr().getValue().str(), node.regex);
                  break;
              }

              if (match) {
                  opStack.push_back(std::make_pair(instance.getReferencedModule(), i + 1));
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
