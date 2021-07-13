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

FilterNode &FilterNode::operator =(const FilterNode &other) {
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

  return *this;
}

// TODO: Parsing errors.
Filter::Filter(std::string &filter) {
  FilterNode n;
  bool stop = false;
  size_t start = 0;

  for (size_t i = 0; i <= filter.size(); i++) {
    if (i < filter.size()) {
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
          } else if (i - 1 != start && filter[i - 1] == '/' && (i - 2 != start || filter[i - 2] != '\\')) {
            stop = true;
          }
          break;

        case FilterType::LITERAL:
          if (c == ':') {
            stop = true;
          }
          break;
      }
    } else {
      stop = true;
    }

    if (stop) {
      switch (n.tag) {
        case FilterType::LITERAL:
          n.literal = filter.substr(start, i - start);
          break;

        case FilterType::REGEX: {
          auto s = filter.substr(start + 1, i - start - 2);
          n.regex = std::regex(s);
          break;
        }

        case FilterType::GLOB:
        case FilterType::RECURSIVE_GLOB:
        case FilterType::UNSET:
          break;
      }

      stop = false;
      nodes.push_back(n);
      FilterNode n2;
      n = n2;

      if (i + 1 < filter.size() && filter[i] == ':' && filter[i + 1] == ':') {
        ++i;
        start = i + 1;
      }
    }
  }
}

std::vector<mlir::Operation *> filterAsVector(Filter &filter, ModuleOp &module) {
  std::vector<mlir::Operation *> filtered;
  std::vector<std::pair<mlir::Operation *, size_t>> opStack;

  if (filter.nodes.empty()) {
    for (auto op : module.getBody()->getOps<hw::HWModuleOp>()) {
      filtered.push_back(op);
    }
    return filtered;
  }

  FilterNode &node = filter.nodes[0];
  for (auto op : module.getBody()->getOps<hw::HWModuleOp>()) {
    bool match = false;
    switch (node.tag) {
      case FilterType::UNSET:
        break;
      case FilterType::GLOB:
        match = true;
        break;
      case FilterType::RECURSIVE_GLOB:
        match = true;
        opStack.push_back(std::make_pair(op, 0));
        break;
      case FilterType::LITERAL:
        match = node.literal == op.getNameAttr().getValue().str();
        break;
      case FilterType::REGEX:
        match = std::regex_match(op.getNameAttr().getValue().str(), node.regex);
        break;
    }

    if (match) {
      opStack.push_back(std::make_pair(op, 1));
    }
  }

  while (!opStack.empty()) {
    std::pair<Operation *, size_t> pair = opStack[opStack.size() - 1];
    Operation *op = pair.first;
    size_t i = pair.second;
    opStack.pop_back();

    if (i >= filter.nodes.size()) {
      filtered.push_back(op);
    } else {
      TypeSwitch<Operation *>(op)
        .Case<hw::HWModuleOp>([&](auto &op) {
          for (auto &child : op.getBody().getOps()) {
            hw::InstanceOp instance;
            if ((instance = dyn_cast_or_null<hw::InstanceOp>(&child))) {
              auto &node = filter.nodes[i];
              bool match = false;
              auto module = dyn_cast<hw::HWModuleOp>(instance.getReferencedModule());
              switch (node.tag) {
                case FilterType::UNSET:
                  break;
                case FilterType::GLOB:
                  match = true;
                  break;
                case FilterType::RECURSIVE_GLOB:
                  match = true;
                  opStack.push_back(std::make_pair(module, i));
                  break;
                case FilterType::LITERAL:
                  match = node.literal == module.getNameAttr().getValue().str();
                  break;
                case FilterType::REGEX:
                  match = std::regex_match(module.getNameAttr().getValue().str(), node.regex);
                  break;
              }

              if (match) {
                opStack.push_back(std::make_pair(module, i + 1));
              }
            }
          }
        });
    }
  }

  return filtered;
}

} /* namespace query */
} /* namespace circt */
