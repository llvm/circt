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

FilterNode FilterNode::newGlob() {
  FilterNode f;
  f.tag = FilterType::GLOB;
  return f;
}

FilterNode FilterNode::newRecursiveGlob() {
  FilterNode f;
  f.tag = FilterType::RECURSIVE_GLOB;
  return f;
}

FilterNode FilterNode::newLiteral(std::string &literal) {
  FilterNode f;
  f.tag = FilterType::LITERAL;
  f.literal = literal;
  return f;
}

FilterNode FilterNode::newRegex(std::string &regex) {
  FilterNode f;
  f.tag = FilterType::REGEX;
  f.regex = regex;
  return f;
}

Filter Filter::newFilter(size_t count, FilterNode nodes[]) {
  Filter f;
  for (size_t i = 0; i < count; i++) {
    f.nodes.push_back(nodes[i]);
  }
  return f;
}

// TODO: Parsing errors.
Filter::Filter(std::string &filter) {
  if (filter.empty()) {
    return;
  }

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

std::vector<std::vector<mlir::Operation *>> filterAsVector(Filter &filter, ModuleOp &module) {
  std::vector<std::vector<mlir::Operation *>> filtered;
  std::vector<std::pair<std::vector<mlir::Operation *>, size_t>> opStack;

  if (filter.nodes.empty()) {
    for (auto op : module.getBody()->getOps<hw::HWModuleOp>()) {
      std::vector<Operation *> vec;
      vec.push_back(op);
      filtered.push_back(vec);
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
      case FilterType::RECURSIVE_GLOB: {
        std::vector<Operation *> vec;
        vec.push_back(op);
        match = true;
        opStack.push_back(std::make_pair(vec, 0));
        break;
      }
      case FilterType::LITERAL:
        match = node.literal == op.getNameAttr().getValue().str();
        break;
      case FilterType::REGEX:
        match = std::regex_match(op.getNameAttr().getValue().str(), node.regex);
        break;
    }

    if (match) {
      std::vector<Operation *> vec;
      vec.push_back(op);
      opStack.push_back(std::make_pair(vec, 1));
    }
  }

  while (!opStack.empty()) {
    std::pair<std::vector<Operation *>, size_t> pair = opStack[opStack.size() - 1];
    std::vector<Operation *> vec = pair.first;
    Operation *op = vec[vec.size() - 1];
    size_t i = pair.second;
    opStack.pop_back();

    if (i >= filter.nodes.size()) {
      filtered.push_back(vec);
    } else {
      TypeSwitch<Operation *>(op)
        .Case<hw::HWModuleOp>([&](auto &op) {
          for (auto &child : op.getBody().getOps()) {
            hw::InstanceOp instance;
            if ((instance = dyn_cast_or_null<hw::InstanceOp>(&child))) {
              auto copy = std::vector<Operation *>(vec);
              auto &node = filter.nodes[i];
              bool match = false;
              auto module = dyn_cast<hw::HWModuleOp>(instance.getReferencedModule());
              switch (node.tag) {
                case FilterType::UNSET:
                  break;
                case FilterType::GLOB:
                  match = true;
                  break;
                case FilterType::RECURSIVE_GLOB: {
                  auto copy2 = std::vector<Operation *>(vec);
                  copy2.push_back(module);
                  match = true;
                  opStack.push_back(std::make_pair(copy2, i));
                  break;
                }
                case FilterType::LITERAL:
                  match = node.literal == module.getNameAttr().getValue().str();
                  break;
                case FilterType::REGEX:
                  match = std::regex_match(module.getNameAttr().getValue().str(), node.regex);
                  break;
              }

              if (match) {
                copy.push_back(module);
                opStack.push_back(std::make_pair(copy, i + 1));
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
