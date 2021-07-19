#include "circt/Query/Query.h"

namespace circt {
namespace query {

bool operator &(ValueTypeType a, ValueTypeType b) {
  return ((size_t) a) & ((size_t) b);
}

ValueTypeType operator |(ValueTypeType a, ValueTypeType b) {
  return (ValueTypeType) (((size_t) a) | ((size_t) b));
}

bool operator &(PortType a, PortType b) {
  return ((size_t) a) & ((size_t) b);
}

PortType operator |(PortType a, PortType b) {
  return (PortType) (((size_t) a) | ((size_t) b));
}

FilterNode::FilterNode(const FilterNode &other) {
  tag = other.tag;
  type = other.type;
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
  type = other.type;
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

FilterNode FilterNode::newGlob(ValueType type) {
  FilterNode f;
  f.tag = FilterType::GLOB;
  f.type = type;
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

FilterNode FilterNode::newLiteral(std::string &literal, ValueType type) {
  FilterNode f;
  f.tag = FilterType::LITERAL;
  f.type = type;
  f.literal = literal;
  return f;
}

FilterNode FilterNode::newRegex(std::string &regex) {
  FilterNode f;
  f.tag = FilterType::REGEX;
  f.regex = regex;
  return f;
}

FilterNode FilterNode::newRegex(std::string &regex, ValueType type) {
  FilterNode f;
  f.tag = FilterType::REGEX;
  f.type = type;
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
          } else if (i - 1 != start && filter[i - 1] == '/' && (i - 2 == start || filter[i - 2] != '\\')) {
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

      if (i + 1 < filter.size() && filter[i] == ':' && filter[i + 1] != ':') {
        ++i;
        start = i;
        bool setType = false;
        ValueTypeType type;
        bool setPort = false;
        PortType port;
        auto widths = std::vector<Range>();
        widths.push_back(Range(0, 1000));
        for (; i <= filter.size(); i++) {
          if (i < filter.size()) {
            char c = filter[i];
            switch (c) {
              // Input
              case 'i':
                if (setPort) {
                  port = port | PortType::INPUT;
                } else {
                  port = PortType::INPUT;
                  setPort = true;
                }
                break;

              // Output
              case 'o':
                if (setPort) {
                  port = port | PortType::OUTPUT;
                } else {
                  port = PortType::OUTPUT;
                  setPort = true;
                }
                break;

              // Not input/output
              case 'n':
                if (setPort) {
                  port = port | PortType::NONE;
                } else {
                  port = PortType::NONE;
                  setPort = true;
                }
                break;

              // Module
              case 'm':
                if (setType) {
                  type = type | ValueTypeType::MODULE;
                } else {
                  type = ValueTypeType::MODULE;
                  setType = true;
                }
                break;

              // Wire
              case 'w':
                if (setType) {
                  type = type | ValueTypeType::WIRE;
                } else {
                  type = ValueTypeType::WIRE;
                  setType = true;
                }
                break;

              // Register
              case 'r':
                if (setType) {
                  type = type | ValueTypeType::REGISTER;
                } else {
                  type = ValueTypeType::REGISTER;
                  setType = true;
                }
                break;

              default:
                break;
            }
          } else {
            break;
          }
        }

        if (!setType) {
          type = ValueTypeType::MODULE;
        }
        if (!setPort) {
          port = PortType::NONE;
        }

        n.type = ValueType(type, port, widths);
      }

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

void matchAndAppend(FilterNode &node, std::vector<Operation *> vec, hw::HWModuleOp &module, std::vector<std::pair<std::vector<Operation *>, size_t>> &opStack, size_t i, std::string &name, bool &match, bool appendSelf) {
  auto copy = std::vector<Operation *>(vec);
  switch (node.tag) {
    case FilterType::UNSET:
      break;
    case FilterType::GLOB:
      match = true;
      break;
    case FilterType::RECURSIVE_GLOB: {
      auto copy2 = std::vector<Operation *>(vec);
      if (appendSelf) {
        copy2.push_back(module);
      }

      match = true;
      opStack.push_back(std::make_pair(copy2, i));
      break;
    }
    case FilterType::LITERAL:
      match = node.literal == name;
      break;
    case FilterType::REGEX:
      match = std::regex_match(name, node.regex);
      break;
  }

  if (match) {
    if (appendSelf) {
      copy.push_back(module);
    }

    opStack.push_back(std::make_pair(copy, i + 1));
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
      bool contained = false;
      for (auto v : filtered) {
        if (v == vec) {
          contained = true;
          break;
        }
      }

      if (!contained) {
        filtered.push_back(vec);
      }
    } else {
      TypeSwitch<Operation *>(op)
        .Case<hw::HWModuleOp>([&](hw::HWModuleOp &op) {
          auto &node = filter.nodes[i];
          auto type = node.type;
          bool match = false;
          if (type.getType() & ValueTypeType::MODULE) {
            for (auto &child : op.getBody().getOps()) {
              hw::InstanceOp instance;
              if ((instance = dyn_cast_or_null<hw::InstanceOp>(&child))) {
                auto module = dyn_cast<hw::HWModuleOp>(instance.getReferencedModule());
                auto name = module.getNameAttr().getValue().str();
                matchAndAppend(node, vec, module, opStack, i, name, match, true);

                if (match) {
                  continue;
                }
              }
            }
          }

          if (!match && (type.getType() & ValueTypeType::WIRE)) {
            if (type.getPort() & PortType::INPUT) {
              for (auto &port : op.getPorts()) {
                if (port.isOutput()) {
                  continue;
                }

                auto name = port.getName().str();
                matchAndAppend(node, vec, op, opStack, i, name, match, false);
                if (match) {
                  return;
                }
              }
            }

            if (type.getPort() & PortType::OUTPUT) {
              for (auto &port : op.getPorts()) {
                if (!port.isOutput()) {
                  continue;
                }

                auto name = port.getName().str();
                matchAndAppend(node, vec, op, opStack, i, name, match, false);
                if (match) {
                  return;
                }
              }
            }

            sv::WireOp wire;
            for (auto &child : op.getBody().getOps()) {
              if ((wire = dyn_cast_or_null<sv::WireOp>(&child))) {
                if (type.getPort() & PortType::NONE) {
                  // TODO
                }
              }
            }
          }

          if (!match && (node.type.getType() & ValueTypeType::REGISTER)) {
            // TODO
          }
        });

      if (filter.nodes[i].tag == FilterType::RECURSIVE_GLOB) {
        opStack.push_back(std::make_pair(vec, i + 1));
      }
    }
  }

  return filtered;
}

} /* namespace query */
} /* namespace circt */
