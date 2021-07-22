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

void matchAndAppend(FilterNode &node, Operation *module, std::vector<std::pair<Operation *, size_t>> &opStack, size_t i, std::string &name, bool &match) {
  switch (node.tag) {
    case FilterType::UNSET:
      break;
    case FilterType::GLOB:
      match = true;
      break;
    case FilterType::RECURSIVE_GLOB: {
      match = true;
      opStack.push_back(std::make_pair(module, i));
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
    opStack.push_back(std::make_pair(module, i + 1));
  }
}

std::vector<Operation *> filterAsVector(Filter &filter, Operation *root) {
  std::vector<mlir::Operation *> filtered;
  std::vector<std::pair<mlir::Operation *, size_t>> opStack;

  if (filter.nodes.empty()) {
    filtered.push_back(root);
    return filtered;
  }

  opStack.push_back(std::make_pair(root, 0));
  while (!opStack.empty()) {
    auto pair = opStack[opStack.size() - 1];
    Operation *op = pair.first;
    size_t i = pair.second;
    opStack.pop_back();

    if (i >= filter.nodes.size()) {
      bool contained = false;
      for (auto *o : filtered) {
        if (o == op) {
          contained = true;
          break;
        }
      }

      if (!contained) {
        filtered.push_back(op);
      }
    } else {
      TypeSwitch<Operation *>(op)
        .Case<ModuleOp>([&](ModuleOp &op) {
          auto &node = filter.nodes[i];
          auto type = node.type;
          bool match = false;
          if (type.getType() & ValueTypeType::MODULE) {
            for (auto child : op.getBody()->getOps<hw::HWModuleOp>()) {
              auto name = child.getNameAttr().getValue().str();
              matchAndAppend(node, child, opStack, i, name, match);
            }

            for (auto child : op.getBody()->getOps<hw::HWModuleExternOp>()) {
              auto name = child.getNameAttr().getValue().str();
              matchAndAppend(node, child, opStack, i, name, match);
            }
          }
        })
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
                matchAndAppend(node, module, opStack, i, name, match);

                if (match) {
                  continue;
                }
              }
            }
          }

          if (!match && (type.getType() & ValueTypeType::WIRE)) {
            if (type.getPort() & PortType::INPUT) {
              for (auto &port : op.getPorts()) {
                if (port.isOutput() || !node.type.containsWidth(port.type.getIntOrFloatBitWidth())) {
                  continue;
                }

                auto name = port.getName().str();
                matchAndAppend(node, op, opStack, i, name, match);
                if (match) {
                  return;
                }
              }
            }

            if (type.getPort() & PortType::OUTPUT) {
              for (auto &port : op.getPorts()) {
                if (!port.isOutput() || !node.type.containsWidth(port.type.getIntOrFloatBitWidth())) {
                  continue;
                }

                auto name = port.getName().str();
                matchAndAppend(node, op, opStack, i, name, match);
                if (match) {
                  return;
                }
              }
            }

            if (type.getPort() & PortType::NONE) {
              sv::WireOp wire;
              for (auto &child : op.getBody().getOps()) {
                if ((wire = dyn_cast_or_null<sv::WireOp>(&child))) {
                  if (type.getPort() & PortType::NONE) {
                    // TODO
                  }
                }
              }
            }
          }

          if (!match && (node.type.getType() & ValueTypeType::REGISTER)) {
            // TODO
          }
        })
        .Case<hw::HWModuleExternOp>([&](auto &op) {
          auto &node = filter.nodes[i];
          auto type = node.type;
          bool match = false;

          if (type.getType() & ValueTypeType::WIRE) {
            if (type.getPort() & PortType::INPUT) {
              for (auto &port : op.getPorts()) {
                if (port.isOutput() || !node.type.containsWidth(port.type.getIntOrFloatBitWidth())) {
                  continue;
                }

                auto name = port.getName().str();
                matchAndAppend(node, op, opStack, i, name, match);
                if (match) {
                  return;
                }
              }
            }

            if (type.getPort() & PortType::OUTPUT) {
              for (auto &port : op.getPorts()) {
                if (!port.isOutput() || !node.type.containsWidth(port.type.getIntOrFloatBitWidth())) {
                  continue;
                }

                auto name = port.getName().str();
                matchAndAppend(node, op, opStack, i, name, match);
                if (match) {
                  return;
                }
              }
            }
          }
        });

      if (filter.nodes[i].tag == FilterType::RECURSIVE_GLOB) {
        opStack.push_back(std::make_pair(op, i + 1));
      }
    }
  }

  return filtered;
}

} /* namespace query */
} /* namespace circt */
