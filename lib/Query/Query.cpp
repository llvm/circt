#include "circt/Query/Query.h"

namespace circt {
namespace query {

Operation *getNextOpFromOp(Operation *op) {
  return TypeSwitch<Operation *, Operation *>(op)
    .Case<hw::InstanceOp>([&](auto &op) {
        return op.getReferencedModule();
    })
    .Default([&](auto *op) {
      return op;
    });
}

std::vector<Operation *> Filter::filter(Operation *root, FilterData &data) {
  std::vector<Operation *> filtered;
  std::vector<std::pair<Operation *, Filter *>> opStack;

  opStack.push_back(std::make_pair(root, this));
  while (!opStack.empty()) {
    auto pair = opStack[opStack.size() - 1];
    Operation *op = pair.first;
    Filter *filter = pair.second;
    opStack.pop_back();

    if (filter == nullptr) {
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
      if (filter->addSelf() && !llvm::dyn_cast_or_null<mlir::ModuleOp>(op) && filter->matches(op, data)) {
        opStack.push_back(std::make_pair(op, filter->nextFilter()));
      }

      for (auto &region : op->getRegions()) {
        for (auto &block : region) {
          for (auto &op : block) {
            auto *next = getNextOpFromOp(&op);

            if (filter->matches(next, data)) {
              auto vec = filter->nextOperations(next, data);

              if (filter->addSelf()) {
                for (auto *op : vec) {
                  opStack.push_back(std::make_pair(op, filter));
                }
              }

              auto *child = filter->nextFilter();
              for (auto *op : vec) {
                opStack.push_back(std::make_pair(op, child));
              }
            }
          }
        }
      }
    }
  }

  return filtered;
}

std::vector<Operation *> Filter::filter(std::vector<Operation *> &results, FilterData &data) {
  std::vector<Operation *> result;
  for (auto *op : results) {
    auto vec = filter(op, data);
    for (auto *op : vec) {
      bool contained = false;
      for (auto *o : result) {
        if (o == op) {
          contained = true;
          break;
        }
      }

      if (!contained) {
        result.push_back(op);
      }
    }
  }

  return result;
}

std::string getNameFromOp(Operation *op, size_t nameIndex) {
  return TypeSwitch<Operation *, std::string>(op)
    .Case<hw::HWModuleOp, hw::HWModuleExternOp>([&](auto &op) {
      if (nameIndex == 0) {
        return op.getNameAttr().getValue().str();
      }
      return std::string();
    })
    .Default([&](auto *op) {
      if (nameIndex < op->getNumResults()) {
        std::string str;
        llvm::raw_string_ostream stream(str);
        op->getResult(nameIndex).print(stream);
        return str;
      }
      return std::string();
    });
}

bool filterAttribute(Attribute &attr, FilterType *type) {
  return TypeSwitch<Attribute, bool>(attr)
    .Case<mlir::BoolAttr>([&](auto &attr) {
        std::string value(attr.getValue() ? "true" : "false");
        return type->valueMatches(value);
    })
    .Case<mlir::IntegerAttr>([&](auto &attr) {
        std::stringstream stream;
        stream << attr.getValue().getZExtValue();
        std::string s;
        stream.str(s);
        return type->valueMatches(s);
    })
    .Case<mlir::StringAttr>([&](StringAttr &attr) {
        auto s = attr.getValue().str();
        return type->valueMatches(s);
    })
    .Case<mlir::ArrayAttr>([&](ArrayAttr &attr) {
        for (auto v : attr) {
          if (filterAttribute(v, type)) {
            return true;
          }
        }
        return false;
    })
    .Case<mlir::DictionaryAttr>([&](auto &attr) {
        std::cerr << "warning: unknown attribute type\n";
        return false;
    })
    .Default([&](auto &attr) {
        std::cerr << "warning: unknown attribute type\n";
        return false;
    });
}

bool AttributeFilter::matches(Operation *op, FilterData &data) {
  if (!op->hasAttr(StringRef(key))) {
    return false;
  }

  auto attr = op->getAttr(StringRef(key));
  return filterAttribute(attr, type);
}

Filter *AttributeFilter::clone() {
  return new AttributeFilter(key, type->clone());
}

bool NameFilter::matches(Operation *op, FilterData &data) {
  std::string name;
  for (size_t nameIndex = 0; nameIndex == 0 || !(name = getNameFromOp(op, nameIndex)).empty(); nameIndex++) {
    if (type->valueMatches(name)) {
      return true;
    }
  }
  return false;
}

Filter *NameFilter::clone() {
  return new NameFilter(type->clone());
}

bool OpFilter::matches(Operation *op, FilterData &data) {
  std::string s(op->getName().stripDialect().str());
  return type->valueMatches(s);
}

Filter *OpFilter::clone() {
  return new OpFilter(type->clone());
}

bool AndFilter::matches(Operation *op, FilterData &data) {
  for (auto *filter : filters) {
    if (filter->filter(op, data).empty()) {
      return false;
    }
  }

  return true;
}

Filter *AndFilter::clone() {
  return new AndFilter(filters);
}

bool OrFilter::matches(Operation *op, FilterData &data) {
  for (auto *filter : filters) {
    if (!filter->filter(op, data).empty()) {
      return true;
    }
  }

  return false;
}

Filter *OrFilter::clone() {
  return new OrFilter(filters);
}

bool InstanceFilter::matches(Operation *op, FilterData &data) {
  return !filter->filter(op, data).empty();
}

bool InstanceFilter::addSelf() {
  return filter->getType()->addSelf();
}

Filter *InstanceFilter::clone() {
  return new InstanceFilter(filter, child);
}

Filter *InstanceFilter::nextFilter() {
  return child;
}

std::vector<Operation *> UsageFilter::nextOperations(Operation *op, FilterData &data) {
  std::vector<Operation *> ops;
  TypeSwitch<Operation *>(op)
    .Case<hw::HWModuleOp>([&](auto &op) {
      std::vector<Operation *> ops;
      auto users = data.userMap.getUsers(op);
      ops.insert(ops.end(), users.begin(), users.end());
    })
    .Default([&](auto *op) {
      for (auto &use : op->getUses()) {
        ops.push_back(use.getOwner());
      }
    });
  return ops;
}

op_attr_map dumpAttributes(std::vector<Operation *> &results, ArrayRef<StringRef> filters) {
  op_attr_map result;

  for (auto *op : results) {
    attr_map attrs;

    if (filters.empty()) {
      for (auto attr : op->getAttrs()) {
        attrs.push_back(std::make_pair(attr.first.strref(), attr.second));
      }
    } else {
      for (auto attrName : filters) {
        if (op->hasAttr(attrName)) {
          attrs.push_back(std::make_pair(attrName, op->getAttr(attrName)));
        }
      }
    }

    result.push_back(std::make_pair(op, attrs));
  }
  return result;
}

} /* namespace query */
} /* namespace circt */
