# The Query API
The query API can be used to perform queries on CIRCT's MLIR dialects by using filters that may compose with each other. Currently, only the HW dialect is supported, but further dialects such as the FIRRTL dialect will be supported in the future.

## Filters and Filter Types
Filters and filter types are the foundation of the query API. Filters allow you to filter operations based on certain criteria, whereas filter types match against a certain property in the operation.

### List of Filters
 - `AttributeFilter`

    Filters based on if the given attribute matches.

 - `NameFilter`

    Filters based on if the operation's result name or module name matches.

 - `OpFilter`

    Filters based on if the operation's name matches.

 - `AndFilter`

    Filters based on if all child filters match.

 - `OrFilter`

    Filters based on if any child filter matches.

 - `InstanceFilter`

    Filters based on if the operation matches the given instance hierarchy.

 - `UsageFilter`

    Filters based on the given child filter and returns all usages of the operation.

### List of Filter Types
 - `GlobFilterType`

    Always matches against a given property.

 - `RecursiveGlobFilterType`

    Always matches against the given property and tells the filter to recursively apply itself.

 - `LiteralFilterType`

    Matches if the given property matches the literal provided.

 - `RegexFilterType`

    Matches if the given property matches the regex provided.

## Extending the Query API
Extending the query API may be useful if you:
 - Want to support more dialects
 - Want to add more filters
 - Want to modify the behaviour of the query API

### Adding a New Filter
Adding a new filter is as easy as extending the `Filter` class (and possibly the `FilterType` class as well).

### Subclasses of `Filter`
```cpp
class Filter {
public:
  virtual ~Filter() {
    delete type;
  }

  virtual bool matches(Operation *op, FilterData &data) { return false; }
  virtual bool addSelf() { return type->addSelf(); }
  virtual Filter *nextFilter() { return nullptr; };
  virtual Filter *clone() { return nullptr; }
  virtual std::vector<Operation *> nextOperations(Operation *op, FilterData &data) { std::vector<Operation *> ops; ops.push_back(op); return ops; }

  FilterType *getType() { return type; }

  std::vector<Operation *> filter(Operation *root, FilterData &data);
  std::vector<Operation *> filter(std::vector<Operation *> &results, FilterData &data);

protected:
  Filter(FilterType *type) : type (type) { }

  FilterType *type;
};
```

 - `virtual bool matches(Operation *, FilterData &)` (REQUIRED)

    Determines whether the given operation matches the filter.

 - `virtual bool addSelf()` (optional)

    Determines if the filter should add itself paired with child operations (useful for filters such as recursive globs).

 - `virtual Filter *nextFilter()` (optional)

    Returns the next filter to follow.

 - `virtual Filter *clone()` (REQUIRED)

  Clones the filter.

 - `virtual std::vector<Operation *> nextOperations(Operation *, FilterData &)` (optional)

    Gets the next operations to continue filtering on.

### Subclasses of `FilterType`
```cpp
class FilterType {
public:
  virtual ~FilterType() { }

  virtual bool valueMatches(llvm::StringRef value) { return false; }
  virtual bool addSelf() { return false; }
  virtual FilterType *clone() { return nullptr; }
};
```

 - `virtual bool valueMatches(StringRef)` (REQUIRED)

    Determines if some passed in property matches the filter type.

 - `virtual bool addSelf()` (optional)

    Determines if the filter type should add itself paired with child operations (useful for filter types such as recursive globs).

 - `virtual FilterType *clone()` (REQUIRED)

    Clones the filter type.

### Adding Support for a New Dialect
There are a few places where HW specific code occurs in the query API. Namely:
 - `getNextOpFromOp`, which gets subsequent operations from a given op such as instance -> referenced module.
 - `getNameFromOp`, which gets the name of an operation (such as module name, for example) and falls back to result names if not applicable.
 - `UsageFilter::nextOperations`, which gets the usages of an operation (which can be more complicated than just calling `->getUses()`; for example, modules refer to the symbol table).

## TODO
 - Support for more dialects
 - Bindings for Python and Swift
 - Type filter
