//===- Report.cpp - Indexed FIRRTL domain inference report ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Report.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <climits>
#include <filesystem>
#include <optional>
#include <queue>
#include <unordered_set>
#include <utility>

using llvm::json::Array;
using llvm::json::Object;
using JsonValue = llvm::json::Value;

namespace circt::domain_report {
namespace {

// This reader keeps one input buffer and one JSON record in memory at a time.
// In particular, a module or the entire provenance array is never a JSON DOM.
class Reader {
public:
  Reader(std::istream &input, std::function<void(uint64_t)> progress)
      : input(input), progress(std::move(progress)) {}

  const std::string &error() const { return message; }
  bool finished() { return skipSpace() && peek() == EOF; }

  template <typename Fn>
  bool object(Fn &&field) {
    if (!expect('{'))
      return false;
    if (consume('}'))
      return true;
    do {
      std::string key;
      if (!string(key) || !expect(':') || !field(key))
        return false;
      if (consume('}'))
        return true;
    } while (expect(','));
    return false;
  }

  template <typename Fn>
  bool array(Fn &&element) {
    if (!expect('['))
      return false;
    if (consume(']'))
      return true;
    size_t index = 0;
    do {
      if (!element(index++))
        return false;
      if (consume(']'))
        return true;
    } while (expect(','));
    return false;
  }

  std::optional<JsonValue> value(unsigned depth = 0) {
    if (depth > 64) {
      fail("JSON nesting is too deep");
      return std::nullopt;
    }
    skipSpace();
    int c = peek();
    if (c == '{') {
      Object result;
      if (!object([&](const std::string &key) {
            auto child = value(depth + 1);
            if (!child)
              return false;
            result[key] = std::move(*child);
            return true;
          }))
        return std::nullopt;
      return JsonValue(std::move(result));
    }
    if (c == '[') {
      Array result;
      if (!array([&](size_t) {
            auto child = value(depth + 1);
            if (!child)
              return false;
            result.push_back(std::move(*child));
            return true;
          }))
        return std::nullopt;
      return JsonValue(std::move(result));
    }
    if (c == '"') {
      std::string result;
      if (!string(result))
        return std::nullopt;
      return JsonValue(std::move(result));
    }
    if (c == 't' || c == 'f' || c == 'n') {
      std::string literal = c == 't' ? "true" : c == 'f' ? "false" : "null";
      for (char expected : literal)
        if (take() != expected) {
          fail("invalid JSON literal");
          return std::nullopt;
        }
      if (literal == "null")
        return JsonValue(nullptr);
      return JsonValue(literal == "true");
    }
    std::string token;
    while (c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' ||
           (c >= '0' && c <= '9')) {
      token.push_back(static_cast<char>(take()));
      c = peek();
    }
    if (token.empty()) {
      fail("expected a JSON value");
      return std::nullopt;
    }
    if (token.find_first_of(".eE+") == std::string::npos &&
        !(token[0] == '0' && token.size() > 1) &&
        !(token.size() > 2 && token[0] == '-' && token[1] == '0')) {
      int64_t number;
      auto parsed =
          std::from_chars(token.data(), token.data() + token.size(), number);
      if (parsed.ec == std::errc() && parsed.ptr == token.data() + token.size())
        return JsonValue(number);
    }
    auto parsed = llvm::json::parse(token);
    if (!parsed) {
      fail("invalid JSON number");
      return std::nullopt;
    }
    return std::move(*parsed);
  }

  bool skipValue(unsigned depth = 0) {
    if (depth > 64)
      return fail("JSON nesting is too deep");
    skipSpace();
    if (peek() == '{')
      return object([&](const std::string &) { return skipValue(depth + 1); });
    if (peek() == '[')
      return array([&](size_t) { return skipValue(depth + 1); });
    return value(depth).has_value();
  }

  bool nullableInteger(std::optional<int64_t> &result) {
    skipSpace();
    if (peek() == 'n') {
      auto parsed = value();
      if (!parsed || !parsed->getAsNull())
        return fail("expected null or integer");
      result.reset();
      return true;
    }
    std::string digits;
    if (peek() == '-')
      digits.push_back(static_cast<char>(take()));
    bool leadingZero = peek() == '0';
    while (peek() >= '0' && peek() <= '9')
      digits.push_back(static_cast<char>(take()));
    int64_t number;
    size_t digitsStart = !digits.empty() && digits.front() == '-' ? 1 : 0;
    if (digits.empty() || digits == "-" ||
        (leadingZero && digits.size() - digitsStart > 1) ||
        std::from_chars(digits.data(), digits.data() + digits.size(), number)
                .ec != std::errc())
      return fail("expected a 64-bit integer");
    result = number;
    return true;
  }

  bool fail(llvm::StringRef reason) {
    if (message.empty())
      message = llvm::formatv("{0} at byte {1}", reason, offset).str();
    return false;
  }

private:
  int peek() {
    if (position == available) {
      input.read(buffer.data(), buffer.size());
      available = static_cast<size_t>(input.gcount());
      position = 0;
      if (!available)
        return EOF;
    }
    return static_cast<unsigned char>(buffer[position]);
  }
  int take() {
    int c = peek();
    if (c != EOF) {
      ++position;
      ++offset;
      if (progress && offset >= nextProgress) {
        progress(offset);
        nextProgress += 32ULL * 1024 * 1024;
      }
    }
    return c;
  }
  bool skipSpace() {
    while (peek() == ' ' || peek() == '\t' || peek() == '\r' || peek() == '\n')
      take();
    return true;
  }
  bool consume(char expected) {
    skipSpace();
    if (peek() != expected)
      return false;
    take();
    return true;
  }
  bool expect(char expected) {
    if (consume(expected))
      return true;
    return fail(llvm::formatv("expected '{0}'", expected).str());
  }
  bool string(std::string &result) {
    if (!expect('"'))
      return false;
    std::string raw;
    bool escaped = false;
    bool needsDecode = false;
    while (true) {
      int c = take();
      if (c == EOF)
        return fail("unterminated JSON string");
      if (c < 0x20)
        return fail("unescaped control character in JSON string");
      if (c == '"' && !escaped)
        break;
      raw.push_back(static_cast<char>(c));
      if (raw.size() > 64ULL * 1024 * 1024)
        return fail("JSON string exceeds 64 MiB");
      if (c == '\\' && !escaped) {
        escaped = true;
        needsDecode = true;
      } else {
        escaped = false;
      }
    }
    if (!needsDecode) {
      if (!llvm::json::isUTF8(raw))
        return fail("invalid UTF-8 in JSON string");
      result = std::move(raw);
      return true;
    }
    std::string quoted = "\"" + raw + "\"";
    auto parsed = llvm::json::parse(quoted);
    if (!parsed || !parsed->getAsString())
      return fail("invalid JSON string");
    result = parsed->getAsString()->str();
    return true;
  }

  std::istream &input;
  std::function<void(uint64_t)> progress;
  std::array<char, 64 * 1024> buffer;
  size_t position = 0;
  size_t available = 0;
  uint64_t offset = 0;
  uint64_t nextProgress = 32ULL * 1024 * 1024;
  std::string message;
};

std::optional<int64_t> integer(const Object &object, llvm::StringRef name) {
  return object.getInteger(name);
}

int64_t optionalInteger(const Object &object, llvm::StringRef name) {
  return object.getInteger(name).value_or(-1);
}

uint32_t tableIndex(const Object &object, llvm::StringRef name) {
  auto number = object.getInteger(name);
  if (!number || *number < 0 || *number >= Report::none)
    return Report::none;
  return static_cast<uint32_t>(*number);
}

std::string stringOrEmpty(const Object &object, llvm::StringRef name) {
  auto value = object.getString(name);
  return value ? value->str() : std::string();
}

std::optional<int64_t> parseId(const Object &params, llvm::StringRef name) {
  auto text = params.getString(name);
  if (!text)
    return std::nullopt;
  int64_t result;
  auto parsed = std::from_chars(text->begin(), text->end(), result);
  if (parsed.ec != std::errc() || parsed.ptr != text->end())
    return std::nullopt;
  return result;
}

std::string normalized(const std::string &path) {
  return std::filesystem::path(path).lexically_normal().string();
}

bool containsInsensitive(llvm::StringRef text, llvm::StringRef needle) {
  return std::search(text.begin(), text.end(), needle.begin(), needle.end(),
                     [](unsigned char a, unsigned char b) {
                       return std::tolower(a) == std::tolower(b);
                     }) != text.end();
}

} // namespace

bool Report::load(std::istream &input, std::string &error,
                  std::function<void(uint64_t)> progress) {
  error.clear();
  *this = Report();
  Reader reader(input, std::move(progress));
  auto addDomain = [&](const JsonValue &record) {
    auto *object = record.getAsObject();
    if (!object || !integer(*object, "id") || !object->getString("name"))
      return reader.fail("invalid domain record");
    Domain domain;
    domain.id = *integer(*object, "id");
    domain.name = stringOrEmpty(*object, "name");
    domain.location = tableIndex(*object, "location_id");
    if (domain.id < 0 || domain.location == none ||
        !domainById.emplace(domain.id, domains.size()).second)
      return reader.fail("invalid or duplicate domain ID");
    domains.push_back(std::move(domain));
    return true;
  };
  auto addLocation = [&](const JsonValue &record) {
    auto *object = record.getAsObject();
    if (!object || !object->getString("display") ||
        !object->getArray("sources"))
      return reader.fail("invalid location record");
    Location location;
    location.display = stringOrEmpty(*object, "display");
    for (const auto &item : *object->getArray("sources")) {
      auto *sourceObject = item.getAsObject();
      if (!sourceObject || !sourceObject->getString("file"))
        return reader.fail("invalid source location");
      Source source;
      source.file = stringOrEmpty(*sourceObject, "file");
      source.range = sourceObject->getInteger("start_line").has_value();
      source.line =
          tableIndex(*sourceObject, source.range ? "start_line" : "line");
      source.column =
          tableIndex(*sourceObject, source.range ? "start_column" : "column");
      if (source.range) {
        source.endLine = tableIndex(*sourceObject, "end_line");
        source.endColumn = tableIndex(*sourceObject, "end_column");
      }
      if (!source.line || !source.column || source.line == none ||
          source.column == none ||
          (source.range &&
           (!source.endLine || !source.endColumn || source.endLine == none ||
            source.endColumn == none ||
            std::pair(source.endLine, source.endColumn) <
                std::pair(source.line, source.column))))
        return reader.fail("invalid source coordinates");
      location.sources.push_back(std::move(source));
    }
    locations.push_back(std::move(location));
    return true;
  };
  auto addValue = [&](const JsonValue &record, uint32_t moduleIndex) {
    auto *object = record.getAsObject();
    if (!object || !integer(*object, "id") || !object->getString("name") ||
        !object->getString("kind"))
      return reader.fail("invalid value record");
    Value value;
    value.id = *integer(*object, "id");
    value.name = stringOrEmpty(*object, "name");
    value.kind = stringOrEmpty(*object, "kind");
    value.definition = stringOrEmpty(*object, "definition");
    value.direction = stringOrEmpty(*object, "port_direction");
    value.module = moduleIndex;
    value.type = tableIndex(*object, "type_id");
    value.location = tableIndex(*object, "location_id");
    value.domainType = optionalInteger(*object, "domain_type_id");
    value.instance = optionalInteger(*object, "instance_id");
    value.portIndex =
        static_cast<int32_t>(optionalInteger(*object, "port_index"));
    value.instancePortIndex =
        static_cast<int32_t>(optionalInteger(*object, "instance_port_index"));
    value.assignmentBegin = assignments.size();
    if (auto *items = object->getArray("domain_assignments")) {
      for (const auto &item : *items) {
        auto *assignmentObject = item.getAsObject();
        if (!assignmentObject ||
            !assignmentObject->getInteger("domain_type_id") ||
            !assignmentObject->getBoolean("inferred"))
          return reader.fail("invalid domain assignment");
        Assignment assignment;
        assignment.domainType = *assignmentObject->getInteger("domain_type_id");
        assignment.domainValue =
            optionalInteger(*assignmentObject, "domain_value_id");
        assignment.inferred = *assignmentObject->getBoolean("inferred");
        assignments.push_back(assignment);
      }
    }
    value.assignmentCount = assignments.size() - value.assignmentBegin;
    if (value.id < 0 || value.type == none || value.location == none ||
        values.size() >= none ||
        !valueById.emplace(value.id, values.size()).second)
      return reader.fail("invalid or duplicate value ID");
    modules[moduleIndex].values.push_back(values.size());
    values.push_back(std::move(value));
    return true;
  };
  auto addPort = [&](const JsonValue &record, uint32_t moduleIndex) {
    auto *object = record.getAsObject();
    if (!object || !integer(*object, "index") ||
        *integer(*object, "index") < 0 ||
        *integer(*object, "index") > INT32_MAX)
      return reader.fail("invalid port record");
    Port port;
    port.index = static_cast<int32_t>(*integer(*object, "index"));
    port.value = optionalInteger(*object, "value_id");
    port.name = stringOrEmpty(*object, "name");
    port.direction = stringOrEmpty(*object, "direction");
    port.type = tableIndex(*object, "type_id");
    port.location = tableIndex(*object, "location_id");
    if (auto *items = object->getArray("domain_assignments")) {
      for (const auto &item : *items) {
        auto *assignmentObject = item.getAsObject();
        if (!assignmentObject ||
            !assignmentObject->getInteger("domain_type_id") ||
            !assignmentObject->getInteger("domain_port_index") ||
            !assignmentObject->getBoolean("inferred"))
          return reader.fail("invalid port assignment");
        PortAssignment assignment;
        assignment.domainType = *assignmentObject->getInteger("domain_type_id");
        assignment.domainPortValue =
            optionalInteger(*assignmentObject, "domain_port_value_id");
        assignment.domainPortIndex = static_cast<int32_t>(
            *assignmentObject->getInteger("domain_port_index"));
        assignment.inferred = *assignmentObject->getBoolean("inferred");
        port.assignments.push_back(assignment);
      }
    }
    modules[moduleIndex].ports.push_back(std::move(port));
    return true;
  };
  auto addInstance = [&](const JsonValue &record, uint32_t moduleIndex) {
    auto *object = record.getAsObject();
    if (!object || !integer(*object, "id") || !object->getString("name") ||
        !object->getArray("targets"))
      return reader.fail("invalid instance record");
    Instance instance;
    instance.id = *integer(*object, "id");
    instance.name = stringOrEmpty(*object, "name");
    instance.module = moduleIndex;
    instance.location = tableIndex(*object, "location_id");
    for (const auto &target : *object->getArray("targets")) {
      if (auto id = target.getAsInteger())
        instance.targets.push_back(std::to_string(*id));
      else if (auto name = target.getAsString())
        instance.targets.push_back(name->str());
      else
        return reader.fail("invalid instance target");
    }
    if (auto *items = object->getArray("effective_domain_bindings")) {
      for (const auto &item : *items) {
        auto *bindingObject = item.getAsObject();
        if (!bindingObject || !bindingObject->getInteger("target_module_id") ||
            !bindingObject->getInteger("domain_type_id"))
          return reader.fail("invalid effective binding");
        Binding binding;
        binding.targetModule = *bindingObject->getInteger("target_module_id");
        binding.portIndex =
            static_cast<int32_t>(optionalInteger(*bindingObject, "port_index"));
        binding.portValue = optionalInteger(*bindingObject, "port_value_id");
        binding.domainType = *bindingObject->getInteger("domain_type_id");
        binding.domainPortIndex = static_cast<int32_t>(
            optionalInteger(*bindingObject, "domain_port_index"));
        binding.effectiveDomainValue =
            optionalInteger(*bindingObject, "effective_domain_value_id");
        binding.effectiveDomainValueName =
            stringOrEmpty(*bindingObject, "effective_domain_value_name");
        binding.location = tableIndex(*bindingObject, "location_id");
        instance.bindings.push_back(std::move(binding));
      }
    }
    if (instance.id < 0 || instance.location == none ||
        instances.size() >= none ||
        !instanceById.emplace(instance.id, instances.size()).second)
      return reader.fail("invalid or duplicate instance ID");
    modules[moduleIndex].instances.push_back(instances.size());
    instances.push_back(std::move(instance));
    return true;
  };
  auto addModule = [&]() {
    if (modules.size() >= none)
      return reader.fail("too many modules");
    uint32_t moduleIndex = modules.size();
    modules.emplace_back();
    bool valid = reader.object([&](const std::string &key) {
      if (key == "id" || key == "name" || key == "kind") {
        auto item = reader.value();
        if (!item)
          return false;
        if (key == "id") {
          auto id = item->getAsInteger();
          if (!id)
            return reader.fail("invalid module ID");
          modules[moduleIndex].id = *id;
        } else if (key == "name") {
          auto name = item->getAsString();
          if (!name)
            return reader.fail("invalid module name");
          modules[moduleIndex].name = name->str();
        } else {
          auto kind = item->getAsString();
          if (!kind)
            return reader.fail("invalid module kind");
          modules[moduleIndex].kind = kind->str();
        }
        return true;
      }
      if (key == "ports" || key == "values" || key == "instances")
        return reader.array([&](size_t) {
          auto item = reader.value();
          if (!item)
            return false;
          if (key == "ports")
            return addPort(*item, moduleIndex);
          if (key == "values")
            return addValue(*item, moduleIndex);
          return addInstance(*item, moduleIndex);
        });
      return reader.skipValue();
    });
    if (!valid)
      return false;
    auto &module = modules[moduleIndex];
    if (module.id < 0 || module.name.empty() ||
        (module.kind != "module" && module.kind != "extmodule") ||
        !moduleById.emplace(module.id, moduleIndex).second)
      return reader.fail("invalid or duplicate module ID");
    return true;
  };
  struct CrossingIDs {
    int64_t owner;
    int64_t domain;
    int64_t location;
    int64_t operation;
    int64_t lhs;
    int64_t rhs;
    int64_t lhsDomain;
    int64_t rhsDomain;
    int64_t lhsSource;
    int64_t rhsSource;
  };
  std::vector<CrossingIDs> crossingIDs;
  auto addCrossing = [&](const JsonValue &record) {
    auto *object = record.getAsObject();
    if (!object || !integer(*object, "owner_module_id") ||
        !integer(*object, "domain_type_id") ||
        !integer(*object, "location_id") ||
        !integer(*object, "operation_kind_id"))
      return reader.fail("invalid illegal crossing record");
    crossingIDs.push_back({*integer(*object, "owner_module_id"),
                           *integer(*object, "domain_type_id"),
                           *integer(*object, "location_id"),
                           *integer(*object, "operation_kind_id"),
                           optionalInteger(*object, "lhs_value_id"),
                           optionalInteger(*object, "rhs_value_id"),
                           optionalInteger(*object, "lhs_domain_value_id"),
                           optionalInteger(*object, "rhs_domain_value_id"),
                           optionalInteger(*object, "lhs_source_value_id"),
                           optionalInteger(*object, "rhs_source_value_id")});
    return true;
  };
  std::array<int, 10> edgePositions;
  bool edgePositionsReady = false;
  static constexpr std::array<llvm::StringLiteral, 10> expectedEdgeFields = {
      "owner_module_id", "kind_id",
      "domain_type_id",  "location_id",
      "flags",           "lhs_value_id",
      "rhs_value_id",    "operation_kind_id",
      "instance_id",     "target_module_id"};
  auto addEdge = [&](Reader &edgeReader) {
    if (!edgePositionsReady) {
      for (size_t i = 0; i < expectedEdgeFields.size(); ++i) {
        auto found = std::find_if(edgeFields.begin(), edgeFields.end(),
                                  [&](const std::string &field) {
                                    return llvm::StringRef(field) ==
                                           expectedEdgeFields[i];
                                  });
        if (found == edgeFields.end())
          return edgeReader.fail("missing provenance edge field");
        edgePositions[i] = found - edgeFields.begin();
      }
      edgePositionsReady = true;
    }
    std::array<std::optional<int64_t>, 10> row;
    size_t fieldCount = 0;
    if (!edgeReader.array([&](size_t index) {
          if (index >= row.size())
            return edgeReader.fail("too many provenance edge fields");
          std::optional<int64_t> field;
          if (!edgeReader.nullableInteger(field))
            return false;
          row[index] = field;
          fieldCount = index + 1;
          return true;
        }))
      return false;
    if (fieldCount != edgeFields.size())
      return edgeReader.fail(
          "provenance edge fields are missing or mismatched");
    auto get = [&](size_t field) -> std::optional<int64_t> {
      return row[edgePositions[field]];
    };
    auto ownerId = get(0);
    auto kind = get(1);
    auto domainId = get(2);
    auto location = get(3);
    auto flags = get(4);
    if (!ownerId || !kind || !domainId || !location || !flags || *kind < 0 ||
        *kind >= none || *location < 0 || *location >= none || *flags < 0 ||
        *flags >= none)
      return edgeReader.fail("invalid provenance edge");
    Edge edge;
    edge.kind = *kind;
    edge.location = *location;
    edge.flags = *flags;
    auto resolve = [&](std::optional<int64_t> id, const auto &table) {
      if (!id)
        return none;
      auto it = table.find(*id);
      return it == table.end() ? none : it->second;
    };
    edge.owner = resolve(ownerId, moduleById);
    edge.domain = resolve(domainId, domainById);
    edge.lhs = resolve(get(5), valueById);
    edge.rhs = resolve(get(6), valueById);
    edge.instance = resolve(get(8), instanceById);
    edge.target = resolve(get(9), moduleById);
    auto operation = get(7);
    if (operation) {
      if (*operation < 0 || *operation >= none)
        return edgeReader.fail("invalid operation kind ID");
      edge.operation = *operation;
    }
    if (edge.owner == none || edge.domain == none ||
        (get(5) && edge.lhs == none) || (get(6) && edge.rhs == none) ||
        (get(8) && edge.instance == none) || (get(9) && edge.target == none)) {
      ++skippedEdges;
      return true;
    }
    if (edges.size() >= none)
      return edgeReader.fail("too many provenance edges");
    edges.push_back(edge);
    return true;
  };

  std::unordered_set<std::string> topLevelFields;
  bool deferredEdges = false;
  bool parsed = reader.object([&](const std::string &key) {
    if (!topLevelFields.insert(key).second)
      return reader.fail("duplicate top-level field");
    if (key == "modules")
      return reader.array([&](size_t) { return addModule(); });
    if (key == "provenance_edges") {
      if (!topLevelFields.count("modules") ||
          !topLevelFields.count("domains") ||
          !topLevelFields.count("provenance_edge_fields")) {
        deferredEdges = true;
        return reader.skipValue();
      }
      return reader.array([&](size_t) { return addEdge(reader); });
    }
    auto readArray = [&](auto &&add) {
      return reader.array([&](size_t) {
        auto item = reader.value();
        return item && add(*item);
      });
    };
    if (key == "domains")
      return readArray(addDomain);
    if (key == "illegal_crossings") {
      hasIllegalCrossings = true;
      return readArray(addCrossing);
    }
    if (key == "locations")
      return readArray(addLocation);
    if (key == "types" || key == "operation_kinds" ||
        key == "provenance_edge_kinds" || key == "provenance_edge_fields")
      return readArray([&](const JsonValue &item) {
        auto text = item.getAsString();
        if (!text)
          return reader.fail("expected a string table entry");
        auto &table = key == "types"                   ? types
                      : key == "operation_kinds"       ? operationKinds
                      : key == "provenance_edge_kinds" ? edgeKinds
                                                       : edgeFields;
        table.push_back(text->str());
        return true;
      });
    if (key != "format" && key != "version" && key != "complete" &&
        key != "instance_binding_direction" && key != "provenance_edge_flags")
      return reader.skipValue();
    auto item = reader.value();
    if (!item)
      return false;
    if (key == "format") {
      hasFormat = item->getAsString() &&
                  *item->getAsString() == "circt-domain-inference";
      return true;
    }
    if (key == "version") {
      hasVersion = item->getAsInteger().has_value();
      version = item->getAsInteger().value_or(-1);
      return true;
    }
    if (key == "complete") {
      hasComplete = item->getAsBoolean().has_value();
      complete = item->getAsBoolean().value_or(false);
      return true;
    }
    if (key == "instance_binding_direction") {
      if (!item->getAsString() ||
          *item->getAsString() != "parent_instance_to_module_template")
        return reader.fail("unsupported instance binding direction");
      return true;
    }
    if (key == "provenance_edge_flags") {
      auto *flags = item->getAsObject();
      auto inferred = flags ? flags->getInteger("inferred") : std::nullopt;
      auto summarized = flags ? flags->getInteger("summarized") : std::nullopt;
      if (!inferred || !summarized || *inferred <= 0 || *summarized <= 0 ||
          *inferred > INT_MAX || *summarized > INT_MAX ||
          (*inferred & (*inferred - 1)) || (*summarized & (*summarized - 1)) ||
          *inferred == *summarized)
        return reader.fail("invalid provenance edge flags");
      inferredFlag = *inferred;
      summarizedFlag = *summarized;
    }
    return true;
  });
  if (!parsed || !reader.finished()) {
    error = reader.error().empty() ? "invalid trailing JSON" : reader.error();
    return false;
  }
  if (!hasFormat || !hasVersion || version != 3 || !hasComplete) {
    error = "expected a circt-domain-inference report with schema version 3";
    return false;
  }
  for (llvm::StringRef field :
       {"types", "locations", "operation_kinds", "domains", "modules",
        "provenance_edge_kinds", "provenance_edge_fields", "provenance_edges",
        "provenance_edge_flags", "instance_binding_direction"}) {
    if (!topLevelFields.count(field.str())) {
      error = llvm::formatv("report is missing '{0}'", field).str();
      return false;
    }
  }
  if (edgeFields.size() != 10 ||
      std::unordered_set<std::string>(edgeFields.begin(), edgeFields.end())
              .size() != edgeFields.size()) {
    error = "invalid provenance edge field table";
    return false;
  }
  for (llvm::StringRef field : expectedEdgeFields)
    if (std::find_if(edgeFields.begin(), edgeFields.end(),
                     [&](const std::string &entry) {
                       return llvm::StringRef(entry) == field;
                     }) == edgeFields.end()) {
      error =
          llvm::formatv("report is missing provenance edge field '{0}'", field)
              .str();
      return false;
    }
  if (deferredEdges) {
    input.clear();
    input.seekg(0, std::ios::beg);
    if (!input) {
      error = "reordered report requires a seekable input file";
      return false;
    }
    Reader replay(input, {});
    bool foundEdges = false;
    bool replayed = replay.object([&](const std::string &key) {
      if (key == "provenance_edges") {
        foundEdges = true;
        return replay.array([&](size_t) { return addEdge(replay); });
      }
      return replay.skipValue();
    });
    if (!replayed || !replay.finished() || !foundEdges) {
      error = replay.error().empty()
                  ? "could not read deferred provenance edges"
                  : replay.error();
      return false;
    }
  }
  if (complete && skippedEdges) {
    error = "complete report contains provenance edges with missing references";
    return false;
  }
  crossingsByDomain.resize(domains.size());
  for (const auto &ids : crossingIDs) {
    auto owner = moduleById.find(ids.owner);
    auto domain = domainById.find(ids.domain);
    if (owner == moduleById.end() || domain == domainById.end() ||
        ids.location < 0 ||
        ids.location >= static_cast<int64_t>(locations.size()) ||
        ids.operation < 0 ||
        ids.operation >= static_cast<int64_t>(operationKinds.size())) {
      error = "illegal crossing references an invalid table index";
      return false;
    }
    auto resolveValue = [&](int64_t id) -> uint32_t {
      auto found = valueById.find(id);
      return found == valueById.end() ? none : found->second;
    };
    Crossing crossing;
    crossing.owner = owner->second;
    crossing.domain = domain->second;
    crossing.location = ids.location;
    crossing.operation = ids.operation;
    crossing.lhs = resolveValue(ids.lhs);
    crossing.rhs = resolveValue(ids.rhs);
    crossing.lhsDomain = resolveValue(ids.lhsDomain);
    crossing.rhsDomain = resolveValue(ids.rhsDomain);
    crossing.lhsSource = resolveValue(ids.lhsSource);
    crossing.rhsSource = resolveValue(ids.rhsSource);
    if (complete &&
        (crossing.lhs == none || crossing.rhs == none ||
         crossing.lhsDomain == none || crossing.rhsDomain == none ||
         crossing.lhsSource == none || crossing.rhsSource == none)) {
      error = "complete report contains a crossing with missing values";
      return false;
    }
    crossingsByDomain[crossing.domain].push_back(crossings.size());
    crossings.push_back(crossing);
  }
  for (const auto &domain : domains)
    if (domain.location >= locations.size()) {
      error = "domain references an invalid location";
      return false;
    }
  for (uint32_t moduleIndex = 0; moduleIndex < modules.size(); ++moduleIndex) {
    const auto &module = modules[moduleIndex];
    for (const auto &port : module.ports) {
      if (module.kind == "module" && port.value < 0 && complete) {
        error = "module port is missing its value ID";
        return false;
      }
      if (port.value >= 0) {
        auto found = valueById.find(port.value);
        if (found == valueById.end() ||
            values[found->second].module != moduleIndex) {
          if (complete) {
            error = "module port references a missing value";
            return false;
          }
        }
      } else if (port.type >= types.size() ||
                 port.location >= locations.size()) {
        error = "external module port references an invalid table index";
        return false;
      }
      for (const auto &assignment : port.assignments)
        if (!domainById.count(assignment.domainType) ||
            assignment.domainPortIndex < 0 ||
            std::none_of(
                module.ports.begin(), module.ports.end(),
                [&](const Port &candidate) {
                  return candidate.index == assignment.domainPortIndex;
                })) {
          error = "port assignment references an invalid domain or port";
          return false;
        }
    }
  }
  valuesByLocation.resize(locations.size());
  for (uint32_t index = 0; index < values.size(); ++index) {
    const auto &value = values[index];
    if (value.type >= types.size() || value.location >= locations.size()) {
      error = "value references an invalid type or location";
      return false;
    }
    if ((value.domainType >= 0 && !domainById.count(value.domainType)) ||
        (value.instance >= 0 && !instanceById.count(value.instance) &&
         complete)) {
      error = "value references a missing domain or instance";
      return false;
    }
    for (uint32_t i = value.assignmentBegin;
         i < value.assignmentBegin + value.assignmentCount; ++i)
      if (!domainById.count(assignments[i].domainType) ||
          (complete && assignments[i].domainValue >= 0 &&
           !valueById.count(assignments[i].domainValue))) {
        error = "domain assignment references a missing domain or value";
        return false;
      }
    valuesByLocation[value.location].push_back(index);
  }
  for (const auto &instance : instances) {
    if (instance.location >= locations.size()) {
      error = "instance references an invalid location";
      return false;
    }
    for (const auto &binding : instance.bindings)
      if (binding.location >= locations.size() ||
          !domainById.count(binding.domainType) || binding.portIndex < 0 ||
          binding.domainPortIndex < 0 ||
          (complete && !moduleById.count(binding.targetModule))) {
        error = "effective binding references a missing module or domain";
        return false;
      }
  }
  adjacencyOffsets.assign(values.size() + 1, 0);
  for (const auto &edge : edges) {
    if (edge.location >= locations.size() || edge.kind >= edgeKinds.size() ||
        (edge.operation != none && edge.operation >= operationKinds.size())) {
      error = "provenance edge references an invalid table index";
      return false;
    }
    if (edge.lhs != none)
      ++adjacencyOffsets[edge.lhs + 1];
    if (edge.rhs != none && edge.rhs != edge.lhs)
      ++adjacencyOffsets[edge.rhs + 1];
  }
  for (size_t index = 1; index < adjacencyOffsets.size(); ++index)
    adjacencyOffsets[index] += adjacencyOffsets[index - 1];
  adjacencyEdges.resize(adjacencyOffsets.back());
  auto cursor = adjacencyOffsets;
  for (uint32_t index = 0; index < edges.size(); ++index) {
    const auto &edge = edges[index];
    if (edge.lhs != none)
      adjacencyEdges[cursor[edge.lhs]++] = index;
    if (edge.rhs != none && edge.rhs != edge.lhs)
      adjacencyEdges[cursor[edge.rhs]++] = index;
  }
  domainNodes.resize(domains.size());
  domainEdges.resize(domains.size());
  domainAssociations.resize(domains.size());
  domainComponents.resize(domains.size());
  unmappedAssociations.resize(domains.size());
  unmappedCrossings.resize(domains.size());
  std::vector<std::vector<uint32_t>> domainAnchors(domains.size());
  for (uint32_t valueIndex = 0; valueIndex < values.size(); ++valueIndex) {
    const auto &value = values[valueIndex];
    if (value.domainType >= 0)
      domainAnchors[domainById.at(value.domainType)].push_back(valueIndex);
    for (uint32_t i = value.assignmentBegin;
         i < value.assignmentBegin + value.assignmentCount; ++i) {
      const auto &assignment = assignments[i];
      auto domainIndex = domainById.at(assignment.domainType);
      domainAnchors[domainIndex].push_back(valueIndex);
      if (!assignment.inferred)
        domainAssociations[domainIndex].push_back(
            {valueIndex, none, none, assignment.domainValue});
    }
  }
  for (uint32_t moduleIndex = 0; moduleIndex < modules.size(); ++moduleIndex) {
    const auto &module = modules[moduleIndex];
    for (uint32_t portIndex = 0; portIndex < module.ports.size(); ++portIndex) {
      const auto &port = module.ports[portIndex];
      auto valueIt = valueById.find(port.value);
      for (const auto &assignment : port.assignments) {
        auto domainIndex = domainById.at(assignment.domainType);
        if (valueIt != valueById.end())
          domainAnchors[domainIndex].push_back(valueIt->second);
        if (assignment.inferred)
          continue;
        bool duplicated = false;
        if (valueIt != valueById.end()) {
          const auto &value = values[valueIt->second];
          for (uint32_t i = value.assignmentBegin;
               i < value.assignmentBegin + value.assignmentCount; ++i)
            if (!assignments[i].inferred &&
                assignments[i].domainType == assignment.domainType &&
                assignments[i].domainValue == assignment.domainPortValue) {
              duplicated = true;
              break;
            }
        }
        if (!duplicated)
          domainAssociations[domainIndex].push_back(
              {valueIt == valueById.end() ? none : valueIt->second, moduleIndex,
               portIndex, assignment.domainPortValue,
               assignment.domainPortIndex});
      }
    }
  }
  for (uint32_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    const auto &edge = edges[edgeIndex];
    if (edge.lhs != none && edge.rhs != none && edge.lhs != edge.rhs)
      domainEdges[edge.domain].push_back(edgeIndex);
  }
  // Constraint edges can be recorded for a domain type even when neither
  // endpoint receives that domain. Keep only components reached from an
  // assignment or a value of the domain type.
  std::vector<uint32_t> reached(values.size(), 0);
  std::vector<uint32_t> componentIndex(values.size(), none);
  for (uint32_t domainIndex = 0; domainIndex < domains.size(); ++domainIndex) {
    uint32_t stamp = domainIndex + 1;
    auto &nodes = domainNodes[domainIndex];
    auto &components = domainComponents[domainIndex];
    for (uint32_t anchor : domainAnchors[domainIndex]) {
      if (reached[anchor] == stamp)
        continue;
      uint32_t index = components.size();
      auto &component = components.emplace_back();
      reached[anchor] = stamp;
      componentIndex[anchor] = index;
      component.nodes.push_back(anchor);
      for (size_t front = 0; front < component.nodes.size(); ++front) {
        uint32_t current = component.nodes[front];
        for (uint32_t i = adjacencyOffsets[current];
             i < adjacencyOffsets[current + 1]; ++i) {
          const auto &edge = edges[adjacencyEdges[i]];
          if (edge.domain != domainIndex || edge.lhs == none ||
              edge.rhs == none || edge.lhs == edge.rhs)
            continue;
          uint32_t neighbor = edge.lhs == current ? edge.rhs : edge.lhs;
          if (reached[neighbor] == stamp)
            continue;
          reached[neighbor] = stamp;
          componentIndex[neighbor] = index;
          component.nodes.push_back(neighbor);
        }
      }
      nodes.insert(nodes.end(), component.nodes.begin(), component.nodes.end());
    }
    std::sort(nodes.begin(), nodes.end());
    auto &connections = domainEdges[domainIndex];
    size_t write = 0;
    for (uint32_t edgeIndex : connections) {
      const auto &edge = edges[edgeIndex];
      if (reached[edge.lhs] != stamp)
        continue;
      connections[write++] = edgeIndex;
      components[componentIndex[edge.lhs]].edges.push_back(edgeIndex);
    }
    connections.resize(write);
    std::stable_sort(components.begin(), components.end(),
                     [](const DomainComponent &a, const DomainComponent &b) {
                       return a.nodes.size() > b.nodes.size();
                     });
    for (uint32_t index = 0; index < components.size(); ++index)
      for (uint32_t node : components[index].nodes)
        componentIndex[node] = index;

    for (uint32_t index = 0; index < domainAssociations[domainIndex].size();
         ++index) {
      const auto &association = domainAssociations[domainIndex][index];
      uint32_t value = association.value;
      if (value == none && association.domainValue >= 0) {
        auto found = valueById.find(association.domainValue);
        if (found != valueById.end())
          value = found->second;
      }
      if (value != none && reached[value] == stamp)
        components[componentIndex[value]].associations.push_back(index);
      else
        unmappedAssociations[domainIndex].push_back(index);
    }

    for (uint32_t crossingIndex : crossingsByDomain[domainIndex]) {
      const auto &crossing = crossings[crossingIndex];
      uint32_t endpoints[] = {crossing.lhs,       crossing.rhs,
                              crossing.lhsDomain, crossing.rhsDomain,
                              crossing.lhsSource, crossing.rhsSource};
      std::vector<uint32_t> involved;
      for (uint32_t value : endpoints) {
        if (value == none || reached[value] != stamp)
          continue;
        uint32_t index = componentIndex[value];
        if (std::find(involved.begin(), involved.end(), index) !=
            involved.end())
          continue;
        involved.push_back(index);
        components[index].crossings.push_back(crossingIndex);
      }
      if (involved.empty())
        unmappedCrossings[domainIndex].push_back(crossingIndex);
    }
  }
  return true;
}

void Report::setSourceRoots(const std::vector<std::string> &roots) {
  sourceRoots.clear();
  for (const auto &root : roots)
    sourceRoots.push_back(normalized(root));
  sourceIndex.clear();
  std::unordered_set<std::string> missingFiles;
  for (uint32_t index = 0; index < locations.size(); ++index) {
    const auto &sources = locations[index].sources;
    for (uint32_t sourceNumber = 0; sourceNumber < sources.size();
         ++sourceNumber) {
      const auto &source = sources[sourceNumber];
      std::filesystem::path file(source.file);
      if (file.is_absolute()) {
        sourceIndex[normalized(source.file)].push_back({index, sourceNumber});
        continue;
      }
      for (const auto &root : sourceRoots) {
        auto candidate =
            normalized((std::filesystem::path(root) / file).string());
        if (!sourceIndex.count(candidate)) {
          if (missingFiles.count(candidate))
            continue;
          std::error_code error;
          if (!std::filesystem::is_regular_file(candidate, error)) {
            missingFiles.insert(candidate);
            continue;
          }
        }
        sourceIndex[candidate].push_back({index, sourceNumber});
      }
    }
  }
}

Object Report::summary() const {
  return Object{{"format", "circt-domain-inference"},
                {"version", version},
                {"complete", complete},
                {"domains", static_cast<int64_t>(domains.size())},
                {"modules", static_cast<int64_t>(modules.size())},
                {"values", static_cast<int64_t>(values.size())},
                {"instances", static_cast<int64_t>(instances.size())},
                {"provenanceEdges", static_cast<int64_t>(edges.size())},
                {"illegalCrossings", static_cast<int64_t>(crossings.size())},
                {"hasIllegalCrossings", hasIllegalCrossings},
                {"skippedEdges", static_cast<int64_t>(skippedEdges)}};
}

Object Report::locationJSON(uint32_t id) const {
  if (id >= locations.size())
    return Object{{"display", "<unknown>"}, {"sources", Array()}};
  const auto &location = locations[id];
  Array sources;
  for (const auto &source : location.sources) {
    Array paths;
    std::filesystem::path file(source.file);
    if (file.is_absolute()) {
      std::error_code error;
      if (std::filesystem::is_regular_file(file, error))
        paths.push_back(normalized(source.file));
    } else {
      for (const auto &root : sourceRoots) {
        auto candidate =
            normalized((std::filesystem::path(root) / file).string());
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error))
          paths.push_back(std::move(candidate));
      }
    }
    Object item{{"file", source.file},
                {"line", static_cast<int64_t>(source.line)},
                {"column", static_cast<int64_t>(source.column)},
                {"paths", std::move(paths)}};
    if (source.range) {
      item["endLine"] = static_cast<int64_t>(source.endLine);
      item["endColumn"] = static_cast<int64_t>(source.endColumn);
    }
    sources.push_back(std::move(item));
  }
  return Object{{"display", location.display}, {"sources", std::move(sources)}};
}

Object Report::valueJSON(uint32_t index, bool details) const {
  const auto &value = values[index];
  Object result{{"id", std::to_string(value.id)},
                {"name", value.name},
                {"kind", value.kind},
                {"moduleId", std::to_string(modules[value.module].id)},
                {"module", modules[value.module].name},
                {"type", value.type < types.size() ? types[value.type] : ""},
                {"location", locationJSON(value.location)}};
  if (!details)
    return result;
  result["definition"] = value.definition;
  result["direction"] = value.direction;
  if (value.portIndex >= 0)
    result["portIndex"] = value.portIndex;
  if (value.instancePortIndex >= 0)
    result["instancePortIndex"] = value.instancePortIndex;
  if (value.instance >= 0)
    result["instanceId"] = std::to_string(value.instance);
  if (value.domainType >= 0)
    result["domainTypeId"] = std::to_string(value.domainType);
  Array items;
  for (uint32_t i = value.assignmentBegin;
       i < value.assignmentBegin + value.assignmentCount; ++i) {
    const auto &assignment = assignments[i];
    auto domain = domainById.find(assignment.domainType);
    Object item{{"domainTypeId", std::to_string(assignment.domainType)},
                {"domain", domain == domainById.end()
                               ? "<unknown>"
                               : domains[domain->second].name},
                {"inferred", assignment.inferred}};
    if (assignment.domainValue >= 0) {
      item["domainValueId"] = std::to_string(assignment.domainValue);
      auto domainValue = valueById.find(assignment.domainValue);
      if (domainValue != valueById.end())
        item["domainValue"] = values[domainValue->second].name;
    }
    items.push_back(std::move(item));
  }
  result["assignments"] = std::move(items);
  return result;
}

Object Report::instanceJSON(uint32_t index, bool details) const {
  const auto &instance = instances[index];
  Array targets;
  for (const auto &target : instance.targets) {
    int64_t id;
    auto parsed =
        std::from_chars(target.data(), target.data() + target.size(), id);
    auto module =
        parsed.ec == std::errc() && parsed.ptr == target.data() + target.size()
            ? moduleById.find(id)
            : moduleById.end();
    targets.push_back(
        module == moduleById.end() ? target : modules[module->second].name);
  }
  Object result{{"id", std::to_string(instance.id)},
                {"name", instance.name},
                {"moduleId", std::to_string(modules[instance.module].id)},
                {"targets", std::move(targets)},
                {"location", locationJSON(instance.location)}};
  if (!details)
    return result;
  Array bindings;
  for (const auto &binding : instance.bindings) {
    auto target = moduleById.find(binding.targetModule);
    auto domain = domainById.find(binding.domainType);
    Object item{{"targetModuleId", std::to_string(binding.targetModule)},
                {"targetModule", target == moduleById.end()
                                     ? "<unknown>"
                                     : modules[target->second].name},
                {"portIndex", binding.portIndex},
                {"domainTypeId", std::to_string(binding.domainType)},
                {"domain", domain == domainById.end()
                               ? "<unknown>"
                               : domains[domain->second].name},
                {"domainPortIndex", binding.domainPortIndex},
                {"effectiveDomainValueName", binding.effectiveDomainValueName},
                {"location", locationJSON(binding.location)}};
    if (binding.portValue >= 0)
      item["portValueId"] = std::to_string(binding.portValue);
    if (binding.effectiveDomainValue >= 0)
      item["effectiveDomainValueId"] =
          std::to_string(binding.effectiveDomainValue);
    bindings.push_back(std::move(item));
  }
  result["bindings"] = std::move(bindings);
  return result;
}

Object Report::edgeJSON(uint32_t index) const {
  const auto &edge = edges[index];
  Object result{{"index", static_cast<int64_t>(index)},
                {"kind", edgeKinds[edge.kind]},
                {"domainTypeId", std::to_string(domains[edge.domain].id)},
                {"domain", domains[edge.domain].name},
                {"inferred", (edge.flags & inferredFlag) != 0},
                {"summarized", (edge.flags & summarizedFlag) != 0},
                {"ownerModule", modules[edge.owner].name},
                {"location", locationJSON(edge.location)}};
  if (edge.lhs != none) {
    result["lhsId"] = std::to_string(values[edge.lhs].id);
    result["lhs"] = values[edge.lhs].name;
    result["lhsModule"] = modules[values[edge.lhs].module].name;
  }
  if (edge.rhs != none) {
    result["rhsId"] = std::to_string(values[edge.rhs].id);
    result["rhs"] = values[edge.rhs].name;
    result["rhsModule"] = modules[values[edge.rhs].module].name;
  }
  if (edge.operation != none)
    result["operation"] = operationKinds[edge.operation];
  if (edge.instance != none) {
    result["instanceId"] = std::to_string(instances[edge.instance].id);
    result["instance"] = instances[edge.instance].name;
  }
  if (edge.target != none)
    result["targetModule"] = modules[edge.target].name;
  return result;
}

Object Report::crossingJSON(uint32_t index) const {
  const auto &crossing = crossings[index];
  Object result{{"index", static_cast<int64_t>(index)},
                {"domainTypeId", std::to_string(domains[crossing.domain].id)},
                {"ownerModule", modules[crossing.owner].name},
                {"operation", operationKinds[crossing.operation]},
                {"location", locationJSON(crossing.location)}};
  auto addValue = [&](llvm::StringRef prefix, uint32_t valueIndex) {
    if (valueIndex == none)
      return;
    const auto &value = values[valueIndex];
    result[prefix.str() + "Id"] = std::to_string(value.id);
    result[prefix.str()] = value.name;
    result[prefix.str() + "Module"] = modules[value.module].name;
  };
  addValue("lhsValue", crossing.lhs);
  addValue("rhsValue", crossing.rhs);
  addValue("lhsDomain", crossing.lhsDomain);
  addValue("rhsDomain", crossing.rhsDomain);
  addValue("lhsSource", crossing.lhsSource);
  addValue("rhsSource", crossing.rhsSource);
  return result;
}

static uint32_t pageOffset(const Object &params) {
  auto value = params.getInteger("offset").value_or(0);
  return value < 0 ? 0
                   : static_cast<uint32_t>(std::min<int64_t>(value, INT32_MAX));
}

static uint32_t pageLimit(const Object &params) {
  auto value = params.getInteger("limit").value_or(100);
  return static_cast<uint32_t>(std::clamp<int64_t>(value, 1, 200));
}

JsonValue Report::request(llvm::StringRef method, const Object &params,
                          std::string &error) const {
  if (method == "summary")
    return summary();
  if (method == "listDomains") {
    Array items;
    uint32_t offset = pageOffset(params), limit = pageLimit(params);
    for (size_t i = offset; i < domains.size() && items.size() < limit; ++i)
      items.push_back(Object{
          {"id", std::to_string(domains[i].id)},
          {"name", domains[i].name},
          {"location", locationJSON(domains[i].location)},
          {"nodes", static_cast<int64_t>(domainNodes[i].size())},
          {"connections", static_cast<int64_t>(domainEdges[i].size())},
          {"explicitAssociations",
           static_cast<int64_t>(domainAssociations[i].size())},
          {"components", static_cast<int64_t>(domainComponents[i].size())},
          {"illegalCrossings",
           static_cast<int64_t>(crossingsByDomain[i].size())}});
    return Object{{"items", std::move(items)},
                  {"total", static_cast<int64_t>(domains.size())}};
  }
  if (method == "listDomainComponents") {
    auto domainId = parseId(params, "domainTypeId");
    auto found = domainId ? domainById.find(*domainId) : domainById.end();
    if (found == domainById.end()) {
      error = "unknown domain type ID";
      return nullptr;
    }
    uint32_t domainIndex = found->second;
    const auto &components = domainComponents[domainIndex];
    bool hasUnmapped = !unmappedAssociations[domainIndex].empty() ||
                       !unmappedCrossings[domainIndex].empty();
    size_t total = components.size() + hasUnmapped;
    uint32_t offset = pageOffset(params), limit = pageLimit(params);
    Array items;
    for (size_t i = offset; i < total && items.size() < limit; ++i) {
      if (i == components.size()) {
        items.push_back(Object{
            {"index", -1},
            {"unmapped", true},
            {"nodes", 0},
            {"connections", 0},
            {"explicitAssociations",
             static_cast<int64_t>(unmappedAssociations[domainIndex].size())},
            {"illegalCrossings",
             static_cast<int64_t>(unmappedCrossings[domainIndex].size())}});
        continue;
      }
      const auto &component = components[i];
      const auto &representative = values[component.nodes.front()];
      items.push_back(
          Object{{"index", static_cast<int64_t>(i)},
                 {"representativeValueId", std::to_string(representative.id)},
                 {"name", representative.name},
                 {"module", modules[representative.module].name},
                 {"nodes", static_cast<int64_t>(component.nodes.size())},
                 {"connections", static_cast<int64_t>(component.edges.size())},
                 {"explicitAssociations",
                  static_cast<int64_t>(component.associations.size())},
                 {"illegalCrossings",
                  static_cast<int64_t>(component.crossings.size())}});
    }
    return Object{{"items", std::move(items)},
                  {"total", static_cast<int64_t>(total)}};
  }
  if (method == "componentForValue") {
    auto domainId = parseId(params, "domainTypeId");
    auto valueId = parseId(params, "valueId");
    auto domain = domainId ? domainById.find(*domainId) : domainById.end();
    auto value = valueId ? valueById.find(*valueId) : valueById.end();
    if (domain == domainById.end() || value == valueById.end()) {
      error = "unknown domain type or value ID";
      return nullptr;
    }
    const auto &components = domainComponents[domain->second];
    for (size_t i = 0; i < components.size(); ++i)
      if (std::find(components[i].nodes.begin(), components[i].nodes.end(),
                    value->second) != components[i].nodes.end())
        return Object{{"index", static_cast<int64_t>(i)}};
    error = "value is not in this domain graph";
    return nullptr;
  }
  if (method == "listIllegalCrossings") {
    auto domainId = parseId(params, "domainTypeId");
    auto found = domainId ? domainById.find(*domainId) : domainById.end();
    if (found == domainById.end()) {
      error = "unknown domain type ID";
      return nullptr;
    }
    uint32_t domainIndex = found->second;
    auto componentNumber = params.getInteger("componentIndex");
    bool unmapped = params.getBoolean("unmapped").value_or(false);
    if ((componentNumber &&
         (*componentNumber < 0 || static_cast<uint64_t>(*componentNumber) >=
                                      domainComponents[domainIndex].size())) ||
        (componentNumber && unmapped)) {
      error = "unknown connected component";
      return nullptr;
    }
    const auto &itemsForDomain =
        unmapped ? unmappedCrossings[domainIndex]
        : componentNumber
            ? domainComponents[domainIndex][*componentNumber].crossings
            : crossingsByDomain[domainIndex];
    uint32_t offset = pageOffset(params), limit = pageLimit(params);
    Array items;
    for (size_t i = offset; i < itemsForDomain.size() && items.size() < limit;
         ++i)
      items.push_back(crossingJSON(itemsForDomain[i]));
    return Object{{"items", std::move(items)},
                  {"total", static_cast<int64_t>(itemsForDomain.size())}};
  }
  if (method == "listModules" || method == "searchValues") {
    Array items;
    auto query = params.getString("query").value_or("");
    uint32_t offset = pageOffset(params), limit = pageLimit(params);
    uint64_t total = 0;
    if (method == "listModules") {
      for (const auto &module : modules) {
        if (!containsInsensitive(module.name, query))
          continue;
        if (total++ >= offset && items.size() < limit)
          items.push_back(Object{
              {"id", std::to_string(module.id)},
              {"name", module.name},
              {"kind", module.kind},
              {"ports", static_cast<int64_t>(module.ports.size())},
              {"values", static_cast<int64_t>(module.values.size())},
              {"instances", static_cast<int64_t>(module.instances.size())}});
      }
    } else {
      if (query.empty()) {
        error = "search query must not be empty";
        return nullptr;
      }
      for (uint32_t i = 0; i < values.size(); ++i) {
        if (!containsInsensitive(values[i].name, query))
          continue;
        if (total++ >= offset && items.size() < limit)
          items.push_back(valueJSON(i, true));
      }
    }
    return Object{{"items", std::move(items)},
                  {"total", static_cast<int64_t>(total)}};
  }
  if (method == "moduleItems") {
    auto moduleId = parseId(params, "moduleId");
    auto found = moduleId ? moduleById.find(*moduleId) : moduleById.end();
    auto category = params.getString("category").value_or("");
    if (found == moduleById.end()) {
      error = "unknown module ID";
      return nullptr;
    }
    const auto &module = modules[found->second];
    uint32_t offset = pageOffset(params), limit = pageLimit(params);
    Array items;
    size_t total;
    if (category == "values") {
      total = module.values.size();
      for (size_t i = offset; i < total && items.size() < limit; ++i)
        items.push_back(valueJSON(module.values[i]));
    } else if (category == "instances") {
      total = module.instances.size();
      for (size_t i = offset; i < total && items.size() < limit; ++i)
        items.push_back(instanceJSON(module.instances[i]));
    } else if (category == "ports") {
      total = module.ports.size();
      for (size_t i = offset; i < total && items.size() < limit; ++i) {
        const auto &port = module.ports[i];
        auto value = valueById.find(port.value);
        Object item{
            {"index", port.index},
            {"name",
             value == valueById.end() ? port.name : values[value->second].name},
            {"direction", value == valueById.end()
                              ? port.direction
                              : values[value->second].direction},
            {"type", value == valueById.end()
                         ? (port.type < types.size() ? types[port.type] : "")
                         : types[values[value->second].type]},
            {"location", locationJSON(value == valueById.end()
                                          ? port.location
                                          : values[value->second].location)}};
        if (port.value >= 0)
          item["valueId"] = std::to_string(port.value);
        Array assignmentsJSON;
        for (const auto &assignment : port.assignments) {
          auto domain = domainById.find(assignment.domainType);
          Object entry{{"domainTypeId", std::to_string(assignment.domainType)},
                       {"domain", domain == domainById.end()
                                      ? "<unknown>"
                                      : domains[domain->second].name},
                       {"domainPortIndex", assignment.domainPortIndex},
                       {"inferred", assignment.inferred}};
          if (assignment.domainPortValue >= 0)
            entry["domainPortValueId"] =
                std::to_string(assignment.domainPortValue);
          assignmentsJSON.push_back(std::move(entry));
        }
        item["assignments"] = std::move(assignmentsJSON);
        items.push_back(std::move(item));
      }
    } else {
      error = "expected ports, values, or instances";
      return nullptr;
    }
    return Object{{"items", std::move(items)},
                  {"total", static_cast<int64_t>(total)}};
  }
  if (method == "getValue" || method == "getInstance") {
    auto id = parseId(params, "id");
    if (method == "getValue") {
      auto found = id ? valueById.find(*id) : valueById.end();
      if (found != valueById.end())
        return valueJSON(found->second, true);
    } else {
      auto found = id ? instanceById.find(*id) : instanceById.end();
      if (found != instanceById.end())
        return instanceJSON(found->second, true);
    }
    error = "unknown record ID";
    return nullptr;
  }
  if (method == "sourceMatches") {
    auto path = params.getString("path");
    auto line = params.getInteger("line");
    auto column = params.getInteger("column");
    if (!path || !line || !column || *line < 1 || *column < 1) {
      error = "expected path, line, and column";
      return nullptr;
    }
    Array items;
    uint64_t total = 0;
    uint32_t offset = pageOffset(params), limit = pageLimit(params);
    auto found = sourceIndex.find(normalized(path->str()));
    if (found != sourceIndex.end()) {
      std::unordered_set<uint32_t> seen;
      for (const auto &hit : found->second) {
        const auto &source = locations[hit.location].sources[hit.sourceIndex];
        auto position = std::pair(*line, *column);
        auto start = std::pair(static_cast<int64_t>(source.line),
                               static_cast<int64_t>(source.column));
        bool matched =
            source.range
                ? position >= start &&
                      position <
                          std::pair(static_cast<int64_t>(source.endLine),
                                    static_cast<int64_t>(source.endColumn))
                : position == start;
        if (!matched)
          continue;
        for (uint32_t value : valuesByLocation[hit.location]) {
          if (!seen.insert(value).second)
            continue;
          if (total++ >= offset && items.size() < limit)
            items.push_back(valueJSON(value, true));
        }
      }
    }
    return Object{{"items", std::move(items)},
                  {"total", static_cast<int64_t>(total)}};
  }
  if (method == "trace")
    return trace(params, error);
  if (method == "crossingPath")
    return crossingPath(params, error);
  if (method == "instanceAssociation")
    return instanceAssociation(params, error);
  if (method == "neighbors")
    return neighbors(params, error);
  if (method == "domainItems")
    return domainItems(params, error);
  if (method == "minCut")
    return minCut(params, error);
  error = "unknown request method";
  return nullptr;
}

JsonValue Report::domainItems(const Object &params, std::string &error) const {
  auto domainId = parseId(params, "domainTypeId");
  auto found = domainId ? domainById.find(*domainId) : domainById.end();
  if (found == domainById.end()) {
    error = "unknown domain type ID";
    return nullptr;
  }
  uint32_t domainIndex = found->second;
  auto componentNumber = params.getInteger("componentIndex");
  bool unmapped = params.getBoolean("unmapped").value_or(false);
  if ((componentNumber &&
       (*componentNumber < 0 || static_cast<uint64_t>(*componentNumber) >=
                                    domainComponents[domainIndex].size())) ||
      (componentNumber && unmapped)) {
    error = "unknown connected component";
    return nullptr;
  }
  const DomainComponent *component =
      componentNumber ? &domainComponents[domainIndex][*componentNumber]
                      : nullptr;
  const std::vector<uint32_t> empty;
  uint32_t offset = pageOffset(params), limit = pageLimit(params);
  auto category = params.getString("category").value_or("");
  Array items;
  uint64_t total = 0;
  if (category == "nodes") {
    auto query = params.getString("query").value_or("");
    const auto &nodes = unmapped    ? empty
                        : component ? component->nodes
                                    : domainNodes[domainIndex];
    for (uint32_t index : nodes) {
      const auto &value = values[index];
      if (!query.empty() && !containsInsensitive(value.name, query) &&
          !containsInsensitive(modules[value.module].name, query) &&
          !containsInsensitive(std::to_string(value.id), query))
        continue;
      if (total++ >= offset && items.size() < limit) {
        auto item = valueJSON(index);
        if (value.portIndex >= 0)
          item["portIndex"] = value.portIndex;
        if (!value.direction.empty())
          item["direction"] = value.direction;
        items.push_back(std::move(item));
      }
    }
  } else if (category == "connections") {
    const auto &connections = unmapped    ? empty
                              : component ? component->edges
                                          : domainEdges[domainIndex];
    total = connections.size();
    for (size_t i = offset; i < total && items.size() < limit; ++i)
      items.push_back(edgeJSON(connections[i]));
  } else if (category == "associations") {
    const auto &associations = domainAssociations[domainIndex];
    const std::vector<uint32_t> *selected =
        unmapped    ? &unmappedAssociations[domainIndex]
        : component ? &component->associations
                    : nullptr;
    total = selected ? selected->size() : associations.size();
    for (size_t i = offset; i < total && items.size() < limit; ++i) {
      const auto &association = associations[selected ? (*selected)[i] : i];
      Object item{{"domainTypeId", std::to_string(*domainId)},
                  {"domain", domains[domainIndex].name}};
      if (association.value != none) {
        const auto &value = values[association.value];
        item["valueId"] = std::to_string(value.id);
        item["name"] = value.name;
        item["module"] = modules[value.module].name;
        item["kind"] = value.kind;
        item["type"] = types[value.type];
        item["location"] = locationJSON(value.location);
        if (value.portIndex >= 0)
          item["portIndex"] = value.portIndex;
      } else {
        const auto &module = modules[association.module];
        const auto &port = module.ports[association.port];
        item["name"] = port.name;
        item["module"] = module.name;
        item["kind"] = "external port";
        item["type"] = port.type < types.size() ? types[port.type] : "";
        item["location"] = locationJSON(port.location);
        item["portIndex"] = port.index;
      }
      if (association.domainValue >= 0) {
        item["domainValueId"] = std::to_string(association.domainValue);
        auto value = valueById.find(association.domainValue);
        if (value != valueById.end())
          item["domainValue"] = values[value->second].name;
      } else if (association.module != none &&
                 association.domainPortIndex >= 0) {
        const auto &module = modules[association.module];
        auto port = std::find_if(module.ports.begin(), module.ports.end(),
                                 [&](const Port &candidate) {
                                   return candidate.index ==
                                          association.domainPortIndex;
                                 });
        if (port != module.ports.end()) {
          auto value = valueById.find(port->value);
          item["domainValue"] = value == valueById.end()
                                    ? port->name
                                    : values[value->second].name;
        }
      }
      items.push_back(std::move(item));
    }
  } else {
    error = "expected nodes, connections, or associations";
    return nullptr;
  }
  return Object{{"items", std::move(items)},
                {"total", static_cast<int64_t>(total)}};
}

JsonValue Report::trace(const Object &params, std::string &error) const {
  auto valueId = parseId(params, "valueId");
  auto domainId = parseId(params, "domainTypeId");
  auto valueIt = valueId ? valueById.find(*valueId) : valueById.end();
  auto domainIt = domainId ? domainById.find(*domainId) : domainById.end();
  if (valueIt == valueById.end() || domainIt == domainById.end()) {
    error = "unknown value or domain type ID";
    return nullptr;
  }
  uint32_t selectedInstance = none;
  if (params.getString("instanceId")) {
    auto id = parseId(params, "instanceId");
    auto found = id ? instanceById.find(*id) : instanceById.end();
    if (found == instanceById.end()) {
      error = "unknown instance context";
      return nullptr;
    }
    selectedInstance = found->second;
  }
  uint32_t start = valueIt->second;
  uint32_t target = none;
  const auto &value = values[start];
  for (uint32_t i = value.assignmentBegin;
       i < value.assignmentBegin + value.assignmentCount; ++i) {
    const auto &assignment = assignments[i];
    if (assignment.domainType == *domainId && assignment.domainValue >= 0) {
      auto found = valueById.find(assignment.domainValue);
      if (found != valueById.end())
        target = found->second;
      break;
    }
  }
  if (value.kind == "domain" && value.domainType == *domainId)
    target = start;

  struct Step {
    uint32_t edge;
    uint32_t from;
    uint32_t to;
  };
  struct SearchResult {
    bool found = false;
    bool contextRequired = false;
    uint32_t endpoint = Report::none;
    std::vector<Step> steps;
  };
  auto search = [&]() {
    SearchResult result;
    std::vector<uint8_t> seen(values.size(), 0);
    std::vector<uint32_t> previousValue(values.size(), none);
    std::vector<uint32_t> previousEdge(values.size(), none);
    std::vector<uint32_t> queue;
    queue.push_back(start);
    seen[start] = 1;
    uint32_t goal = none, boundary = none, extraEdge = none, extraOther = none;
    for (size_t front = 0; front < queue.size() && goal == none; ++front) {
      uint32_t current = queue[front];
      if (current == target) {
        goal = current;
        break;
      }
      for (uint32_t i = adjacencyOffsets[current];
           i < adjacencyOffsets[current + 1]; ++i) {
        uint32_t edgeIndex = adjacencyEdges[i];
        const auto &edge = edges[edgeIndex];
        if (edge.domain != domainIt->second)
          continue;
        bool binding = edgeKinds[edge.kind] == "instance_binding";
        if (binding && edge.instance != selectedInstance) {
          result.contextRequired = true;
          if (boundary == none)
            boundary = current;
          continue;
        }
        uint32_t other = edge.lhs == current ? edge.rhs : edge.lhs;
        if (other == none)
          continue;
        bool explicitAssociation = edgeKinds[edge.kind] == "association" &&
                                   !(edge.flags & inferredFlag);
        if (explicitAssociation) {
          goal = current;
          extraEdge = edgeIndex;
          extraOther = other;
          break;
        }
        if (seen[other])
          continue;
        seen[other] = 1;
        previousValue[other] = current;
        previousEdge[other] = edgeIndex;
        if (other == target) {
          goal = other;
          break;
        }
        queue.push_back(other);
      }
    }
    if (goal == none && boundary == none)
      return result;
    result.found = goal != none;
    result.endpoint = goal == none         ? boundary
                      : extraOther == none ? goal
                                           : extraOther;
    for (uint32_t current = goal == none ? boundary : goal; current != start;
         current = previousValue[current]) {
      uint32_t previous = previousValue[current];
      if (previous == none) {
        result.found = false;
        return result;
      }
      result.steps.push_back({previousEdge[current], previous, current});
    }
    std::reverse(result.steps.begin(), result.steps.end());
    if (extraEdge != none)
      result.steps.push_back({extraEdge, goal, extraOther});
    return result;
  };

  auto result = search();
  Array steps;
  bool usesSummarizedEdges = false;
  for (const auto &step : result.steps) {
    auto item = edgeJSON(step.edge);
    usesSummarizedEdges |= (edges[step.edge].flags & summarizedFlag) != 0;
    item["fromId"] = std::to_string(values[step.from].id);
    item["toId"] = std::to_string(values[step.to].id);
    steps.push_back(std::move(item));
  }
  Object response{{"found", result.found},
                  {"complete", complete},
                  {"contextRequired", result.contextRequired},
                  {"usesSummarizedEdges", usesSummarizedEdges},
                  {"steps", std::move(steps)}};
  if (result.found)
    response["endpointValueId"] = std::to_string(values[result.endpoint].id);
  else {
    if (result.endpoint != none)
      response["boundaryValueId"] = std::to_string(values[result.endpoint].id);
    response["message"] = complete
                              ? "No path was recorded in this instance context"
                              : "No path was recorded in this partial report";
  }
  return response;
}

JsonValue Report::instanceAssociation(const Object &params,
                                      std::string &error) const {
  auto index = params.getInteger("edgeIndex");
  if (!index || *index < 0 || *index >= static_cast<int64_t>(edges.size())) {
    error = "unknown provenance edge index";
    return nullptr;
  }
  const auto &edge = edges[*index];
  if (edge.kind >= edgeKinds.size() || edgeKinds[edge.kind] != "association" ||
      edge.instance == none || edge.lhs == none || edge.rhs == none) {
    error = "edge is not an instance association";
    return nullptr;
  }

  const auto &instance = instances[edge.instance];
  const auto &portValue = values[edge.lhs];
  const auto &domainValue = values[edge.rhs];
  Array targets;
  for (const auto &binding : instance.bindings) {
    if (binding.portValue != portValue.id ||
        binding.effectiveDomainValue != domainValue.id ||
        binding.domainType != domains[edge.domain].id)
      continue;
    auto targetIt = moduleById.find(binding.targetModule);
    if (targetIt == moduleById.end())
      continue;
    const auto &module = modules[targetIt->second];
    auto findPort = [&](int32_t portIndex) {
      return std::find_if(
          module.ports.begin(), module.ports.end(),
          [&](const Port &port) { return port.index == portIndex; });
    };
    auto port = findPort(binding.portIndex);
    auto domainPort = findPort(binding.domainPortIndex);
    if (port == module.ports.end() || domainPort == module.ports.end())
      continue;
    auto valueForPort = [&](const Port &port) -> const Value * {
      auto valueIt = valueById.find(port.value);
      return valueIt == valueById.end() ||
                     values[valueIt->second].module != targetIt->second
                 ? nullptr
                 : &values[valueIt->second];
    };
    const Value *targetValue = valueForPort(*port);
    const Value *targetDomain = valueForPort(*domainPort);
    auto portName = [&](const Port &port, const Value *value) {
      if (value)
        return value->name;
      return port.name.empty() ? ("port#" + std::to_string(port.index))
                               : port.name;
    };
    auto portLocation = [&](const Port &port, const Value *value) {
      return locationJSON(value ? value->location : port.location);
    };
    Object item{
        {"moduleId", std::to_string(module.id)},
        {"module", module.name},
        {"port", portName(*port, targetValue)},
        {"portIndex", binding.portIndex},
        {"domainPort", portName(*domainPort, targetDomain)},
        {"domainPortIndex", binding.domainPortIndex},
        {"location", portLocation(*port, targetValue)},
        {"domainPortLocation", portLocation(*domainPort, targetDomain)}};
    if (targetValue)
      item["portValueId"] = std::to_string(targetValue->id);
    if (targetDomain)
      item["domainPortValueId"] = std::to_string(targetDomain->id);
    auto assignment = std::find_if(
        port->assignments.begin(), port->assignments.end(),
        [&](const PortAssignment &candidate) {
          return candidate.domainType == binding.domainType &&
                 candidate.domainPortIndex == binding.domainPortIndex;
        });
    if (assignment != port->assignments.end())
      item["inferred"] = assignment->inferred;
    targets.push_back(std::move(item));
  }
  return Object{{"instanceId", std::to_string(instance.id)},
                {"instance", instance.name},
                {"ownerModule", modules[instance.module].name},
                {"targets", std::move(targets)}};
}

JsonValue Report::crossingPath(const Object &params, std::string &error) const {
  auto index = params.getInteger("crossingIndex");
  if (!index || *index < 0 ||
      *index >= static_cast<int64_t>(crossings.size())) {
    error = "unknown illegal crossing index";
    return nullptr;
  }
  const auto &crossing = crossings[*index];
  Object response{{"found", false}, {"complete", complete}};
  if (crossing.lhs == none || crossing.rhs == none ||
      crossing.lhsSource == none || crossing.rhsSource == none) {
    response["message"] = "A crossing endpoint is missing from this report";
    return response;
  }

  struct Step {
    uint32_t edge;
    uint32_t from;
    uint32_t to;
  };
  auto findPath = [&](uint32_t start, uint32_t goal, std::vector<Step> &path) {
    std::vector<uint32_t> previousValue(values.size(), none);
    std::vector<uint32_t> previousEdge(values.size(), none);
    std::vector<uint32_t> queue{start};
    previousValue[start] = start;
    for (size_t front = 0; front < queue.size() && previousValue[goal] == none;
         ++front) {
      uint32_t current = queue[front];
      for (uint32_t i = adjacencyOffsets[current];
           i < adjacencyOffsets[current + 1]; ++i) {
        uint32_t edgeIndex = adjacencyEdges[i];
        const auto &edge = edges[edgeIndex];
        if (edge.domain != crossing.domain || edge.lhs == none ||
            edge.rhs == none)
          continue;
        uint32_t other = edge.lhs == current ? edge.rhs : edge.lhs;
        if (previousValue[other] != none)
          continue;
        previousValue[other] = current;
        previousEdge[other] = edgeIndex;
        queue.push_back(other);
      }
    }
    if (previousValue[goal] == none)
      return false;
    for (uint32_t current = goal; current != start;
         current = previousValue[current])
      path.push_back({previousEdge[current], previousValue[current], current});
    std::reverse(path.begin(), path.end());
    return true;
  };

  std::vector<Step> lhsPath, rhsPath;
  if (!findPath(crossing.lhsSource, crossing.lhs, lhsPath) ||
      !findPath(crossing.rhs, crossing.rhsSource, rhsPath)) {
    response["message"] =
        complete
            ? "No recorded path joins an annotation to the failed connection"
            : "No recorded path joins an annotation to the failed connection "
              "in this partial report";
    return response;
  }

  Array steps;
  bool usesSummarizedEdges = false;
  auto addSteps = [&](const std::vector<Step> &path) {
    for (const auto &step : path) {
      auto item = edgeJSON(step.edge);
      item["fromId"] = std::to_string(values[step.from].id);
      item["toId"] = std::to_string(values[step.to].id);
      usesSummarizedEdges |= (edges[step.edge].flags & summarizedFlag) != 0;
      steps.push_back(std::move(item));
    }
  };
  addSteps(lhsPath);
  steps.push_back(
      Object{{"kind", "failed_constraint"},
             {"domainTypeId", std::to_string(domains[crossing.domain].id)},
             {"domain", domains[crossing.domain].name},
             {"ownerModule", modules[crossing.owner].name},
             {"operation", operationKinds[crossing.operation]},
             {"lhsId", std::to_string(values[crossing.lhs].id)},
             {"rhsId", std::to_string(values[crossing.rhs].id)},
             {"lhs", values[crossing.lhs].name},
             {"rhs", values[crossing.rhs].name},
             {"lhsModule", modules[values[crossing.lhs].module].name},
             {"rhsModule", modules[values[crossing.rhs].module].name},
             {"fromId", std::to_string(values[crossing.lhs].id)},
             {"toId", std::to_string(values[crossing.rhs].id)},
             {"inferred", false},
             {"summarized", false},
             {"location", locationJSON(crossing.location)}});
  addSteps(rhsPath);
  response["found"] = true;
  response["sourceValueId"] = std::to_string(values[crossing.lhsSource].id);
  response["targetValueId"] = std::to_string(values[crossing.rhsSource].id);
  response["usesSummarizedEdges"] = usesSummarizedEdges;
  response["steps"] = std::move(steps);
  return response;
}

JsonValue Report::neighbors(const Object &params, std::string &error) const {
  auto valueId = parseId(params, "valueId");
  auto domainId = parseId(params, "domainTypeId");
  auto valueIt = valueId ? valueById.find(*valueId) : valueById.end();
  auto domainIt = domainId ? domainById.find(*domainId) : domainById.end();
  if (valueIt == valueById.end() || domainIt == domainById.end()) {
    error = "unknown value or domain type ID";
    return nullptr;
  }
  std::vector<uint32_t> kinds;
  bool filterKinds = false;
  if (const auto *requested = params.get("kinds")) {
    const auto *items = requested->getAsArray();
    if (!items) {
      error = "expected provenance edge kinds array";
      return nullptr;
    }
    filterKinds = true;
    for (const auto &item : *items) {
      auto name = item.getAsString();
      auto found = name ? std::find_if(edgeKinds.begin(), edgeKinds.end(),
                                       [&](const std::string &kind) {
                                         return llvm::StringRef(kind) == *name;
                                       })
                        : edgeKinds.end();
      if (found == edgeKinds.end()) {
        error = "unknown provenance edge kind";
        return nullptr;
      }
      kinds.push_back(found - edgeKinds.begin());
    }
  }
  uint32_t current = valueIt->second;
  uint32_t offset = pageOffset(params), limit = pageLimit(params);
  uint64_t total = 0;
  Array items;
  for (uint32_t i = adjacencyOffsets[current];
       i < adjacencyOffsets[current + 1]; ++i) {
    uint32_t edgeIndex = adjacencyEdges[i];
    const auto &edge = edges[edgeIndex];
    if (edge.domain != domainIt->second ||
        (filterKinds &&
         std::find(kinds.begin(), kinds.end(), edge.kind) == kinds.end()))
      continue;
    uint32_t other = edge.lhs == current ? edge.rhs : edge.lhs;
    if (other == none)
      continue;
    if (total++ >= offset && items.size() < limit) {
      auto item = edgeJSON(edgeIndex);
      item["otherValueId"] = std::to_string(values[other].id);
      item["otherValue"] = values[other].name;
      items.push_back(std::move(item));
    }
  }
  return Object{{"items", std::move(items)},
                {"total", static_cast<int64_t>(total)}};
}

namespace {

// A report edge has unit weight, even when it summarizes a larger relation.
// Parallel provenance rows remain separate edges in this multigraph.
struct CutGraph {
  struct Edge {
    uint32_t lhs;
    uint32_t rhs;
    uint32_t reportIndex;
  };
  struct Result {
    std::vector<uint8_t> sourceSide;
    std::vector<uint32_t> edgeIndices;
  };

  explicit CutGraph(size_t nodeCount) : adjacency(nodeCount) {}

  void addEdge(uint32_t lhs, uint32_t rhs, uint32_t reportIndex) {
    uint32_t index = edges.size();
    edges.push_back({lhs, rhs, reportIndex});
    adjacency[lhs].push_back(index);
    adjacency[rhs].push_back(index);
  }

  uint32_t other(uint32_t node, uint32_t edgeIndex) const {
    const auto &edge = edges[edgeIndex];
    return edge.lhs == node ? edge.rhs : edge.lhs;
  }

  std::vector<uint8_t> reachable(uint32_t start,
                                 uint32_t blockedEdge = Report::none) const {
    std::vector<uint8_t> seen(adjacency.size(), 0);
    std::vector<uint32_t> queue{start};
    seen[start] = 1;
    for (size_t front = 0; front < queue.size(); ++front) {
      uint32_t node = queue[front];
      for (uint32_t edgeIndex : adjacency[node]) {
        if (edgeIndex == blockedEdge)
          continue;
        uint32_t neighbor = other(node, edgeIndex);
        if (!seen[neighbor]) {
          seen[neighbor] = 1;
          queue.push_back(neighbor);
        }
      }
    }
    return seen;
  }

  Result result(std::vector<uint8_t> side) const {
    Result result{std::move(side), {}};
    for (const auto &edge : edges)
      if (result.sourceSide[edge.lhs] != result.sourceSide[edge.rhs])
        result.edgeIndices.push_back(edge.reportIndex);
    return result;
  }

  uint32_t findBridge() const {
    const uint32_t none = Report::none;
    std::vector<uint32_t> discovery(adjacency.size(), none);
    std::vector<uint32_t> low(adjacency.size(), none);
    std::vector<uint32_t> parent(adjacency.size(), none);
    std::vector<uint32_t> parentEdge(adjacency.size(), none);
    struct Frame {
      uint32_t node;
      size_t next = 0;
    };
    std::vector<Frame> stack{{0}};
    discovery[0] = low[0] = 0;
    uint32_t time = 1;
    while (!stack.empty()) {
      auto &frame = stack.back();
      uint32_t node = frame.node;
      if (frame.next < adjacency[node].size()) {
        uint32_t edgeIndex = adjacency[node][frame.next++];
        if (edgeIndex == parentEdge[node])
          continue;
        uint32_t neighbor = other(node, edgeIndex);
        if (discovery[neighbor] == none) {
          parent[neighbor] = node;
          parentEdge[neighbor] = edgeIndex;
          discovery[neighbor] = low[neighbor] = time++;
          stack.push_back({neighbor});
        } else {
          low[node] = std::min(low[node], discovery[neighbor]);
        }
      } else {
        stack.pop_back();
        if (parent[node] != none) {
          if (low[node] > discovery[parent[node]])
            return parentEdge[node];
          low[parent[node]] = std::min(low[parent[node]], low[node]);
        }
      }
    }
    return none;
  }

  Result minimumSTCut(uint32_t source, uint32_t target) const {
    struct Arc {
      uint32_t to;
      uint32_t reverse;
      uint32_t capacity;
    };
    std::vector<std::vector<Arc>> residual(adjacency.size());
    for (const auto &edge : edges) {
      uint32_t lhsReverse = residual[edge.rhs].size();
      uint32_t rhsReverse = residual[edge.lhs].size();
      residual[edge.lhs].push_back({edge.rhs, lhsReverse, 1});
      residual[edge.rhs].push_back({edge.lhs, rhsReverse, 1});
    }
    // Dinic's level graph finds many unit-capacity paths per graph walk. Keep
    // the blocking-flow walk iterative so long chains do not exhaust the stack.
    std::vector<int32_t> level(adjacency.size(), -1);
    while (true) {
      std::fill(level.begin(), level.end(), -1);
      std::vector<uint32_t> queue{source};
      level[source] = 0;
      for (size_t front = 0; front < queue.size(); ++front) {
        uint32_t node = queue[front];
        for (const auto &arc : residual[node]) {
          if (!arc.capacity || level[arc.to] >= 0)
            continue;
          level[arc.to] = level[node] + 1;
          queue.push_back(arc.to);
        }
      }
      if (level[target] < 0) {
        std::vector<uint8_t> side(adjacency.size(), 0);
        for (uint32_t node : queue)
          side[node] = 1;
        return result(std::move(side));
      }
      std::vector<uint32_t> nextArc(adjacency.size(), 0);
      std::vector<uint32_t> path{source};
      std::vector<uint32_t> pathArcs;
      while (!path.empty()) {
        uint32_t node = path.back();
        if (node == target) {
          uint32_t amount = UINT32_MAX;
          for (size_t i = 0; i < pathArcs.size(); ++i)
            amount = std::min(amount, residual[path[i]][pathArcs[i]].capacity);
          for (size_t i = 0; i < pathArcs.size(); ++i) {
            auto &arc = residual[path[i]][pathArcs[i]];
            arc.capacity -= amount;
            residual[arc.to][arc.reverse].capacity += amount;
          }
          size_t blocked = 0;
          while (residual[path[blocked]][pathArcs[blocked]].capacity)
            ++blocked;
          path.resize(blocked + 1);
          pathArcs.resize(blocked);
          ++nextArc[path.back()];
          continue;
        }
        auto &cursor = nextArc[node];
        while (cursor < residual[node].size() &&
               (!residual[node][cursor].capacity ||
                level[residual[node][cursor].to] != level[node] + 1))
          ++cursor;
        if (cursor == residual[node].size()) {
          path.pop_back();
          if (!pathArcs.empty()) {
            pathArcs.pop_back();
            if (!path.empty())
              ++nextArc[path.back()];
          }
        } else {
          pathArcs.push_back(cursor);
          path.push_back(residual[node][cursor].to);
        }
      }
    }
  }

  Result minimumGlobalCut() const {
    size_t nodeCount = adjacency.size();
    auto connected = reachable(0);
    if (std::find(connected.begin(), connected.end(), 0) != connected.end())
      return result(std::move(connected));

    uint32_t smallestDegree = 0;
    for (uint32_t node = 1; node < nodeCount; ++node)
      if (adjacency[node].size() < adjacency[smallestDegree].size())
        smallestDegree = node;
    auto singleton = [&]() {
      std::vector<uint8_t> side(nodeCount, 0);
      side[smallestDegree] = 1;
      return result(std::move(side));
    };
    if (adjacency[smallestDegree].size() == 1)
      return singleton();
    if (uint32_t bridge = findBridge(); bridge != Report::none)
      return result(reachable(0, bridge));
    if (adjacency[smallestDegree].size() == 2)
      return singleton();

    // Stoer-Wagner contracts one vertex per phase. The phase's last vertex
    // defines an exact candidate cut, and the minimum over phases is global.
    std::vector<std::unordered_map<uint32_t, uint32_t>> weights(nodeCount);
    for (const auto &edge : edges) {
      ++weights[edge.lhs][edge.rhs];
      ++weights[edge.rhs][edge.lhs];
    }
    std::vector<std::vector<uint32_t>> members(nodeCount);
    std::vector<uint32_t> active;
    active.reserve(nodeCount);
    for (uint32_t node = 0; node < nodeCount; ++node) {
      members[node].push_back(node);
      active.push_back(node);
    }
    uint32_t bestWeight = adjacency[smallestDegree].size();
    std::vector<uint32_t> bestSide{smallestDegree};
    while (active.size() > 1) {
      std::vector<uint8_t> added(nodeCount, 0);
      std::vector<uint32_t> connectionWeight(nodeCount, 0);
      std::priority_queue<std::pair<uint32_t, uint32_t>> frontier;
      uint32_t previous = active.front();
      added[previous] = 1;
      for (auto [neighbor, weight] : weights[previous]) {
        connectionWeight[neighbor] = weight;
        frontier.push({weight, neighbor});
      }
      for (size_t step = 1; step < active.size(); ++step) {
        while (
            !frontier.empty() &&
            (added[frontier.top().second] ||
             frontier.top().first != connectionWeight[frontier.top().second]))
          frontier.pop();
        uint32_t current = frontier.top().second;
        uint32_t weight = frontier.top().first;
        frontier.pop();
        added[current] = 1;
        if (step + 1 == active.size()) {
          if (weight < bestWeight) {
            bestWeight = weight;
            bestSide = members[current];
          }
          for (auto [neighbor, count] : weights[current]) {
            if (neighbor == previous)
              continue;
            weights[previous][neighbor] += count;
            weights[neighbor][previous] += count;
            weights[neighbor].erase(current);
          }
          weights[previous].erase(current);
          weights[current].clear();
          members[previous].insert(members[previous].end(),
                                   members[current].begin(),
                                   members[current].end());
          members[current].clear();
          active.erase(std::find(active.begin(), active.end(), current));
          break;
        }
        previous = current;
        for (auto [neighbor, count] : weights[current]) {
          if (added[neighbor])
            continue;
          connectionWeight[neighbor] += count;
          frontier.push({connectionWeight[neighbor], neighbor});
        }
      }
      // There is no bridge, so a cut of two cannot be improved further.
      if (bestWeight == 2)
        break;
    }
    std::vector<uint8_t> side(nodeCount, 0);
    for (uint32_t node : bestSide)
      side[node] = 1;
    return result(std::move(side));
  }

  std::vector<Edge> edges;
  std::vector<std::vector<uint32_t>> adjacency;
};

} // namespace

JsonValue Report::minCut(const Object &params, std::string &error) const {
  auto domainId = parseId(params, "domainTypeId");
  auto domainIt = domainId ? domainById.find(*domainId) : domainById.end();
  if (domainIt == domainById.end()) {
    error = "unknown domain type ID";
    return nullptr;
  }
  bool associationOnly = false;
  if (const auto *requested = params.get("associationOnly")) {
    auto enabled = requested->getAsBoolean();
    if (!enabled) {
      error = "associationOnly must be a boolean";
      return nullptr;
    }
    associationOnly = *enabled;
  }
  uint32_t domainIndex = domainIt->second;
  auto componentNumber = params.getInteger("componentIndex");
  if (componentNumber &&
      (*componentNumber < 0 || static_cast<uint64_t>(*componentNumber) >=
                                   domainComponents[domainIndex].size())) {
    error = "unknown connected component";
    return nullptr;
  }
  const auto &nodes =
      componentNumber ? domainComponents[domainIndex][*componentNumber].nodes
                      : domainNodes[domainIndex];
  const auto &connections =
      componentNumber ? domainComponents[domainIndex][*componentNumber].edges
                      : domainEdges[domainIndex];
  if (nodes.size() < 2)
    return Object{{"available", false},
                  {"message", "The graph has fewer than two nodes"},
                  {"associationOnly", associationOnly}};
  bool hasSource = params.getString("sourceId").has_value();
  bool hasTarget = params.getString("targetId").has_value();
  if (hasSource != hasTarget || (componentNumber && hasSource)) {
    error = "provide either a component or both sourceId and targetId";
    return nullptr;
  }
  uint32_t source = none, target = none;
  if (hasSource) {
    auto sourceId = parseId(params, "sourceId");
    auto targetId = parseId(params, "targetId");
    auto sourceIt = sourceId ? valueById.find(*sourceId) : valueById.end();
    auto targetIt = targetId ? valueById.find(*targetId) : valueById.end();
    if (sourceIt == valueById.end() || targetIt == valueById.end()) {
      error = "unknown source or target value ID";
      return nullptr;
    }
    auto sourcePos =
        std::lower_bound(nodes.begin(), nodes.end(), sourceIt->second);
    auto targetPos =
        std::lower_bound(nodes.begin(), nodes.end(), targetIt->second);
    if (sourcePos == nodes.end() || *sourcePos != sourceIt->second ||
        targetPos == nodes.end() || *targetPos != targetIt->second) {
      error = "source and target must belong to this domain graph";
      return nullptr;
    }
    source = sourcePos - nodes.begin();
    target = targetPos - nodes.begin();
    if (source == target) {
      error = "source and target must be different graph nodes";
      return nullptr;
    }
  }

  uint32_t startValue = nodes[hasSource ? source : 0];
  std::vector<uint8_t> inComponent(values.size(), 0);
  std::vector<uint32_t> component;
  if (componentNumber) {
    component = nodes;
    for (uint32_t node : component)
      inComponent[node] = 1;
  } else {
    component.push_back(startValue);
    inComponent[startValue] = 1;
    for (size_t front = 0; front < component.size(); ++front) {
      uint32_t current = component[front];
      for (uint32_t i = adjacencyOffsets[current];
           i < adjacencyOffsets[current + 1]; ++i) {
        const auto &edge = edges[adjacencyEdges[i]];
        if (edge.domain != domainIndex || edge.lhs == none ||
            edge.rhs == none || edge.lhs == edge.rhs)
          continue;
        uint32_t neighbor = edge.lhs == current ? edge.rhs : edge.lhs;
        if (!inComponent[neighbor]) {
          inComponent[neighbor] = 1;
          component.push_back(neighbor);
        }
      }
    }
  }
  if ((!hasSource && !componentNumber && component.size() < nodes.size()) ||
      (hasSource && !inComponent[nodes[target]])) {
    Object response{{"available", true},
                    {"mode", hasSource ? "between" : "global"},
                    {"associationOnly", associationOnly},
                    {"count", 0},
                    {"nodeCount", static_cast<int64_t>(nodes.size())},
                    {"sourceSideSize", static_cast<int64_t>(component.size())},
                    {"targetSideSize",
                     static_cast<int64_t>(nodes.size() - component.size())},
                    {"edges", Array()}};
    if (hasSource) {
      response["sourceId"] = std::to_string(*parseId(params, "sourceId"));
      response["targetId"] = std::to_string(*parseId(params, "targetId"));
    }
    return response;
  }

  std::vector<uint32_t> localIndex(values.size(), none);
  for (uint32_t i = 0; i < component.size(); ++i)
    localIndex[component[i]] = i;
  std::vector<uint32_t> cutNode(component.size());
  uint32_t cutNodeCount = component.size();
  auto associationKind =
      std::find(edgeKinds.begin(), edgeKinds.end(), "association") -
      edgeKinds.begin();
  if (associationOnly) {
    // Every non-association edge must remain intact. Contract its endpoints
    // before finding a cut among the remaining association edges.
    std::vector<uint32_t> parent(component.size()), size(component.size(), 1);
    for (uint32_t i = 0; i < component.size(); ++i)
      parent[i] = i;
    auto findRoot = [&](uint32_t node) {
      while (parent[node] != node) {
        parent[node] = parent[parent[node]];
        node = parent[node];
      }
      return node;
    };
    for (uint32_t edgeIndex : connections) {
      const auto &edge = edges[edgeIndex];
      if (!inComponent[edge.lhs] || edge.kind == associationKind)
        continue;
      uint32_t lhs = findRoot(localIndex[edge.lhs]);
      uint32_t rhs = findRoot(localIndex[edge.rhs]);
      if (lhs == rhs)
        continue;
      if (size[lhs] < size[rhs])
        std::swap(lhs, rhs);
      parent[rhs] = lhs;
      size[lhs] += size[rhs];
    }
    std::vector<uint32_t> rootToNode(component.size(), none);
    cutNodeCount = 0;
    for (uint32_t i = 0; i < component.size(); ++i) {
      uint32_t root = findRoot(i);
      if (rootToNode[root] == none)
        rootToNode[root] = cutNodeCount++;
      cutNode[i] = rootToNode[root];
    }
    if (cutNodeCount < 2 ||
        (hasSource && cutNode[localIndex[nodes[source]]] ==
                          cutNode[localIndex[nodes[target]]]))
      return Object{
          {"available", false},
          {"mode", hasSource         ? "between"
                   : componentNumber ? "component"
                                     : "global"},
          {"associationOnly", true},
          {"message", hasSource
                          ? "No association-only cut exists: these nodes are "
                            "connected without crossing an association edge"
                          : "No association-only cut exists: non-association "
                            "edges keep this component connected"}};
  } else {
    for (uint32_t i = 0; i < component.size(); ++i)
      cutNode[i] = i;
  }

  CutGraph graph(cutNodeCount);
  for (uint32_t edgeIndex : connections) {
    const auto &edge = edges[edgeIndex];
    if (!inComponent[edge.lhs] ||
        (associationOnly && edge.kind != associationKind))
      continue;
    uint32_t lhs = cutNode[localIndex[edge.lhs]];
    uint32_t rhs = cutNode[localIndex[edge.rhs]];
    if (lhs != rhs)
      graph.addEdge(lhs, rhs, edgeIndex);
  }
  auto cut = hasSource ? graph.minimumSTCut(cutNode[localIndex[nodes[source]]],
                                            cutNode[localIndex[nodes[target]]])
                       : graph.minimumGlobalCut();
  uint64_t sourceSideSize = 0;
  for (uint32_t node : cutNode)
    sourceSideSize += cut.sourceSide[node];
  Array cutEdges;
  for (uint32_t edgeIndex : cut.edgeIndices)
    cutEdges.push_back(edgeJSON(edgeIndex));
  Object response{
      {"available", true},
      {"mode", hasSource         ? "between"
               : componentNumber ? "component"
                                 : "global"},
      {"associationOnly", associationOnly},
      {"count", static_cast<int64_t>(cut.edgeIndices.size())},
      {"nodeCount", static_cast<int64_t>(nodes.size())},
      {"sourceSideSize", static_cast<int64_t>(sourceSideSize)},
      {"targetSideSize", static_cast<int64_t>(nodes.size() - sourceSideSize)},
      {"edges", std::move(cutEdges)}};
  if (hasSource) {
    response["sourceId"] = std::to_string(*parseId(params, "sourceId"));
    response["targetId"] = std::to_string(*parseId(params, "targetId"));
  }
  if (componentNumber)
    response["componentIndex"] = *componentNumber;
  return response;
}

} // namespace circt::domain_report
