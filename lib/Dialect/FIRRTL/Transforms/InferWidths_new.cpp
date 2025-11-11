//===- InferWidths_new.cpp - Infer width of types -------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferWidths pass.(new)
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"

#define DEBUG_TYPE "infer-widths-new"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERWIDTHS_NEW
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

namespace circt {

struct Zero {
  bool operator==(const Zero &) const { return true; }
  bool operator==(const FieldRef &) const { return false; }
  bool operator!=(const Zero &) const { return false; }
  bool operator!=(const FieldRef &) const { return true; }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Zero &) {
    return os << "Zero";
  }
};

bool FieldRef::operator==(const Zero &) const { return false; }
bool FieldRef::operator!=(const Zero &) const { return true; }
} // namespace circt

using mlir::InferTypeOpInterface;
using mlir::WalkOrder;

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static void diagnoseUninferredType(InFlightDiagnostic &diag, Type t,
                                   Twine str) {
  auto basetype = type_dyn_cast<FIRRTLBaseType>(t);
  if (!basetype)
    return;
  if (!basetype.hasUninferredWidth())
    return;

  if (basetype.isGround())
    diag.attachNote() << "Field: \"" << str << "\"";
  else if (auto vecType = type_dyn_cast<FVectorType>(basetype))
    diagnoseUninferredType(diag, vecType.getElementType(), str + "[]");
  else if (auto bundleType = type_dyn_cast<BundleType>(basetype))
    for (auto &elem : bundleType.getElements())
      diagnoseUninferredType(diag, elem.type, str + "." + elem.name.getValue());
}

/// Calculate the "InferWidths-fieldID" equivalent for the given fieldID + type.
static uint64_t convertFieldIDToOurVersion(uint64_t fieldID, FIRRTLType type) {
  uint64_t convertedFieldID = 0;

  auto curFID = fieldID;
  Type curFType = type;
  while (curFID != 0) {
    auto [child, subID] =
        hw::FieldIdImpl::getSubTypeByFieldID(curFType, curFID);
    if (isa<FVectorType>(curFType))
      convertedFieldID++; // Vector fieldID is 1.
    else
      convertedFieldID += curFID - subID; // Add consumed portion.
    curFID = subID;
    curFType = child;
  }

  return convertedFieldID;
}

//===----------------------------------------------------------------------===//
// Constraint Expressions
//===----------------------------------------------------------------------===//

/// Check if a type contains any FIRRTL type with uninferred widths.
static bool hasUninferredWidth(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<FIRRTLBaseType>([](auto base) { return base.hasUninferredWidth(); })
      .Case<RefType>(
          [](auto ref) { return ref.getType().hasUninferredWidth(); })
      .Default([](auto) { return false; });
}

using Valuation = std::unordered_map<FieldRef, int32_t, FieldRef::Hash>;
using nat = unsigned int;

class Term {
private:
  nat coe_;
  FieldRef var_;

public:
  Term(nat val, const FieldRef &var) : coe_(val), var_(var) {}

  nat coe() const { return coe_; }
  const FieldRef &var() const { return var_; }

  void setCoe(nat newCoe) { coe_ = newCoe; }
  void setVar(const FieldRef &newVar) { var_ = newVar; }

  bool operator==(const Term &other) const {
    return (coe_ == other.coe_) && (var_ == other.var_);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Term &term) {
    return os << term.coe() << " * [" << term.var().getValue()
              << " (fieldID: " << term.var().getFieldID() << ")]";
  }
};

#include <list>
class Terms {
private:
  std::list<Term> terms_;

public:
  Terms() = default;
  Terms(std::initializer_list<Term> init) : terms_(init) {}
  Terms(const std::list<Term> &terms) : terms_(terms) {}

  auto begin() const { return terms_.begin(); }
  auto end() const { return terms_.end(); }
  auto begin() { return terms_.begin(); }
  auto end() { return terms_.end(); }

  const std::list<Term> &get_terms() const { return terms_; }
  size_t size() const { return terms_.size(); }
  bool empty() const { return terms_.empty(); }

  void push_front(const Term &term) { terms_.push_front(term); }
  void push_back(const Term &term) { terms_.push_back(term); }

  bool operator==(const Terms &other) const { return terms_ == other.terms_; }
  bool operator!=(const Terms &other) const { return !(*this == other); }

  Terms combine_term(const Term &term) const {
    Terms result = *this; 
    auto it =
        std::find_if(result.terms_.begin(), result.terms_.end(),
                     [&](const Term &t) { return t.var() == term.var(); });

    if (it == result.terms_.end()) {
      result.push_front(term);
    } else {
      Term combined(term.coe() + it->coe(), term.var());
      result.terms_.erase(it);
      result.push_front(combined);
    }
    return result;
  }

  Terms combine_terms(const Terms &other) const {
    Terms result = *this;
    for (const auto &term : other.terms_) {
      result = result.combine_term(term);
    }
    return result;
  }

  static std::tuple<Terms, Terms, int>
  combine_terms(const Terms &terms1, const Terms &terms2, int cst1, int cst2) {
    Terms result = terms1.combine_terms(terms2);
    int new_cst = cst1 + cst2;
    return {result, Terms(), new_cst};
  }

  long long evaluate(const Valuation &v) const {
    long long result = 0;
    for (const auto &term : terms_) {
      auto it = v.find(term.var());
      nat var_value = (it != v.end()) ? it->second : 0;
      result += static_cast<long long>(term.coe()) * var_value;
    }
    return result;
  }

  std::optional<FieldRef> findVarWithCoeGreaterThanOne() const {
    for (const auto &term : terms_) {
      if (term.coe() > 1) {
        return term.var(); 
      }
    }
    return std::nullopt; 
  }

  std::optional<std::pair<FieldRef, FieldRef>> findFirstTwoVars() const {
    if (terms_.size() < 2) {
      return std::nullopt; 
    }

    auto it = terms_.begin();
    const FieldRef &firstVar = it->var();
    ++it;
    const FieldRef &secondVar = it->var(); 

    return std::make_pair(firstVar, secondVar);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Terms &terms) {
    bool first = true;
    for (const auto &term : terms) {
      if (!first)
        os << " + ";
      os << term;
      first = false;
    }
    if (first)
      os << "0";
    return os;
  }
};

class Constraint1 {
private:
  FieldRef lhs_var1_;
  int rhs_const1_;
  Terms rhs_terms1_;
  std::optional<FieldRef> rhs_power_;

public:
  Constraint1(FieldRef lhs_var, int rhs_const, Terms rhs_terms,
              std::optional<FieldRef> rhs_power)
      : lhs_var1_(lhs_var), rhs_const1_(rhs_const), rhs_terms1_(rhs_terms),
        rhs_power_(rhs_power) {}

  const FieldRef &lhs_var1() const { return lhs_var1_; }
  int rhs_const1() const { return rhs_const1_; }
  const Terms &rhs_terms1() const { return rhs_terms1_; }
  const std::optional<FieldRef> &rhs_power() const { return rhs_power_; }

  void set_lhs_var1(const FieldRef &var) { lhs_var1_ = var; }
  void set_rhs_const1(int constant) { rhs_const1_ = constant; }
  void set_rhs_terms1(const Terms &terms) { rhs_terms1_ = terms; }
  void set_rhs_power(const std::optional<FieldRef> &terms) {
    rhs_power_ = terms;
  }

  bool operator==(const Constraint1 &other) const {
    return (lhs_var1_ == other.lhs_var1_) &&
           (rhs_const1_ == other.rhs_const1_) &&
           (rhs_terms1_ == other.rhs_terms1_) &&
           (rhs_power_ == other.rhs_power_);
  }
  bool operator!=(const Constraint1 &other) const { return !(*this == other); }

  bool satisfies(const Valuation &v) const;

  long long power_value(const Valuation &v) const {
    if (rhs_power_.has_value()) {
      FieldRef pv_power(rhs_power_.value());
      auto it = v.find(pv_power);
      long long exponent = (it != v.end()) ? it->second : 0;
      return 1LL << exponent;
    } else
      return 0;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Constraint1 &c) {
    os << "[" << c.lhs_var1().getValue()
       << " (fieldID: " << c.lhs_var1().getFieldID()
       << ")] >= " << c.rhs_const1();
    if (!c.rhs_terms1().empty()) {
      os << " + " << c.rhs_terms1();
    }
    if (c.rhs_power().has_value()) {
      os << " + 2 ^ [" << c.rhs_power().value().getValue()
         << " (fieldID: " << c.rhs_power().value().getFieldID() << ")]";
    }
    return os;
  }
};

bool Constraint1::satisfies(const Valuation &v) const {
  auto lhs_it = v.find(lhs_var1_);
  if (lhs_it == v.end()) {
    return false; 
  }
  long long lhs_value = lhs_it->second;
  long long linear_value = rhs_terms1_.evaluate(v);
  long long power_val = power_value(v);
  long long rhs_total = linear_value + power_val + rhs_const1_;

  return lhs_value >= rhs_total;
}

class Constraint_Min {
private:
  FieldRef lhs_;
  int const1_;
  int const2_;
  std::optional<FieldRef> fr1_;
  std::optional<FieldRef> fr2_;

public:
  Constraint_Min(FieldRef lhs, int const1, int const2,
                 std::optional<FieldRef> fr1, std::optional<FieldRef> fr2)
      : lhs_(lhs), const1_(const1), const2_(const2), fr1_(fr1), fr2_(fr2) {}

  const FieldRef &lhs() const { return lhs_; }
  int const1() const { return const1_; }
  int const2() const { return const2_; }
  const std::optional<FieldRef> &fr1() const { return fr1_; }
  const std::optional<FieldRef> &fr2() const { return fr2_; }

  void set_lhs_(const FieldRef &var) { lhs_ = var; }
  void set_const1(int constant) { const1_ = constant; }
  void set_fr1(const std::optional<FieldRef> &fr) { fr1_ = fr; }
  void set_fr2(const std::optional<FieldRef> &fr) { fr2_ = fr; }

  bool operator==(const Constraint_Min &other) const {
    return (lhs_ == other.lhs_) && (const1_ == other.const1_) &&
           (const2_ == other.const2_) && (fr1_ == other.fr1_) &&
           (fr2_ == other.fr2_);
  }

  bool operator!=(const Constraint_Min &other) const {
    return !(*this == other);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Constraint_Min &c) {
    os << "[" << c.lhs().getValue() << " (fieldID: " << c.lhs().getFieldID()
       << ")] >= min(" << c.const1();
    if (c.fr1().has_value()) {
      os << " + [" << c.fr1().value().getValue()
         << " (fieldID: " << c.fr1().value().getFieldID() << ")]";
    }

    os << ", " << c.const2();
    if (c.fr2().has_value()) {
      os << " + [" << c.fr2().value().getValue()
         << " (fieldID: " << c.fr2().value().getFieldID() << ")]";
    }
    os << ")\n";

    return os;
  }
};

class Constraint2 {
private:
  nat lhs_const2_;
  Terms rhs_terms2_;

public:
  Constraint2(nat lhs_const = 0, Terms rhs_terms = Terms())
      : lhs_const2_(lhs_const), rhs_terms2_(rhs_terms) {}

  nat lhs_const2() const { return lhs_const2_; }
  const Terms &rhs_terms2() const { return rhs_terms2_; }

  void set_lhs_const2(nat constant) { lhs_const2_ = constant; }
  void set_rhs_terms2(const Terms &terms) { rhs_terms2_ = terms; }

  bool operator==(const Constraint2 &other) const {
    return (lhs_const2_ == other.lhs_const2_) &&
           (rhs_terms2_ == other.rhs_terms2_);
  }
  bool operator!=(const Constraint2 &other) const { return !(*this == other); }

  bool satisfies(const Valuation &v) const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Constraint2 &c) {
    os << c.lhs_const2_ << " >= " << c.rhs_terms2_;
    return os;
  }
};

bool Constraint2::satisfies(const Valuation &v) const {
  long long rhs_value = rhs_terms2_.evaluate(v);
  return static_cast<long long>(lhs_const2_) >= rhs_value;
}

using FieldRef_with_default = std::variant<Zero, FieldRef>;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FieldRef_with_default& node) {
    if (std::holds_alternative<Zero>(node)) {
        os << "Zero";
    } else {
        auto& pv = std::get<FieldRef>(node);
        os << "FieldRef[" << pv.getValue() << " (fieldID: " << pv.getFieldID() << ")]";
    }
    return os;
}

struct FieldRef_wd_Hash {
    size_t operator()(const FieldRef_with_default& node) const {
        if (std::holds_alternative<Zero>(node)) {
            return 0; 
        } else {
            auto& pv = std::get<FieldRef>(node);
            return static_cast<size_t>(hash_value(pv));
        }
    }
};

struct Node {
    FieldRef_with_default field;
    std::vector<Node*> successors;
    
    Node(Zero z) : field(z) {}
    Node(Value value, unsigned id) : field(FieldRef{value, id}) {}
    Node(const FieldRef_with_default& fr) : field(fr) {}
    
    void addSuccessor(Node* node) {
        successors.push_back(node);
    }
};

struct FieldRefGraph {
    std::deque<Node> nodes;
    std::unordered_map<FieldRef_with_default, Node*, FieldRef_wd_Hash> nodeMap;
    
    Node* addNode_wd(Zero z) {
        FieldRef_with_default key = z;
        if (auto it = nodeMap.find(key); it != nodeMap.end()) {
            return it->second;
        }
 
        nodes.emplace_back(z);
        Node* newNode = &nodes.back();
        nodeMap[key] = std::move(newNode);
        return newNode;
    }
    
    Node* addNode_wd(Value value, unsigned id) {
        FieldRef_with_default key = FieldRef{value, id};
        if (auto it = nodeMap.find(key); it != nodeMap.end()) {
            return it->second;
        }
        
        nodes.emplace_back(value, id);
        Node* newNode = &nodes.back();
        nodeMap[key] = std::move(newNode);

        Node* root = addNode_wd(Zero{});
        root->addSuccessor(newNode);

        return newNode;
    }

    Node* addNode_wd(FieldRef key) {
        if (auto it = nodeMap.find(key); it != nodeMap.end()) {
            return it->second;
        }
        
        nodes.emplace_back(key);
        Node* newNode = &nodes.back();
        nodeMap[key] = std::move(newNode);

        Node* root = addNode_wd(Zero{});
        root->addSuccessor(newNode);

        return newNode;
    }

    Node* addNode(FieldRef key) {
        if (auto it = nodeMap.find(key); it != nodeMap.end()) {
            return it->second;
        }
        
        nodes.emplace_back(key);
        Node* newNode = &nodes.back();
        nodeMap[key] = std::move(newNode);
        return newNode;
    }
    
    Node* getEntryNode() {
      Node* root = addNode_wd(Zero{});
      return root;
    }

  FieldRefGraph() = default;

  FieldRefGraph(const FieldRefGraph& other) {
    std::unordered_map<const Node*, Node*> oldToNewMap;

    for (const Node& oldNode : other.nodes) {
        nodes.emplace_back(oldNode.field); 
        Node* newNode = &nodes.back();
        oldToNewMap[&oldNode] = newNode;
        nodeMap[newNode->field] = &nodes.back();
    }

    for (const Node& oldNode : other.nodes) {
        Node* newNode = oldToNewMap[&oldNode];
        for (Node* oldSuccessor : oldNode.successors) {
            auto it = oldToNewMap.find(oldSuccessor);
            if (it != oldToNewMap.end()) {
                newNode->successors.push_back(it->second);
            }
        }
    }
  }
};

namespace llvm {
    template <> struct GraphTraits<FieldRefGraph*> {
        using NodeRef = Node*; 
        using ChildIteratorType = std::vector<Node*>::iterator;  
        
        static NodeRef getEntryNode(FieldRefGraph* G) {
            return G->getEntryNode();
        }
        
        static ChildIteratorType child_begin(NodeRef node) {
            return node->successors.begin();
        }
        
        static ChildIteratorType child_end(NodeRef node) {
            return node->successors.end();
        }
        
        static NodeRef nodes_begin(FieldRefGraph* G) {
            return G->nodes.empty() ? nullptr : &G->nodes[0];
        }
        
        static NodeRef nodes_end(FieldRefGraph* G) {
            return G->nodes.empty() ? nullptr : &G->nodes[0] + G->nodes.size();
        }
        
    };
}

//===----------------------------------------------------------------------===//
// Constraint Solver
//===----------------------------------------------------------------------===//

class ConstraintSolver {
private:

  std::unordered_map<FieldRef, std::vector<Constraint1>, FieldRef::Hash>
      constraints_;

  Valuation solution_ = {};

  FieldRefGraph graph_;

public:
  explicit ConstraintSolver(
      std::unordered_map<FieldRef, std::vector<Constraint1>, FieldRef::Hash>
          &constraints, FieldRefGraph &graph)
      : constraints_(constraints), graph_(graph) {}

  void addConstraint(const Constraint1 &c) {
    auto &vec = constraints_[c.lhs_var1()];
    vec.push_back(c);

    FieldRef lhs = c.lhs_var1();
    Node* lhs_node = graph_.addNode_wd(lhs);

    for (const auto &term : c.rhs_terms1()) {
      FieldRef rhs_var = term.var();
      Node* rhs_node = graph_.addNode_wd(rhs_var);
      lhs_node->addSuccessor(rhs_node);
    }

    if (c.rhs_power().has_value()) {
      FieldRef rhs_var = c.rhs_power().value();
      Node* rhs_node = graph_.addNode_wd(rhs_var);
      lhs_node->addSuccessor(rhs_node);
    }
  }

  FieldRefGraph &fieldRefGraph() { return graph_; }

  std::vector<Constraint1> constraints() const {
    std::vector<Constraint1> result;
    for (const auto &[_, constraints_vec] : constraints_) {
      result.insert(result.end(), constraints_vec.begin(),
                    constraints_vec.end());
    }
    return result;
  }

  const std::unordered_map<FieldRef, std::vector<Constraint1>, FieldRef::Hash> &
  constraints_map() const {
    return constraints_;
  }

  LogicalResult solve();

  const Valuation &solution() const { return solution_; }
};

bool is_simple_cycle(const std::vector<Constraint1> &cs) {
  return std::all_of(cs.begin(), cs.end(), [](const Constraint1 &c) {
    const auto &terms = c.rhs_terms1();
    if (c.rhs_power().has_value()) {
      return false;
    }

    if (terms.empty()) {
      return true;
    }

    auto it = terms.begin();
    if (it->coe() != 1) { 
      return false;
    }

    return std::next(it) == terms.end();
  });
}

std::vector<Constraint1>
filterConstraints(const std::vector<FieldRef> &targetVars,
                  const std::vector<Constraint1> &constraints) {
  std::unordered_set<FieldRef, FieldRef::Hash> targetSet(targetVars.begin(),
                                                         targetVars.end());

  std::vector<Constraint1> result;
  for (const auto &constraint : constraints) {
    if (targetSet.find(constraint.lhs_var1()) != targetSet.end()) {
      result.push_back(constraint);
    }
  }
  return result;
}

std::vector<Constraint1> filterConstraints(
    const std::vector<FieldRef> &targetVars,
    std::unordered_map<FieldRef, std::vector<Constraint1>, FieldRef::Hash>
        &constraints_map) {
  std::vector<Constraint1> result;
  for (const auto &var : targetVars) {
    auto &vec = constraints_map[var];
    result.insert(result.end(), vec.begin(), vec.end());
  }
  return result;
}

std::pair<Terms, long long> remove_solved(const Valuation &values,
                                          const Terms &terms) {
  Terms new_terms;
  long long total_constant = 0;

  for (const Term &term : terms.get_terms()) {
    auto it = values.find(term.var());
    if (it != values.end()) {
      unsigned int val = it->second;
      total_constant += static_cast<long long>(term.coe()) * val;
    } else {
      new_terms.push_back(term);
    }
  }

  return {new_terms, total_constant};
}

Constraint1 remove_solved_c(const Valuation &values, const Constraint1 &c) {
  auto [new_terms, term_constant] = remove_solved(values, c.rhs_terms1());
  if (c.rhs_power().has_value()) {
    auto it = values.find(c.rhs_power().value());
    if (it != values.end()) {
      long long power_value = c.power_value(values);
      return Constraint1(
          c.lhs_var1(),
          c.rhs_const1() + static_cast<int>(term_constant + power_value),
          new_terms,
          std::nullopt 
      );
    } else {
      return Constraint1(c.lhs_var1(),
                         c.rhs_const1() + static_cast<int>(term_constant),
                         new_terms,
                         c.rhs_power() 
      );
    }
  } else {
    return Constraint1(c.lhs_var1(),
                       c.rhs_const1() + static_cast<int>(term_constant),
                       new_terms, c.rhs_power());
  }
}

std::vector<Constraint1> remove_solveds(const Valuation &values,
                                        const std::vector<Constraint1> &cs) {
  std::vector<Constraint1> result;
  result.reserve(cs.size());

  for (const Constraint1 &c : cs) {
    result.push_back(remove_solved_c(values, c));
  }

  return result;
}

std::optional<Valuation> merge_solution(const std::vector<FieldRef> &tbsolved,
                                        const Valuation &initial,
                                        const Valuation &solution_of_tbsolved) {
  Valuation result = initial; 

  for (const FieldRef &var : tbsolved) {
    auto it = solution_of_tbsolved.find(var);
    if (it == solution_of_tbsolved.end()) {
      return std::nullopt; 
    }

    result[var] = it->second;
  }

  return result;
}

Valuation bab(const std::vector<Constraint1> &constraints,
              const std::vector<FieldRef> &tbsolved);

std::vector<FieldRef> extractFieldRefs(const std::vector<Node*>& nodes) {
    std::vector<FieldRef> result;
    for (Node* node : nodes) {
        if (!node) continue;
        
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, FieldRef>) {
                result.push_back(arg);  
            }
        }, node->field);
    }
    return result;
}

const int INF = 1e9;
Valuation floyd(const std::vector<Constraint1> &constraints, const std::vector<FieldRef> &tbsolved) {
  std::unordered_map<FieldRef_with_default, int, FieldRef_wd_Hash> var_to_index;

  int next_index = 0;
  for (const auto& var : tbsolved) 
    var_to_index[var] = next_index++;
  var_to_index[Zero{}] = next_index;

  LLVM_DEBUG(llvm::dbgs() << "floyd邻接矩阵标号:\n");
  for (const auto &[var, index] : var_to_index) 
    LLVM_DEBUG(llvm::dbgs() << var << " : " << index << "\n");
    
  // 初始化邻接矩阵（用INT_MAX表示无连接）
  int n = var_to_index.size();
  std::vector<std::vector<int>> graph(n, std::vector<int>(n, INF));
    
  // 填充邻接矩阵
  for (const auto& constraint : constraints) {
    FieldRef_with_default source = constraint.lhs_var1();
    FieldRef_with_default target;
    if (constraint.rhs_terms1().empty()) 
      // 右侧没有项，指向Zero节点
      target = Zero{};
    else 
      // 指向右侧第一个项的变量
      target = constraint.rhs_terms1().begin()->var();

    int i = var_to_index[source];
    int j = var_to_index[target];
    int weight = -constraint.rhs_const1();  // x >= y + c  =>  edge: x -> y with weight -c
    // 保留权重最小值（对应最严格约束）
    if (weight < graph[i][j]) 
      graph[i][j] = weight;
  }
      
  LLVM_DEBUG(llvm::dbgs() << "初始邻接矩阵:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (graph[i][j] == INF) LLVM_DEBUG(llvm::dbgs() << "  INF  ");
      else LLVM_DEBUG(llvm::dbgs() << "  " << graph[i][j] << "  ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  // 开始floyd算法
  // 三重循环动态规划
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        // 跳过无穷大路径避免溢出
        if (graph[i][k] < INF && graph[k][j] < INF) {
          graph[i][j] = std::min(graph[i][j], graph[i][k] + graph[k][j]);
        }
      }
    }
  }

  // 检测负权环
  for (int i = 0; i < n; i++) {
    if (graph[i][i] < 0) {
      LLVM_DEBUG(llvm::dbgs() << "发现负权环！无法计算有效最短路径" << "\n");
    }
  }

  // 打印所有最短路径
  LLVM_DEBUG(llvm::dbgs() << "最短路径矩阵:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (graph[i][j] == INF) LLVM_DEBUG(llvm::dbgs() << "  INF  ");
      else LLVM_DEBUG(llvm::dbgs() << "  " << graph[i][j] << "  ");
    }
  LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  Valuation valuation;
  for (int i = 0; i < n-1; i++) {
    FieldRef source = tbsolved[i];
    int min_distance = 0;
    for (int j = 0; j < n; j++) {
      int d = graph[i][j];
      // 更新最短路径值
      if (d < min_distance) {
        min_distance = d;
      }
    }
    // 将结果存入valuation
    valuation[source] = -min_distance;
  }

  LLVM_DEBUG(llvm::dbgs() << "floyd算法结果:\n");
  for (const auto &[var, value] : valuation) 
    LLVM_DEBUG(llvm::dbgs() << var_to_index[var] << " : " << value << "\n");

  return valuation;
}

LogicalResult ConstraintSolver::solve() {
  //SCCResult result = computeSCC();
  // solver.printSCCResult(result);

  for (auto sccIter = llvm::scc_begin(&graph_); sccIter != llvm::scc_end(&graph_); ++sccIter) {
    const auto& node_list = *sccIter;
    LLVM_DEBUG(llvm::dbgs() << "SCC (size " << node_list.size() << "): ");
    for (Node* node : node_list) 
      LLVM_DEBUG(llvm::dbgs() << node->field << "; ");
    LLVM_DEBUG(llvm::dbgs() << "\n");

  //for (size_t i = 0; i < result.topological_order.size(); ++i) {
    //const auto &component = result.components[i];
    std::vector<FieldRef> component = extractFieldRefs(node_list);
    if (component.empty()) { continue; }
    std::vector<Constraint1> tbsolved_cs1;
    tbsolved_cs1 = filterConstraints(component, constraints_);
    auto cs1 = remove_solveds(solution_, tbsolved_cs1);
    // for (const auto& c : tbsolved_cs1) {
    //   LLVM_DEBUG(llvm::dbgs() << c << "\n");
    // }

    Valuation ns;
    if (component.size() == 1) {
      int init = 0;
      for (const auto &c : cs1) {
        init = std::max<int>(init, c.rhs_const1());
      }
      ns = {{component[0], init}};
    } else if (is_simple_cycle(tbsolved_cs1)) {
      ns = floyd(cs1, component);
    } else {
      ns = bab(cs1, component);
    }
    auto merge_result = merge_solution(component, solution_, ns);
    if (!merge_result) {
      LLVM_DEBUG(llvm::dbgs() << "合并失败: 变量未找到\n");
      return failure();
    }
    solution_ = *merge_result;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Constraint Map
//===----------------------------------------------------------------------===//

class InferenceMapping {
private:
  // 待求解约束集合
  std::vector<Constraint_Min> constraints_min_;
  std::unordered_map<FieldRef, std::vector<Constraint1>, FieldRef::Hash>
      constraints_;

  // 依赖关系图
  //DependencyGraph dependency_graph_;
  FieldRefGraph graph_;

  // 顶点映射（变量名 → 顶点描述符）
  // std::unordered_map<FieldRef,
  //                    boost::graph_traits<DependencyGraph>::vertex_descriptor,
  //                    FieldRef::Hash>
  //     vertex_map_;

  /// The fully inferred modules that were skipped entirely.
  SmallPtrSet<Operation *, 16> skippedModules;
  bool allModulesSkipped = true;

  /// Cache of module symbols
  SymbolTable &symtbl;

  /// Full design inner symbol information.
  hw::InnerRefNamespace irn;

  // solution
  Valuation final_solution_ = {};

public:
  InferenceMapping(SymbolTable &symtbl, hw::InnerSymbolTableCollection &istc)
      : symtbl(symtbl), irn{symtbl, istc} {}

  /// Return whether all modules in the mapping were fully inferred.
  bool areAllModulesSkipped() const { return allModulesSkipped; }

  /// Return whether a module was skipped due to being fully inferred already.
  bool isModuleSkipped(FModuleOp module) const {
    return skippedModules.count(module);
  }

  // 获取或添加顶点
  // boost::graph_traits<DependencyGraph>::vertex_descriptor
  // get_or_add_vertex(const FieldRef &var) {
  //   auto it = vertex_map_.find(var);
  //   if (it != vertex_map_.end()) {
  //     return it->second;
  //   } else {
  //     auto vertex = boost::add_vertex(var, dependency_graph_);
  //     vertex_map_[var] = vertex;
  //     return vertex;
  //   }
  // }

  // 添加约束
  void addConstraint(const Constraint_Min &constraint) {
    constraints_min_.push_back(constraint);
  }

  void addConstraint(const Constraint1 &c) {
    auto &vec = constraints_[c.lhs_var1()];
    vec.push_back(c);

    // 直接添加边
    FieldRef lhs = c.lhs_var1();
    //auto lhs_vertex = get_or_add_vertex(lhs);
    Node* lhs_node = graph_.addNode_wd(lhs);

    // 遍历右侧项
    for (const auto &term : c.rhs_terms1()) {
      FieldRef rhs_var = term.var();
      //auto rhs_vertex = get_or_add_vertex(rhs_var);
      Node* rhs_node = graph_.addNode_wd(rhs_var);
      //boost::add_edge(lhs_vertex, rhs_vertex, dependency_graph_);
      lhs_node->addSuccessor(rhs_node);
    }

    // 遍历幂项
    if (c.rhs_power().has_value()) {
      FieldRef rhs_var = c.rhs_power().value();
      //auto rhs_vertex = get_or_add_vertex(rhs_var);
      Node* rhs_node = graph_.addNode_wd(rhs_var);
      //boost::add_edge(lhs_vertex, rhs_vertex, dependency_graph_);
      lhs_node->addSuccessor(rhs_node);
    }
  }

  // 获取依赖图（只读）
  //const DependencyGraph &dependencyGraph() const { return dependency_graph_; }
  FieldRefGraph &fieldRefGraph() { return graph_; }
  // const std::unordered_map<
  //     FieldRef, boost::graph_traits<DependencyGraph>::vertex_descriptor,
  //     FieldRef::Hash> &
  // vertex_map() const {
  //   return vertex_map_;
  // }

  // 获取约束列表
  const std::vector<Constraint_Min> &constraints_min() const {
    return constraints_min_;
  }
  const std::unordered_map<FieldRef, std::vector<Constraint1>, FieldRef::Hash> &
  constraints_map() const {
    return constraints_;
  }

  std::vector<Constraint1> constraints() const {
    std::vector<Constraint1> result;
    for (const auto &[_, constraints_vec] : constraints_) {
      result.insert(result.end(), constraints_vec.begin(),
                    constraints_vec.end());
    }
    return result;
  }

  // 抽取约束
  LogicalResult map(CircuitOp op);
  LogicalResult extractConstraints(Operation *op);
  bool allWidthsKnown(Operation *op);
  void generateConstraints(Value value);
  void generateConstraints(FieldRef lhs, FieldRef rhs, FIRRTLType type);
  void generateConstraints(Value larger, Value smaller);
  void generateConstraints(Value result, Value lhs, Value rhs);
  std::vector<Constraint1> list_Constraint_Min(const Constraint_Min &minc);

  // 求解
  LogicalResult solve();

  // 获取解
  const Valuation &final_solution() const { return final_solution_; }
};

LogicalResult InferenceMapping::map(CircuitOp op) {
  LLVM_DEBUG(llvm::dbgs()
             << "\n===----- Mapping ops to constraint exprs -----===\n\n");

  for (auto module : op.getOps<FModuleOp>()) {
    // Check if the module contains *any* uninferred widths. This allows us to
    // do an early skip if the module is already fully inferred.
    bool anyUninferred = false;
    for (auto arg : module.getArguments()) {
      anyUninferred |= hasUninferredWidth(arg.getType());
      if (anyUninferred)
        break;
    }
    module.walk([&](Operation *op) {
      for (auto type : op->getResultTypes())
        anyUninferred |= hasUninferredWidth(type);
      if (anyUninferred)
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (!anyUninferred) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping fully-inferred module '"
                              << module.getName() << "'\n");
      skippedModules.insert(module);
      continue;
    }

    allModulesSkipped = false;

    // Go through operations in the module and generating constraints.
    auto result = module.getBodyBlock()->walk(
        [&](Operation *op) { return WalkResult(extractConstraints(op)); });
    if (result.wasInterrupted())
      return failure();
  }
  
  /*auto it = llvm::GraphTraits<FieldRefGraph*>::nodes_begin(&graph_);
  auto end = llvm::GraphTraits<FieldRefGraph*>::nodes_end(&graph_);
  for (; it != end; ++it) {
    llvm::errs() << it->field << "的后继: \n";
    for (Node* child : it->successors)
      llvm::errs() << "  " << child->field << "; \n";
  }
  llvm::errs() << "\n";

  // 使用 SCC 迭代器分解强连通分量
  llvm::errs() << "FieldRef Graph SCC Decomposition:\n";
  for (auto sccIter = llvm::scc_begin(&graph_); sccIter != llvm::scc_end(&graph_); ++sccIter) {
    const auto& component = *sccIter;
    llvm::errs() << "SCC (size " << component.size() << "): ";
    for (Node* node : component) 
      llvm::errs() << node->field << "; ";
    llvm::errs() << "\n";
  }*/

  return success();
}

// 将 Constraint_Min 拆开为 Constraint1
std::vector<Constraint1>
InferenceMapping::list_Constraint_Min(const Constraint_Min &minc) {
  std::vector<Constraint1> constraints;

  if (minc.fr1().has_value()) {
    FieldRef fr(minc.fr1().value());
    Term term(1, fr);
    Constraint1 c1(minc.lhs(), minc.const1(), {term}, std::nullopt);
    constraints.push_back(c1);
  } else {
    Constraint1 c1(minc.lhs(), minc.const1(), {}, std::nullopt);
    constraints.push_back(c1);
  }

  if (minc.fr2().has_value()) {
    FieldRef fr(minc.fr2().value());
    Term term(1, fr);
    Constraint1 c1(minc.lhs(), minc.const2(), {term}, std::nullopt);
    constraints.push_back(c1);
  } else {
    Constraint1 c1(minc.lhs(), minc.const2(), {}, std::nullopt);
    constraints.push_back(c1);
  }

  return constraints;
}

void InferenceMapping::generateConstraints(Value value) {
  // generate constraints for every unknown width in the type. If this is a
  // Bundle type or a FVector type, we will have to potentially create many
  // constraints.
  unsigned fieldID = 0;
  std::function<void(FIRRTLBaseType)> generate = [&](FIRRTLBaseType type) {
    auto width = type.getBitWidthOrSentinel();
    if (width >= 0) {
      fieldID++;
    } else if (width == -1) {
      // Unknown width integers generate a constraint.
      FieldRef field(value, fieldID);
      Constraint1 temp_c(field, 0, {}, std::nullopt);
      LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
      addConstraint(temp_c);
      fieldID++;
    } else if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      // Bundle types recursively generate for all bundle elements.
      fieldID++;
      for (auto &element : bundleType)
        generate(element.type);
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      generate(vecType.getElementType());
      // Skip past the rest of the elements
      fieldID = save + vecType.getMaxFieldID();
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  if (auto type = getBaseType(value.getType()))
    generate(type);
}

/// generate constraints of the fields in the `src` argument as the
/// expressions for the `dst` argument. Both fields must be of the given `type`.
void InferenceMapping::generateConstraints(FieldRef lhs, FieldRef rhs,
                                           FIRRTLType type) {
  // Fast path.
  if (lhs == rhs)
    return;

  // Co-iterate the two field refs, recurring into every leaf element and
  // generate
  auto fieldID = 0;
  std::function<void(FIRRTLBaseType)> generate = [&](FIRRTLBaseType type) {
    auto width = type.getBitWidthOrSentinel();
    if (width >= 0) {
      fieldID++;
    } else if (width == -1) {
      // Leaf element, uninferred, generate
      FieldRef lhsFieldRef(lhs.getValue(), lhs.getFieldID() + fieldID);
      FieldRef rhsFieldRef(rhs.getValue(), rhs.getFieldID() + fieldID);
      Term t_src(1, rhsFieldRef);
      Constraint1 temp_c(lhsFieldRef, 0, {t_src}, std::nullopt);
      LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
      addConstraint(temp_c);
      fieldID++;
    } else if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      fieldID++;
      for (auto &element : bundleType) {
        generate(element.type);
      }
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      // Skip 0 length vectors.
      if (vecType.getNumElements() > 0) {
        generate(vecType.getElementType());
      }
      fieldID = save + vecType.getMaxFieldID();
    } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      FieldRef lhsFieldRef(lhs.getValue(), lhs.getFieldID() + fieldID);
      FieldRef rhsFieldRef(rhs.getValue(), rhs.getFieldID() + fieldID);
      Term t_src(1, rhsFieldRef);
      Constraint1 temp_c(lhsFieldRef, 0, {t_src}, std::nullopt);
      LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
      addConstraint(temp_c);
      fieldID++;
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  if (auto ftype = getBaseType(type))
    generate(ftype);
}

/// 递归获取 Value 的最底层定义 Operation
Operation *getRootDefiningOp(Value value) {
  Operation *currentOp = value.getDefiningOp();

  // 循环追溯直到找到底层操作
  while (currentOp != nullptr) {
    // 检查是否为需要继续追溯的操作类型
    if (auto subfield = llvm::dyn_cast<SubfieldOp>(currentOp)) {
      value = subfield.getInput();
      currentOp = value.getDefiningOp(); // 继续追溯
    } else if (auto subindex = llvm::dyn_cast<SubindexOp>(currentOp)) {
      value = subindex.getInput();
      currentOp = value.getDefiningOp();
    } else if (auto subaccess = llvm::dyn_cast<SubaccessOp>(currentOp)) {
      value = subaccess.getInput();
      currentOp = value.getDefiningOp();
    } else {
      break;
    }
  }
  return currentOp; // 可能是底层操作或 nullptr
}

/// generate constraints to ensure `larger` are greater than or equal to
/// `smaller`.
void InferenceMapping::generateConstraints(Value larger, Value smaller) {
  // Recurse to every leaf element and set larger >= smaller. Ignore foreign
  // types as these do not participate in width inference.

  auto fieldID = 0;
  bool is_invalid = false;
  Operation *rootOp = getRootDefiningOp(smaller);
  if (rootOp) {
    TypeSwitch<Operation *>(rootOp)
        .Case<InvalidValueOp>([&](auto) { is_invalid = true; })
        .Default([&](auto op) {});
  }

  std::function<void(FIRRTLBaseType, Value, Value)> generate =
      [&](FIRRTLBaseType type, Value larger, Value smaller) {
        // if (larger && smaller) {
        if (auto bundleType = type_dyn_cast<BundleType>(type)) {
          fieldID++;
          for (auto &element : bundleType.getElements()) {
            if (element.isFlip)
              generate(element.type, smaller, larger);
            else
              generate(element.type, larger, smaller);
          }
        } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
          fieldID++;
          auto save = fieldID;
          // Skip 0 length vectors.
          if (vecType.getNumElements() > 0) {
            generate(vecType.getElementType(), larger, smaller);
          }
          fieldID = save + vecType.getMaxFieldID();
          /*} else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
            constrainTypes(getExpr(FieldRef(larger, fieldID)),
                           getExpr(FieldRef(smaller, fieldID)), false, equal);
            fieldID++;*/
        } else if (type.isGround()) {
          // Leaf element, generate the constraint.
          FieldRef fieldRef_result(larger, fieldID);
          auto baseType_result =
              getBaseType(fieldRef_result.getValue().getType());
          auto type_result = hw::FieldIdImpl::getFinalTypeByFieldID(
              baseType_result, fieldRef_result.getFieldID());
          auto width_result =
              cast<FIRRTLBaseType>(type_result).getBitWidthOrSentinel();
          LLVM_DEBUG(llvm::dbgs() << "\nlooking at width_result : "
                                  << width_result << "\n\n");
          if (width_result >= 0) {
          } else if (width_result == -1) { // lhs 是uninferred，generate
            FieldRef fieldRef_input(smaller, fieldID);
            auto baseType_input =
                getBaseType(fieldRef_input.getValue().getType());
            auto type_input = hw::FieldIdImpl::getFinalTypeByFieldID(
                baseType_input, fieldRef_input.getFieldID());
            auto width_input =
                cast<FIRRTLBaseType>(type_input).getBitWidthOrSentinel();
            LLVM_DEBUG(llvm::dbgs() << "\nlooking at width_input : "
                                    << width_input << "\n\n");
            if (width_input >= 0) { // rhs已知
              Constraint1 temp_c(fieldRef_result, static_cast<int>(width_input),
                                 {}, std::nullopt);
              LLVM_DEBUG(llvm::dbgs()
                         << "build constraint1 : " << temp_c << "\n");
              addConstraint(temp_c);
            } else if (width_input ==
                       -1) { // rhs含有uninferred，则rhs作为变量生成不等式
              if (is_invalid) {
                // 要求invalid变量的位宽严格等于
                // 被声明为invalid的变量的位宽，所以同时添加 lhs>=rhs 和
                // rhs>=lhs
                LLVM_DEBUG(llvm::dbgs()
                           << "this one is connected to invalid\n");
                Term t_dest(1, fieldRef_result), t_src(1, fieldRef_input);
                Constraint1 c_dest(fieldRef_result, 0, {t_src}, std::nullopt),
                    c_src(fieldRef_input, 0, {t_dest}, std::nullopt);
                LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << c_dest
                                        << "\nand : " << c_src << "\n");
                addConstraint(c_src);
                addConstraint(c_dest);
              } else {
                Term t_src(1, fieldRef_input);
                Constraint1 temp_c(fieldRef_result, 0, {t_src}, std::nullopt);
                LLVM_DEBUG(llvm::dbgs()
                           << "build constraint1 : " << temp_c << "\n");
                addConstraint(temp_c);
              }
            }
          }
          fieldID++;
        } else {
          llvm_unreachable("Unknown type inside a bundle!");
        }
      };

  if (auto type = getBaseType(larger.getType()))
    generate(type, larger, smaller);
}

/// generate constraints of the fields in the `result` argument as
/// the max of `rhs` and `lhs`. Both fields must be the same type.
void InferenceMapping::generateConstraints(Value result, Value lhs, Value rhs) {
  // Recurse to every leaf element and set result >= rhs /\ result >= lhs.
  auto fieldID = 0;
  std::function<void(FIRRTLBaseType, Value, Value, Value)> generate =
      [&](FIRRTLBaseType type, Value result, Value lhs, Value rhs) {
        if (auto bundleType = type_dyn_cast<BundleType>(type)) {
          fieldID++;
          for (auto &element : bundleType.getElements()) {
            generate(element.type, result, lhs, rhs);
          }
        } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
          fieldID++;
          auto save = fieldID;
          // Skip 0 length vectors.
          if (vecType.getNumElements() > 0) {
            generate(vecType.getElementType(), result, lhs, rhs);
          }
          fieldID = save + vecType.getMaxFieldID();
          /*} else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
            constrainTypes(getExpr(FieldRef(larger, fieldID)),
                           getExpr(FieldRef(smaller, fieldID)), false, equal);
            fieldID++;*/
        } else if (type.isGround()) {
          // Leaf element, generate the constraint.
          FieldRef fieldRef_result(result, fieldID);
          auto baseType_result =
              getBaseType(fieldRef_result.getValue().getType());
          auto type_result = hw::FieldIdImpl::getFinalTypeByFieldID(
              baseType_result, fieldRef_result.getFieldID());
          auto width_result =
              cast<FIRRTLBaseType>(type_result).getBitWidthOrSentinel();
          if (width_result >= 0) {
          } else {
            // uninferred
            FieldRef fr_l(lhs, fieldID);
            auto baseType_l = getBaseType(fr_l.getValue().getType());
            auto type_l = hw::FieldIdImpl::getFinalTypeByFieldID(
                baseType_l, fr_l.getFieldID());
            auto wl = cast<FIRRTLBaseType>(type_l).getBitWidthOrSentinel();
            if (wl >= 0) { // lhs已知
              FieldRef fr_r(rhs, fieldID);
              auto baseType_r = getBaseType(fr_r.getValue().getType());
              auto type_r = hw::FieldIdImpl::getFinalTypeByFieldID(
                  baseType_r, fr_r.getFieldID());
              auto wr = cast<FIRRTLBaseType>(type_r).getBitWidthOrSentinel();
              if (wr >= 0) {         // rhs已知
              } else if (wr == -1) { // rhs 未知ground
                Term t_2(1, fr_r);
                Constraint1 temp_c1(fieldRef_result, static_cast<int>(wl), {},
                                    std::nullopt),
                    temp_c2(fieldRef_result, 0, {t_2}, std::nullopt);
                LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                        << "\nand : " << temp_c2 << "\n");
                addConstraint(temp_c1);
                // if (fr_l != fr_r)
                addConstraint(temp_c2);
              }
            } else if (wl == -1) { // lhs含有uninferred，则lhs作为变量生成不等式
              FieldRef fr_r(rhs, fieldID);
              auto baseType_r = getBaseType(fr_r.getValue().getType());
              auto type_r = hw::FieldIdImpl::getFinalTypeByFieldID(
                  baseType_r, fr_r.getFieldID());
              auto wr = cast<FIRRTLBaseType>(type_r).getBitWidthOrSentinel();
              if (wr >= 0) { // rhs已知
                Term t_1(1, fr_l);
                Constraint1 temp_c2(fieldRef_result, static_cast<int>(wr), {},
                                    std::nullopt),
                    temp_c1(fieldRef_result, 0, {t_1}, std::nullopt);
                LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                        << "\nand : " << temp_c2 << "\n");
                addConstraint(temp_c1);
                // if (fr_l != fr_r)
                addConstraint(temp_c2);
              } else if (wr == -1) { // rhs 未知ground
                Term t_1(1, fr_l), t_2(1, fr_r);
                Constraint1 temp_c1(fieldRef_result, 0, {t_1}, std::nullopt),
                    temp_c2(fieldRef_result, 0, {t_2}, std::nullopt);
                LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                        << "\nand : " << temp_c2 << "\n");
                addConstraint(temp_c1);
                // if (fr_l != fr_r)
                addConstraint(temp_c2);
              }
            }
          }
          fieldID++;
        } else {
          llvm_unreachable("Unknown type inside a bundle!");
        }
      };
  if (auto type = getBaseType(result.getType()))
    generate(type, result, lhs, rhs);
}

bool InferenceMapping::allWidthsKnown(Operation *op) {
  /// Ignore property assignments, no widths to infer.
  if (isa<PropAssignOp>(op))
    return true;

  // If this is a mux, and the select signal is uninferred, we need to set an
  // upperbound limit on it.
  if (isa<MuxPrimOp, Mux4CellIntrinsicOp, Mux2CellIntrinsicOp>(op))
    if (hasUninferredWidth(op->getOperand(0).getType()))
      return false;

  //  We need to propagate through connects.
  if (isa<FConnectLike, AttachOp>(op))
    return false;

  // Check if we know the width of every result of this operation.
  return llvm::all_of(op->getResults(), [&](auto result) {
    // Only consider FIRRTL types for width constraints. Ignore any foreign
    // types as they don't participate in the width inference process.
    if (auto type = type_dyn_cast<FIRRTLType>(result.getType()))
      if (hasUninferredWidth(type))
        return false;
    return true;
  });
}

LogicalResult InferenceMapping::extractConstraints(Operation *op) {
  if (allWidthsKnown(op))
    return success();

  bool mappingFailed = false;
  TypeSwitch<Operation *>(op)
      .Case<ConstantOp>([&](auto op) {
        // If the constant has a known width, use that. Otherwise pick the
        // smallest number of bits necessary to represent the constant.
        auto v = op.getValue();
        if (type_cast<FIRRTLType>(getBaseType(op.getResult().getType()))
                .hasUninferredWidth()) {
          auto w = v.getBitWidth() - (v.isNegative() ? v.countLeadingOnes()
                                                     : v.countLeadingZeros());
          if (v.isSigned())
            w += 1;
          FieldRef fr_dest(op.getResult(), 0);
          Constraint1 temp_c(fr_dest, std::max(w, 1u), {}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })
      .Case<InvalidValueOp>(
          [&](auto op) { generateConstraints(op.getResult()); })
      .Case<RegResetOp>([&](auto op) {
        generateConstraints(op.getResult(), op.getResetValue());
      })
      .Case<NodeOp>([&](auto op) {
        // Nodes have the same type as their input.
        generateConstraints(FieldRef(op.getResult(), 0),
                            FieldRef(op.getInput(), 0),
                            op.getResult().getType());
      })

      // Aggregate Values
      .Case<SubfieldOp>([&](auto op) {
        BundleType bundleType = op.getInput().getType();
        auto fieldID = bundleType.getFieldID(op.getFieldIndex());
        generateConstraints(FieldRef(op.getInput(), fieldID),
                            FieldRef(op.getResult(), 0), op.getType());
        generateConstraints(FieldRef(op.getResult(), 0),
                            FieldRef(op.getInput(), fieldID), op.getType());
      })
      .Case<SubindexOp, SubaccessOp>([&](auto op) {
        // All vec fields unify to the same thing. Always use the first element
        // of the vector, which has a field ID of 1.
        generateConstraints(FieldRef(op.getInput(), 1),
                            FieldRef(op.getResult(), 0), op.getType());
        generateConstraints(FieldRef(op.getResult(), 0),
                            FieldRef(op.getInput(), 1), op.getType());
      })
      .Case<RefSubOp>([&](RefSubOp op) {
        uint64_t fieldID = TypeSwitch<FIRRTLBaseType, uint64_t>(
                               op.getInput().getType().getType())
                               .Case<FVectorType>([](auto _) { return 1; })
                               .Case<BundleType>([&](auto type) {
                                 return type.getFieldID(op.getIndex());
                               });
        generateConstraints(FieldRef(op.getResult(), 0),
                            FieldRef(op.getInput(), fieldID), op.getType());
        generateConstraints(FieldRef(op.getInput(), fieldID),
                            FieldRef(op.getResult(), 0), op.getType());
      })

      // Arithmetic and Logical Binary Primitives
      .Case<AddPrimOp, SubPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getLhs().getType()))
                .hasUninferredWidth()) {
          if (type_cast<FIRRTLType>(getBaseType(op.getRhs().getType()))
                  .hasUninferredWidth()) {
            FieldRef fr_1(op.getLhs(), 0), fr_2(op.getRhs(), 0),
                fr_dest(op.getResult(), 0);
            Term t_1(1, fr_1), t_2(1, fr_2);
            Constraint1 temp_c1(fr_dest, 1, {t_1}, std::nullopt),
                temp_c2(fr_dest, 1, {t_2}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                    << "\nand : " << temp_c2 << "\n");
            addConstraint(temp_c1);
            // if (op.getLhs() != op.getRhs())
            addConstraint(temp_c2);
          } else {
            // rhs 已知，不使用变量名
            auto w = getBaseType(op.getRhs().getType()).getBitWidthOrSentinel();
            FieldRef fr_1(op.getLhs(), 0), fr_dest(op.getResult(), 0);
            Term t_1(1, fr_1);
            Constraint1 temp_c1(fr_dest, 1, {t_1}, std::nullopt),
                temp_c2(fr_dest, static_cast<int>(w) + 1, {}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                    << "\nand : " << temp_c2 << "\n");
            addConstraint(temp_c1);
            // if (op.getLhs() != op.getRhs())
            addConstraint(temp_c2);
          }
        } else { // lhs 已知，不使用变量名
          if (type_cast<FIRRTLType>(getBaseType(op.getRhs().getType()))
                  .hasUninferredWidth()) {
            auto w = getBaseType(op.getLhs().getType()).getBitWidthOrSentinel();
            FieldRef fr_2(op.getRhs(), 0), fr_dest(op.getResult(), 0);
            Term t_2(1, fr_2);
            Constraint1 temp_c1(fr_dest, static_cast<int>(w) + 1, {}, {}),
                temp_c2(fr_dest, 1, {t_2}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                    << "\nand : " << temp_c2 << "\n");
            addConstraint(temp_c1);
            // if (op.getLhs() != op.getRhs())
            addConstraint(temp_c2);
          } /*else {
            // lhs和rhs 已知，不使用变量名
            auto w0 =
          getBaseType(op.getLhs().getType()).getBitWidthOrSentinel(); auto w1 =
          getBaseType(op.getRhs().getType()).getBitWidthOrSentinel(); FieldRef
          fr_dest(op.getResult(), 0); Constraint1 temp_c1(fr_dest,
          static_cast<int>(w0) + 1, {}, {}), temp_c2(fr_dest,
          static_cast<int>(w1) + 1, {}, std::nullopt); LLVM_DEBUG(llvm::dbgs()
          << "build constraint1 : " << temp_c1 << "\nand : " << temp_c2 <<
          "\n");
          }*/
        }
      })
      .Case<MulPrimOp>([&](auto op) {
        auto w0 = getBaseType(op.getLhs().getType()).getBitWidthOrSentinel();
        if (w0 >= 0) { // lhs 已知，不使用变量名
          auto w1 = getBaseType(op.getRhs().getType()).getBitWidthOrSentinel();
          if (w1 >= 0) {         // rhs 已知，不使用变量名
          } else if (w1 == -1) { // rhs未知，使用变量名
            FieldRef fr_2(op.getRhs(), 0), fr_dest(op.getResult(), 0);
            Term t_2(1, fr_2);
            Constraint1 temp_c(fr_dest, static_cast<int>(w0), {t_2},
                               std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          }
        } else if (w0 == -1) { // lhs未知，使用变量名
          auto w1 = getBaseType(op.getRhs().getType()).getBitWidthOrSentinel();
          if (w1 >= 0) { // rhs 已知，不使用变量名
            FieldRef fr_1(op.getLhs(), 0), fr_dest(op.getResult(), 0);
            Term t_1(1, fr_1);
            Constraint1 temp_c(fr_dest, static_cast<int>(w1), {t_1},
                               std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          } else if (w1 == -1) { // rhs未知，使用变量名
            FieldRef fr_1(op.getLhs(), 0), fr_2(op.getRhs(), 0),
                fr_dest(op.getResult(), 0);
            Term t_1(1, fr_1), t_2(1, fr_2);
            Terms result_t = {t_1};
            result_t = result_t.combine_term(t_2);
            Constraint1 temp_c(fr_dest, 0, result_t, std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          }
        }
      })
      .Case<DivPrimOp>([&](auto op) {
        auto w = getBaseType(op.getLhs().getType()).getBitWidthOrSentinel();
        if (w >= 0) {
        } else if (w == -1) { // lhs未知，使用变量名
          FieldRef fr_1(op.getLhs(), 0), fr_dest(op.getResult(), 0);
          Term t_1(1, fr_1);
          if (op.getType().base().isSigned()) {
            Constraint1 temp_c(fr_dest, 1, {t_1}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          } else {
            Constraint1 temp_c(fr_dest, 0, {t_1}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          }
        }
      })
      .Case<RemPrimOp>([&](auto op) {
        auto w0 = getBaseType(op.getLhs().getType()).getBitWidthOrSentinel();
        if (w0 >= 0) { // lhs 已知，不使用变量名
          auto w1 = getBaseType(op.getRhs().getType()).getBitWidthOrSentinel();
          if (w1 >= 0) {         // rhs 已知，不使用变量名
          } else if (w1 == -1) { // rhs未知，使用变量名
            FieldRef fr_2(op.getRhs(), 0), fr_dest(op.getResult(), 0);
            Constraint_Min temp_c(fr_dest, static_cast<int>(w0), 0,
                                  std::nullopt, fr_2);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint_min : " << temp_c << "\n");
            addConstraint(temp_c);
          }
        } else if (w0 == -1) { // lhs未知，使用变量名
          auto w1 = getBaseType(op.getRhs().getType()).getBitWidthOrSentinel();
          if (w1 >= 0) { // rhs 已知，不使用变量名
            FieldRef fr_1(op.getLhs(), 0), fr_dest(op.getResult(), 0);
            Constraint_Min temp_c(fr_dest, 0, static_cast<int>(w1), fr_1,
                                  std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint_min : " << temp_c << "\n");
            addConstraint(temp_c);
          } else if (w1 == -1) { // rhs未知，使用变量名
            FieldRef fr_1(op.getLhs(), 0), fr_2(op.getRhs(), 0),
                fr_dest(op.getResult(), 0);
            Constraint_Min temp_c(fr_dest, 0, 0, fr_1, fr_2);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint_min : " << temp_c << "\n");
            addConstraint(temp_c);
          }
        }
      })
      .Case<AndPrimOp, OrPrimOp, XorPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getLhs().getType()))
                .hasUninferredWidth()) {
          if (type_cast<FIRRTLType>(getBaseType(op.getRhs().getType()))
                  .hasUninferredWidth()) {
            FieldRef fr_1(op.getLhs(), 0), fr_2(op.getRhs(), 0),
                fr_dest(op.getResult(), 0);
            Term t_1(1, fr_1), t_2(1, fr_2);
            Constraint1 temp_c1(fr_dest, 0, {t_1}, std::nullopt),
                temp_c2(fr_dest, 0, {t_2}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                    << "\nand : " << temp_c2 << "\n");
            addConstraint(temp_c1);
            // if (op.getLhs() != op.getRhs())
            addConstraint(temp_c2);
          } else {
            // rhs 已知，不使用变量名
            auto w = getBaseType(op.getRhs().getType()).getBitWidthOrSentinel();
            FieldRef fr_1(op.getLhs(), 0), fr_dest(op.getResult(), 0);
            Term t_1(1, fr_1);
            Constraint1 temp_c1(fr_dest, 0, {t_1}, std::nullopt),
                temp_c2(fr_dest, static_cast<int>(w), {}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                    << "\nand : " << temp_c2 << "\n");
            addConstraint(temp_c1);
            // if (op.getLhs() != op.getRhs())
            addConstraint(temp_c2);
          }
        } else { // lhs 已知，不使用变量名
          if (type_cast<FIRRTLType>(getBaseType(op.getRhs().getType()))
                  .hasUninferredWidth()) {
            auto w = getBaseType(op.getLhs().getType()).getBitWidthOrSentinel();
            FieldRef fr_2(op.getRhs(), 0), fr_dest(op.getResult(), 0);
            Term t_2(1, fr_2);
            Constraint1 temp_c1(fr_dest, static_cast<int>(w), {}, {}),
                temp_c2(fr_dest, 0, {t_2}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                    << "\nand : " << temp_c2 << "\n");
            addConstraint(temp_c1);
            // if (op.getLhs() != op.getRhs())
            addConstraint(temp_c2);
          }
        }
      })
      .Case<CatPrimOp>([&](auto op) {
        FieldRef fr_dest(op.getResult(), 0);
        if (op.getInputs().empty()) {
          Constraint1 temp_c(fr_dest, 0, {}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        } else {
          auto w = getBaseType(op.getInputs().front().getType())
                       .getBitWidthOrSentinel();
          if (w >= 0) { // 第一项为已知
            Terms result_t = {};
            int result_cst = static_cast<int>(w);
            for (Value operand : op.getInputs().drop_front()) {
              auto w0 = getBaseType(operand.getType()).getBitWidthOrSentinel();
              if (w0 >= 0) // 当前项为已知
                result_cst += static_cast<int>(w0);
              else // 当前项未知
              {
                FieldRef fr_2(operand, 0);
                Term t_2(1, fr_2);
                result_t = result_t.combine_term(t_2);
              }
            }
            Constraint1 temp_c(fr_dest, result_cst, result_t, std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          } else {
            FieldRef fr_1(op.getInputs().front(), 0);
            Term t_1(1, fr_1);
            Terms result_t = {t_1};
            int result_cst = 0;
            for (Value operand : op.getInputs().drop_front()) {
              auto w0 = getBaseType(operand.getType()).getBitWidthOrSentinel();
              if (w0 >= 0) // 当前项为已知
                result_cst += static_cast<int>(w0);
              else // 当前项未知
              {
                FieldRef fr_2(operand, 0);
                Term t_2(1, fr_2);
                result_t = result_t.combine_term(t_2);
              }
            }
            Constraint1 temp_c(fr_dest, result_cst, result_t, std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          }
        }
      })
      // Misc Binary Primitives
      .Case<DShlPrimOp>([&](auto op) {
        auto w0 = getBaseType(op.getLhs().getType()).getBitWidthOrSentinel();
        if (w0 >= 0) { // lhs 已知，不使用变量名
          auto w1 = getBaseType(op.getRhs().getType()).getBitWidthOrSentinel();
          if (w1 >= 0) {         // rhs 已知，不使用变量名
          } else if (w1 == -1) { // rhs未知，使用变量名
            FieldRef fr_2(op.getRhs(), 0), fr_dest(op.getResult(), 0);
            Constraint1 temp_c(fr_dest, static_cast<int>(w0) - 1, {}, fr_2);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          }
        } else if (w0 == -1) { // lhs未知，使用变量名
          auto w1 = getBaseType(op.getRhs().getType()).getBitWidthOrSentinel();
          if (w1 >= 0) { // rhs 已知，不使用变量名
            FieldRef fr_1(op.getLhs(), 0), fr_dest(op.getResult(), 0);
            Term t_1(1, fr_1);
            Constraint1 temp_c(fr_dest, (1 << static_cast<int>(w1)) - 1, {t_1},
                               std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          } else if (w1 == -1) { // rhs未知，使用变量名
            FieldRef fr_1(op.getLhs(), 0), fr_2(op.getRhs(), 0),
                fr_dest(op.getResult(), 0);
            Term t_1(1, fr_1);
            Constraint1 temp_c(fr_dest, -1, {t_1}, fr_2);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          }
        }
      })
      .Case<DShlwPrimOp, DShrPrimOp>([&](auto op) {
        auto w = getBaseType(op.getLhs().getType()).getBitWidthOrSentinel();
        if (w >= 0) {         // rhs 已知，不使用变量名
        } else if (w == -1) { // rhs未知，使用变量名
          FieldRef fr_1(op.getLhs(), 0), fr_dest(op.getResult(), 0);
          Term t_1(1, fr_1);
          Constraint1 temp_c(fr_dest, 0, {t_1}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })

      // Unary operators
      .Case<NegPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getType()))
                .hasUninferredWidth()) {
          FieldRef fr_src(op.getInput(), 0), fr_dest(op.getResult(), 0);
          Term t_src(1, fr_src);
          Constraint1 temp_c(fr_dest, 1, {t_src}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })
      .Case<CvtPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getType()))
                .hasUninferredWidth()) {
          FieldRef fr_src(op.getInput(), 0), fr_dest(op.getResult(), 0);
          Term t_src(1, fr_src);
          if (op.getInput().getType().base().isSigned()) {
            Constraint1 temp_c(fr_dest, 0, {t_src}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          } else {
            Constraint1 temp_c(fr_dest, 1, {t_src}, std::nullopt);
            LLVM_DEBUG(llvm::dbgs()
                       << "build constraint1 : " << temp_c << "\n");
            addConstraint(temp_c);
          }
        }
      })

      // Miscellaneous
      .Case<BitsPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getType()))
                .hasUninferredWidth()) {
          FieldRef fr_dest(op.getResult(), 0);
          Constraint1 temp_c(fr_dest, op.getHi() - op.getLo() + 1, {},
                             std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })
      .Case<HeadPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getType()))
                .hasUninferredWidth()) {
          FieldRef fr_dest(op.getResult(), 0);
          Constraint1 temp_c(fr_dest, op.getAmount(), {}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })
      .Case<TailPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getType()))
                .hasUninferredWidth()) {
          FieldRef fr_src(op.getInput(), 0), fr_dest(op.getResult(), 0);
          Term t_src(1, fr_src);
          Constraint1 temp_c(fr_dest, -op.getAmount(), {t_src}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })
      .Case<PadPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getType()))
                .hasUninferredWidth()) {
          FieldRef fr_src(op.getInput(), 0), fr_dest(op.getResult(), 0);
          Term t_src(1, fr_src);
          Constraint1 temp_c1(fr_dest, 0, {t_src}, std::nullopt),
              temp_c2(fr_dest, op.getAmount(), {}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                  << "\nand : " << temp_c2 << "\n");
          addConstraint(temp_c1);
          addConstraint(temp_c2);
        }
      })
      .Case<ShlPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getType()))
                .hasUninferredWidth()) {
          FieldRef fr_src(op.getInput(), 0), fr_dest(op.getResult(), 0);
          Term t_src(1, fr_src);
          Constraint1 temp_c(fr_dest, op.getAmount(), {t_src}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })
      .Case<ShrPrimOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getType()))
                .hasUninferredWidth()) {
          FieldRef fr_src(op.getInput(), 0), fr_dest(op.getResult(), 0);
          Term t_src(1, fr_src);
          // UInt saturates at 0 bits, SInt at 1 bit
          auto minWidth = op.getInput().getType().base().isUnsigned() ? 0 : 1;
          Constraint1 temp_c1(fr_dest, -op.getAmount(), {t_src}, std::nullopt),
              temp_c2(fr_dest, minWidth, {}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c1
                                  << "\nand : " << temp_c2 << "\n");
          addConstraint(temp_c1);
          addConstraint(temp_c2);
        }
      })

      // Handle operations whose output width matches the input width.
      .Case<NotPrimOp, AsSIntPrimOp, AsUIntPrimOp, ConstCastOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getType()))
                .hasUninferredWidth()) {
          FieldRef fr_src(op.getInput(), 0), fr_dest(op.getResult(), 0);
          Term t_src(1, fr_src);
          Constraint1 temp_c(fr_dest, 0, {t_src}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })
      .Case<mlir::UnrealizedConversionCastOp>([&](auto op) {
        if (type_cast<FIRRTLType>(getBaseType(op.getResult(0).getType()))
                .hasUninferredWidth()) {
          FieldRef fr_src(op.getOperand(0), 0), fr_dest(op.getResult(0), 0);
          Term t_src(1, fr_src);
          Constraint1 temp_c(fr_dest, 0, {t_src}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })

      // Handle operations with a single result type that always has a
      // well-known width.
      .Case<LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp, EQPrimOp, NEQPrimOp,
            AsClockPrimOp, AsAsyncResetPrimOp, AndRPrimOp, OrRPrimOp,
            XorRPrimOp>([&](auto op) {
        auto w = getBaseType(op.getType()).getBitWidthOrSentinel();
        if (w >= 0) {
        } else {
          FieldRef fr_dest(op.getResult(), 0);
          Constraint1 temp_c(fr_dest, static_cast<int>(w), {}, std::nullopt);
          LLVM_DEBUG(llvm::dbgs() << "build constraint1 : " << temp_c << "\n");
          addConstraint(temp_c);
        }
      })
      .Case<MuxPrimOp, Mux2CellIntrinsicOp>([&](auto op) {
        // auto *sel = getExpr(op.getSel());
        // constrainTypes(solver.known(1), sel, true); TBD : select signal <= 1
        generateConstraints(op.getResult(), op.getHigh(), op.getLow());
      })
      .Case<Mux4CellIntrinsicOp>([&](Mux4CellIntrinsicOp op) {
        // auto *sel = getExpr(op.getSel());
        // constrainTypes(solver.known(2), sel, true);
        generateConstraints(op.getResult(), op.getV3(), op.getV2());
        generateConstraints(op.getResult(), op.getV0(), op.getV1());
      })

      .Case<ConnectOp, MatchingConnectOp>(
          [&](auto op) { generateConstraints(op.getDest(), op.getSrc()); })
      .Case<RefDefineOp>([&](auto op) {
        // Dest >= Src and Src <= Dest
        generateConstraints(op.getDest(), op.getSrc());
        generateConstraints(op.getSrc(), op.getDest());
      })
      .Case<AttachOp>([&](auto op) {
        // Attach connects multiple analog signals together. All signals must
        // have the same bit width. Signals without bit width inherit from the
        // other signals.
        if (op.getAttached().empty())
          return;
        auto prev = op.getAttached()[0];
        for (auto operand : op.getAttached().drop_front()) {
          generateConstraints(prev, operand);
          generateConstraints(operand, prev);
          prev = operand;
        }
      })

      // Handle the no-ops that don't interact with width inference.
      .Case<PrintFOp, FFlushOp, SkipOp, StopOp, WhenOp, AssertOp, AssumeOp,
            UnclockedAssumeIntrinsicOp, CoverOp>([&](auto) {})

      // Handle instances of other modules.
      .Case<InstanceOp>([&](auto op) {
        auto refdModule = op.getReferencedOperation(symtbl);
        auto module = dyn_cast<FModuleOp>(&*refdModule);
        if (!module) {
          auto diag = mlir::emitError(op.getLoc());
          diag << "extern module `" << op.getModuleName()
               << "` has ports of uninferred width";

          auto fml = cast<FModuleLike>(&*refdModule);
          auto ports = fml.getPorts();
          for (auto &port : ports) {
            auto baseType = getBaseType(port.type);
            if (baseType && baseType.hasUninferredWidth()) {
              diag.attachNote(op.getLoc()) << "Port: " << port.name;
              if (!baseType.isGround())
                diagnoseUninferredType(diag, baseType, port.name.getValue());
            }
          }

          diag.attachNote(op.getLoc())
              << "Only non-extern FIRRTL modules may contain unspecified "
                 "widths to be inferred automatically.";
          diag.attachNote(refdModule->getLoc())
              << "Module `" << op.getModuleName() << "` defined here:";
          mappingFailed = true;
          return;
        }
        // Simply look up the free variables created for the instantiated
        // module's ports, and use them for instance port wires. This way,
        // constraints imposed onto the ports of the instance will transparently
        // apply to the ports of the instantiated module.
        for (auto [result, arg] :
             llvm::zip(op->getResults(), module.getArguments())) {
          generateConstraints(
              {result, 0},
              {arg, 0}, // 这里是产生FieldRef, 相当于把instance端口看作bundle
              type_cast<FIRRTLType>(result.getType()));
          generateConstraints(
              {arg, 0},
              {result, 0}, // 这里是产生FieldRef, 相当于把instance端口看作bundle
              type_cast<FIRRTLType>(result.getType()));
        }
      })

      .Case<MemOp>([&](MemOp op) {
        // 记录第一个非debug端口
        unsigned nonDebugPort = 0;
        for (const auto &result : llvm::enumerate(op.getResults())) {
          if (!type_isa<RefType>(result.value().getType()))
            nonDebugPort = result.index();
        }

        // A helper function that returns the indeces of the "data", "rdata",
        // and "wdata" fields in the bundle corresponding to a memory port.
        auto dataFieldIndices = [](MemOp::PortKind kind) -> ArrayRef<unsigned> {
          static const unsigned indices[] = {3, 5};
          static const unsigned debug[] = {0};
          switch (kind) {
          case MemOp::PortKind::Read:
          case MemOp::PortKind::Write:
            return ArrayRef<unsigned>(indices, 1); // {3}
          case MemOp::PortKind::ReadWrite:
            return ArrayRef<unsigned>(indices); // {3, 5}
          case MemOp::PortKind::Debug:
            return ArrayRef<unsigned>(debug);
          }
          llvm_unreachable("Imposible PortKind");
        };

        // This creates independent variables for every data port. Yet, what we
        // actually want is for all data ports to share the same variable. To do
        // this, we find the first data port declared, and use that port's vars
        // for all the other ports.
        unsigned firstFieldIndex =
            dataFieldIndices(op.getPortKind(nonDebugPort))[0];
        FieldRef firstData(
            op.getResult(nonDebugPort),
            type_cast<BundleType>(op.getPortType(nonDebugPort).getPassiveType())
                .getFieldID(firstFieldIndex));
        LLVM_DEBUG(llvm::dbgs() << "Adjusting memory port variables:\n");

        // Reuse data port variables.
        auto dataType = op.getDataType();
        for (unsigned i = 0, e = op.getResults().size(); i < e; ++i) {
          auto result = op.getResult(i);
          if (type_isa<RefType>(result.getType())) {
            // Debug ports are firrtl.ref<vector<data-type, depth>>
            // Use FieldRef of 1, to indicate the first vector element must be
            // of the dataType.
            generateConstraints(FieldRef(result, 1), firstData, dataType);
            generateConstraints(firstData, FieldRef(result, 1), dataType);
            continue;
          }

          auto portType =
              type_cast<BundleType>(op.getPortType(i).getPassiveType());
          for (auto fieldIndex : dataFieldIndices(op.getPortKind(i))) {
            generateConstraints(
                FieldRef(result, portType.getFieldID(fieldIndex)), firstData,
                dataType);
            generateConstraints(
                firstData, FieldRef(result, portType.getFieldID(fieldIndex)),
                dataType);
          }
        }
      })

      .Case<RefSendOp>([&](auto op) {
        generateConstraints(op.getBase(), op.getResult());
        generateConstraints(op.getResult(), op.getBase());
      })
      .Case<RefResolveOp>([&](auto op) {
        generateConstraints(op.getResult(), op.getRef());
        generateConstraints(op.getRef(), op.getResult());
      })
      .Case<RefCastOp>([&](auto op) {
        generateConstraints(op.getInput(), op.getResult());
        generateConstraints(op.getResult(), op.getInput());
      })
      .Case<RWProbeOp>([&](auto op) {
        auto ist = irn.lookup(op.getTarget());
        if (!ist) {
          op->emitError("target of rwprobe could not be resolved");
          mappingFailed = true;
          return;
        }
        auto ref = getFieldRefForTarget(ist);
        if (!ref) {
          op->emitError("target of rwprobe resolved to unsupported target");
          mappingFailed = true;
          return;
        }
        auto newFID = convertFieldIDToOurVersion(
            ref.getFieldID(), type_cast<FIRRTLType>(ref.getValue().getType()));
        generateConstraints(FieldRef(op.getResult(), 0),
                            FieldRef(ref.getValue(), newFID), op.getType());
      })
      .Case<SpecialConstantOp, WireOp, RegOp>([&](auto op) {
        // Nothing required.
      })
      .Default([&](auto op) {
        op->emitOpError("not supported in width inference");
        mappingFailed = true;
      });

  // Forceable declarations should have the ref constrained to data result.
  if (auto fop = dyn_cast<Forceable>(op); fop && fop.isForceable())
    generateConstraints(FieldRef(fop.getDataRef(), 0),
                        FieldRef(fop.getDataRaw(), 0), fop.getDataType());

  return failure(mappingFailed);
}

//===----------------------------------------------------------------------===//
// floyd
//===----------------------------------------------------------------------===//
/*
// 节点类型
using Node = std::variant<Zero, FieldRef>;

// 为节点类型定义输出运算符
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Node &node) {
  if (std::holds_alternative<Zero>(node)) {
    os << "Zero";
  } else {
    auto &pv = std::get<FieldRef>(node);
    os << "FieldRef[" << pv.getValue() << " (fieldID: " << pv.getFieldID()
       << ")]";
  }
  return os;
}

// 节点哈希函数
struct NodeHash {
  size_t operator()(const Node &node) const {
    if (std::holds_alternative<Zero>(node)) {
      return 0; // 所有Zero实例具有相同的哈希值
    } else {
      auto &pv = std::get<FieldRef>(node);
      return static_cast<size_t>(hash_value(pv));
    }
  }
};

// 定义图类型
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                                    boost::property<boost::vertex_name_t, Node>,
                                    boost::property<boost::edge_weight_t, int>>;

using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Edge = boost::graph_traits<Graph>::edge_descriptor;

struct GraphResult {
  Graph g;
  std::unordered_map<Node, Vertex, NodeHash> node_to_vertex;
  std::vector<Node> nodes_in_order;
};

// 构建约束图的函数
GraphResult
build_constraint_graph(const std::vector<Constraint1> &constraints) {
  GraphResult result;

  // 添加节点的辅助函数
  auto add_node = [&](const Node &node) -> Vertex {
    auto it = result.node_to_vertex.find(node);
    if (it != result.node_to_vertex.end()) {
      return it->second;
    }
    Vertex v = add_vertex(node, result.g);
    result.node_to_vertex[node] = v;
    result.nodes_in_order.push_back(node);
    return v;
  };

  // 确保Zero节点存在
  Node zero_node = Zero{};
  Vertex v_zero = add_node(zero_node);

  auto weight_map = get(boost::edge_weight, result.g);
  auto add_edge = [&](Vertex u, Vertex v, int weight) {
    auto [edge, added] = boost::add_edge(u, v, result.g);
    if (added) {
      weight_map[edge] = weight;
    }
    return edge;
  };

  for (const auto &constraint : constraints) {
    // 获取源节点（左侧变量）
    Node source_node = constraint.lhs_var1();
    Vertex source_vertex = add_node(source_node);
    Vertex target_vertex;
    if (constraint.rhs_terms1().empty()) {
      // 右侧没有项，指向Zero节点
      target_vertex = v_zero;
    } else {
      // 指向右侧第一个项的变量
      Node target_node = constraint.rhs_terms1().begin()->var();
      target_vertex = add_node(target_node);
    }

    // 添加边（权重取负）
    add_edge(source_vertex, target_vertex, -constraint.rhs_const1());
  }

  return result;
}

Valuation my_floyd(const std::vector<Constraint1> &constraints) {
  // int i = 1;
  // for (const auto& c : constraints) {
  //     LLVM_DEBUG(llvm::dbgs() << "约束" << i++ << ": " << c << "\n");
  // }

  // 构建图
  GraphResult result = build_constraint_graph(constraints);
  // print_graph_edges(result.g);
  auto weight_map = get(boost::edge_weight, result.g);

  // 准备距离矩阵
  size_t vertex_count = num_vertices(result.g);
  using DistanceMatrix = std::vector<std::vector<int>>;
  DistanceMatrix dist(
      vertex_count,
      std::vector<int>(vertex_count, std::numeric_limits<int>::max()));

  // 初始化对角线为0
  for (size_t i = 0; i < vertex_count; ++i) {
    dist[i][i] = 0;
  }

  // Floyd-Warshall算法
  bool has_negative_cycle = false;
  has_negative_cycle = !boost::floyd_warshall_all_pairs_shortest_paths(
      result.g, dist, boost::weight_map(weight_map));

  LLVM_DEBUG(llvm::dbgs() << "图中顶点数量: " << vertex_count << "\n");

  if (has_negative_cycle) {
    LLVM_DEBUG(llvm::dbgs() << "\n警告：图中存在负权环！结果可能无效\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "\n图中不存在负权环，计算有效\n");
  }

  // 分析每个节点的最短路径
  for (size_t i = 0; i < vertex_count; ++i) {
    Vertex source_vertex = result.node_to_vertex[result.nodes_in_order[i]];
    Node source_node = result.nodes_in_order[i];

    int min_distance = std::numeric_limits<int>::max();
    size_t min_target_index = vertex_count;

    for (size_t j = 0; j < vertex_count; ++j) {
      if (i == j)
        continue;

      Vertex target_vertex = result.node_to_vertex[result.nodes_in_order[j]];
      int d = dist[source_vertex][target_vertex];

      if (d != std::numeric_limits<int>::max()) {

        // 更新最短路径
        if (d < min_distance) {
          min_distance = d;
          min_target_index = j;
        }
      }
    }

    // 输出当前节点的分析结果
    LLVM_DEBUG(llvm::dbgs() << "源节点: " << source_node << "\n");

    if (min_target_index < vertex_count) {
      Node min_target_node = result.nodes_in_order[min_target_index];
      LLVM_DEBUG(llvm::dbgs()
                 << "最短路径: " << source_node << " → " << min_target_node
                 << " (距离: " << min_distance << ")\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "未找到到其他节点的路径\n");
    }

    // 输出当前节点到所有节点的路径详情
    LLVM_DEBUG(llvm::dbgs() << "路径详情:\n");
    for (size_t j = 0; j < vertex_count; ++j) {
      if (i == j)
        continue;

      Vertex target_vertex = result.node_to_vertex[result.nodes_in_order[j]];
      Node target_node = result.nodes_in_order[j];
      int d = dist[source_vertex][target_vertex];

      LLVM_DEBUG(llvm::dbgs() << "  -> " << target_node << ": ");
      if (d == std::numeric_limits<int>::max()) {
        LLVM_DEBUG(llvm::dbgs() << "不可达");
      } else {
        LLVM_DEBUG(llvm::dbgs() << d);
        if (j == min_target_index) {
          LLVM_DEBUG(llvm::dbgs() << " (最短)");
        }
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
    }
  }

  Valuation valuation;

  // 分析每个节点的最短路径
  for (size_t i = 0; i < vertex_count; ++i) {
    Vertex source_vertex = result.node_to_vertex[result.nodes_in_order[i]];
    Node source_node = result.nodes_in_order[i];

    // 仅处理非Zero节点
    if (std::holds_alternative<FieldRef>(source_node)) {
      FieldRef pv = std::get<FieldRef>(source_node);
      int min_distance = std::numeric_limits<int>::max();

      for (size_t j = 0; j < vertex_count; ++j) {
        if (i == j)
          continue;

        Vertex target_vertex = result.node_to_vertex[result.nodes_in_order[j]];
        int d = dist[source_vertex][target_vertex];

        // 更新最短路径值
        if (d != std::numeric_limits<int>::max() && d < min_distance) {
          min_distance = d;
        }
      }

      // 将结果存入valuation
      if (min_distance != std::numeric_limits<int>::max()) {
        valuation[pv] = -min_distance;
      } else {
        valuation[pv] = 0; // 或使用特殊值表示不可达
      }
    }
  }

  return valuation;
}*/

//===----------------------------------------------------------------------===//
// upper_bound
//===----------------------------------------------------------------------===//

// DFS访问器类 (记录前驱并在找到终点时终止)
/*template <typename DependencyGraph>
class dfs_visitor : public boost::default_dfs_visitor {
public:
  dfs_visitor(
      std::vector<typename DependencyGraph::vertex_descriptor> &predecessor,
      typename DependencyGraph::vertex_descriptor end, bool &found)
      : predecessor_(predecessor), end_(end), found_(found) {}

  // 当边被加入DFS树时调用
  template <typename Edge, typename DependencyGraph_t>
  void tree_edge(Edge e, DependencyGraph_t &g) {
    if (found_)
      return; // 已找到终点，跳过处理

    auto u = boost::source(e, g);
    auto v = boost::target(e, g);
    predecessor_[v] = u; // 记录前驱关系

    // 若找到终点则抛出异常终止DFS
    if (v == end_) {
      found_ = true; // 设置终止标志
    }
  }

private:
  std::vector<typename DependencyGraph::vertex_descriptor> &predecessor_;
  typename DependencyGraph::vertex_descriptor end_;
  bool &found_; // 标志位
};

// 查找路径的函数
template <typename DependencyGraph>
std::vector<typename DependencyGraph::vertex_property_type> // 返回Node类型向量
find_path(const DependencyGraph &g,
          typename DependencyGraph::vertex_descriptor start,
          typename DependencyGraph::vertex_descriptor end) {

  using vertex_descriptor = typename DependencyGraph::vertex_descriptor;
  using Node = typename DependencyGraph::vertex_property_type;

  // 处理起点和终点相同的情况
  if (start == end) {
    return {g[start]}; // 直接返回包含单个Node的向量
  }

  // 初始化前驱映射
  size_t num_vertices = boost::num_vertices(g);
  std::vector<vertex_descriptor> predecessor(
      num_vertices, boost::graph_traits<DependencyGraph>::null_vertex());

  // 创建DFS访问器
  bool found = false;
  dfs_visitor<DependencyGraph> vis(predecessor, end, found);

  // 记录顶点访问状态
  std::vector<boost::default_color_type> color_map(num_vertices,
                                                   boost::white_color);

  // DFS
  boost::depth_first_search(
      g, boost::visitor(vis)
             .color_map(boost::make_iterator_property_map(
                 color_map.begin(), boost::get(boost::vertex_index, g)))
             .root_vertex(start));

  // 检查终点是否可达
  if (predecessor[end] == boost::graph_traits<DependencyGraph>::null_vertex()) {
    return {}; // 返回空路径表示不可达
  }

  // 回溯
  std::vector<vertex_descriptor> vertex_path;
  vertex_descriptor current = end;
  while (current != boost::graph_traits<DependencyGraph>::null_vertex()) {
    vertex_path.push_back(current);
    current = predecessor[current];
  }
  std::reverse(vertex_path.begin(), vertex_path.end());

  std::vector<Node> node_path;
  node_path.reserve(vertex_path.size());
  for (auto vd : vertex_path) {
    node_path.push_back(g[vd]);
  }

  return node_path;
}

struct DependencyGraphResult {
  DependencyGraph g;
  std::unordered_map<FieldRef,
                     boost::graph_traits<DependencyGraph>::vertex_descriptor,
                     FieldRef::Hash>
      vertex_map_;
};

// 构建依赖关系图
DependencyGraphResult
buildDependencyGraph(const std::vector<Constraint1> &constraints_) {
  DependencyGraphResult result;

  // 添加节点的辅助函数
  auto get_or_add_vertex = [&](const FieldRef &node)
      -> boost::graph_traits<DependencyGraph>::vertex_descriptor {
    auto it = result.vertex_map_.find(node);
    if (it != result.vertex_map_.end()) {
      return it->second;
    } else {
      auto vertex = boost::add_vertex(node, result.g);
      result.vertex_map_[node] = vertex;
      return vertex;
    };
  };

  // 遍历所有约束
  for (const auto &c : constraints_) {
    FieldRef lhs = c.lhs_var1();
    auto lhs_vertex = get_or_add_vertex(lhs);

    // 遍历右侧项
    for (const auto &term : c.rhs_terms1()) {
      FieldRef rhs_var = term.var();
      auto rhs_vertex = get_or_add_vertex(rhs_var);
      boost::add_edge(lhs_vertex, rhs_vertex, result.g);
    }
  }
  return result;
}*/

std::vector<Node*> findPathBetween(Node* start, Node* end) {
        if (!start || !end) return {};
        if (start == end) return {start};  // 相同节点

        // DFS数据结构
        struct StackFrame {
            Node* node;
            size_t nextChildIndex;
        };
        std::stack<StackFrame> stack;
        std::unordered_set<Node*> visited;
        std::vector<Node*> path;

        // 初始化
        stack.push({start, 0});
        path.push_back(start);
        visited.insert(start);

        while (!stack.empty()) {
            auto& current = stack.top();
            
            // 到达目标节点
            if (current.node == end) {
                return path;
            }

            // 处理子节点
            if (current.nextChildIndex < current.node->successors.size()) {
                Node* nextChild = current.node->successors[current.nextChildIndex];
                current.nextChildIndex++;
                
                if (!visited.count(nextChild)) {
                    visited.insert(nextChild);
                    path.push_back(nextChild);
                    stack.push({nextChild, 0});
                }
            } 
            // 回溯
            else {
                stack.pop();
                path.pop_back();
            }
        }
        
        return {}; // 未找到路径
    }

static bool termsContains(const Terms &terms, const FieldRef &var) {
  for (const auto &term : terms) {
    if (term.var() == var) {
      return true;
    }
  }
  return false;
}

std::vector<Constraint1>
orderConstraints(const std::vector<FieldRef> &vars,
                 const std::vector<Constraint1> &constraints) {
  std::vector<Constraint1> result;
  if (vars.size() < 2) {
    return result; // 至少需要两个变量才能形成一对
  }

  // 标记已收集的约束索引
  std::vector<bool> used(constraints.size(), false);

  // 遍历变量序列（从第一个到倒数第二个）
  for (size_t i = 0; i < vars.size() - 1; ++i) {
    const FieldRef &cur = vars[i];
    const FieldRef &next = vars[i + 1];
    bool constraintFound = false;

    // 在约束序列中查找匹配的约束
    for (size_t j = 0; j < constraints.size(); ++j) {
      if (used[j])
        continue;

      const Constraint1 &c = constraints[j];

      // 检查约束左端是否为当前变量
      if (c.lhs_var1() == cur) {
        // 检查右端项（普通项或幂项）是否包含下一个变量
        if (termsContains(c.rhs_terms1(), next)) {
          // 满足所有条件，添加约束到结果
          result.push_back(c);
          used[j] = true;
          constraintFound = true;
          break; // 跳出当前约束查找循环
        }
      }
    }

    // 如果未找到匹配约束，抛出异常
    if (!constraintFound) {
      LLVM_DEBUG(llvm::dbgs()
                 << "No matching constraint found for variable pair: "
                 << "[" << cur.getValue() << " (fieldID: " << cur.getFieldID()
                 << ")]"
                 << " -> "
                 << "[" << next.getValue() << " (fieldID: " << next.getFieldID()
                 << ")]");
    }
  }

  return result;
}

Constraint1 substitute_constraint(const Constraint1 &c, const FieldRef &v,
                                  const std::list<Term> &terms, int cst) {
  // 查找目标变量v
  const auto &rhs_terms = c.rhs_terms1().get_terms();
  auto found_it = std::find_if(rhs_terms.begin(), rhs_terms.end(),
                               [&](const Term &t) { return t.var() == v; });

  // 如果未找到，直接返回原始约束
  if (found_it == rhs_terms.end()) {
    return c;
  }

  // 提取找到项的系数并从原始列表复制新列表
  nat coe = found_it->coe();
  std::list<Term> new_terms_list;

  // 复制除目标项外的所有项
  for (const auto &term : rhs_terms) {
    if (!(term.var() == v && term.coe() == coe)) {
      new_terms_list.push_back(term);
    }
  }

  // 处理每个替换项
  for (const auto &term : terms) {
    nat coe_t = term.coe();
    const FieldRef &var_t = term.var();
    nat new_coe = coe * coe_t * 1;

    auto exist_it =
        std::find_if(new_terms_list.begin(), new_terms_list.end(),
                     [&](const Term &t) { return t.var() == var_t; });

    if (exist_it != new_terms_list.end()) {
      // 合并系数并更新
      nat updated_coe = exist_it->coe() + new_coe;
      Term updated_term(updated_coe, var_t);
      *exist_it = updated_term;
    } else {
      // 添加新项
      new_terms_list.emplace_back(new_coe, var_t);
    }
  }

  int new_const = c.rhs_const1() + static_cast<int>(coe) * cst;

  // 创建新的约束对象
  return Constraint1(c.lhs_var1(), new_const, Terms(new_terms_list),
                     c.rhs_power());
}

Constraint1 substitute_c(const Constraint1 &c1, const Constraint1 &c2) {
  return substitute_constraint(c1, c2.lhs_var1(), c2.rhs_terms1().get_terms(),
                               c2.rhs_const1());
}

std::optional<Constraint1>
substitute_cs(const std::vector<Constraint1> &constraints) {
  if (constraints.empty()) {
    return std::nullopt;
  }

  if (constraints.size() == 1) {
    return constraints.front();
  }

  // 使用递归处理约束列表
  auto new_head = substitute_c(constraints[0], constraints[1]);
  std::vector<Constraint1> new_tail;
  new_tail.push_back(new_head);
  new_tail.insert(new_tail.end(), std::next(std::next(constraints.begin())),
                  constraints.end());

  return substitute_cs(new_tail);
}

std::optional<int> compute_ub(const Constraint1 &c) {
  const auto &terms = c.rhs_terms1().get_terms();
  const FieldRef &lhs_var = c.lhs_var1();

  // 查找左侧变量在右侧项中的位置
  auto found_it = std::find_if(terms.begin(), terms.end(), [&](const Term &t) {
    return t.var() == lhs_var;
  });

  if (found_it == terms.end()) {
    return std::nullopt;
  }

  nat coe = found_it->coe();
  // 检查系数是否 >= 2
  if (coe < 2) {
    return std::nullopt;
  }

  // 计算上界: -常数项 / (系数 - 1)
  int abs_const = -c.rhs_const1();
  nat divisor = coe - 1;

  // 处理除数为0的情况（理论上不会发生，因为coe>=2）
  if (divisor == 0) {
    return std::nullopt;
  }

  return abs_const / static_cast<int>(divisor);
}

std::optional<int> solve_ub_case1(const FieldRef &x, const FieldRef &var,
                                  const Constraint1 &c,
                                  const std::vector<Constraint1> &constraints,
                                  //const DependencyGraphResult &result,
                                  FieldRefGraph &graph_) {
  // c : lhs >= coe * var + ... + cst c
  // 找 x >= ? * lhs + ... 和 var >= ? * x + ...
  LLVM_DEBUG(llvm::dbgs() << "Solving [" << x.getValue()
                          << " (fieldID: " << x.getFieldID() << ")]\n");
  // auto find_vertex = [&](const FieldRef &p)
  //     -> std::optional<decltype(result.g)::vertex_descriptor> {
  //   if (auto it = result.vertex_map_.find(p); it != result.vertex_map_.end()) {
  //     return it->second;
  //   }
  //   return std::nullopt;
  // };

  // 顶点描述符
  // auto var_vertex = find_vertex(var);
  // auto x_vertex = find_vertex(x);
  // auto lhs_vertex = find_vertex(c.lhs_var1());

  // // 检查所有顶点是否存在
  // if (!var_vertex || !x_vertex || !lhs_vertex) {
  //   return std::nullopt;
  // }

  // std::vector<FieldRef> path1 =
  //     find_path(result.g, var_vertex.value(), x_vertex.value());
  // std::vector<FieldRef> path0 =
  //     find_path(result.g, x_vertex.value(), lhs_vertex.value());

  Node* var_node = graph_.addNode(var);
  Node* x_node = graph_.addNode(x);
  Node* lhs_node = graph_.addNode(c.lhs_var1());

  if (!var_node || !x_node || !lhs_node) {
    return std::nullopt;
  }

  std::vector<Node*> node_list1 = findPathBetween(var_node, x_node);
  std::vector<Node*> node_list0 = findPathBetween(x_node, lhs_node);
  std::vector<FieldRef> path1 = extractFieldRefs(node_list1);
  std::vector<FieldRef> path0 = extractFieldRefs(node_list0);

  if (path0.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No path found!\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Path found: "
                            << "[" << path0[0].getValue()
                            << " (fieldID: " << path0[0].getFieldID() << ")]");

    for (size_t i = 1; i < path0.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << " -> "
                              << "[" << path0[i].getValue() << " (fieldID: "
                              << path0[i].getFieldID() << ")]");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  if (path1.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No path found!\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Path found: "
                            << "[" << path1[0].getValue()
                            << " (fieldID: " << path1[0].getFieldID() << ")]");
    for (size_t i = 1; i < path1.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << " -> "
                              << "[" << path1[i].getValue() << " (fieldID: "
                              << path1[i].getFieldID() << ")]");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  // 按顺序找出约束
  auto conslist0 = orderConstraints(path0, constraints);
  auto conslist1 = orderConstraints(path1, constraints);

  for (size_t i = 0; i < conslist0.size(); ++i) {
    LLVM_DEBUG(llvm::dbgs() << conslist0[i] << "\n");
  }
  for (size_t i = 0; i < conslist1.size(); ++i) {
    LLVM_DEBUG(llvm::dbgs() << conslist1[i] << "\n");
  }

  // 依次带入不等式
  std::vector<Constraint1> conslist;
  conslist.reserve(conslist0.size() + 1 + conslist1.size());
  // 按顺序添加元素
  conslist.insert(conslist.end(), conslist0.begin(),
                  conslist0.end()); // 添加 conslist0
  conslist.push_back(c);            // 添加单个元素 c
  conslist.insert(conslist.end(), conslist1.begin(),
                  conslist1.end()); // 添加 conslist1

  auto ConResult = substitute_cs(conslist);
  if (ConResult.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "成功获取约束: " << ConResult.value() << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "错误：无法获取约束\n");
  }

  // 求出上界
  auto ub = compute_ub(ConResult.value());
  if (ub.has_value()) {
    return ub;
  } else {
    return 0;
  }
}

Valuation solve_ubs_case1(const std::vector<FieldRef> &tbsolved,
                          const FieldRef &var, const Constraint1 &c,
                          const std::vector<Constraint1> &constraints,
                          //const DependencyGraphResult &result,
                          FieldRefGraph &graph_) {
  Valuation v;
  for (const FieldRef &x : tbsolved) {
    // 求解当前变量
    auto ub = solve_ub_case1(x, var, c, constraints, graph_);
    if (ub.has_value()) {
      v[x] = ub.value();
      LLVM_DEBUG(llvm::dbgs() << "成功求解"
                              << "[" << x.getValue()
                              << " (fieldID: " << x.getFieldID() << ")]"
                              << "\n的上界: " << ub.value() << "\n\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[" << x.getValue()
                              << " (fieldID: " << x.getFieldID() << ")]"
                              << "求解失败\n");
    }
  };
  return v;
}

std::optional<int> solve_ub_case2(const FieldRef &x, const FieldRef &var1,
                                  const FieldRef &var2, const Constraint1 &c,
                                  const std::vector<Constraint1> &constraints,
                                  //const DependencyGraphResult &result,
                                  FieldRefGraph &graph_) {
  // c : lhs >= coe0 * var0 + coe1 * var1 + ... + cst c
  // 找 x >= ? * lhs + ... 和 var0 >= ? * x + ... 和 var1 >= ? * x + ...

  // auto it = llvm::GraphTraits<FieldRefGraph*>::nodes_begin(&graph_);
  // auto end = llvm::GraphTraits<FieldRefGraph*>::nodes_end(&graph_);
  // for (; it != end; ++it) {
  //   llvm::errs() << it->field << "的后继: \n";
  //   for (Node* child : it->successors)
  //     llvm::errs() << "  " << child->field << "; \n";
  // }
  // llvm::errs() << "\n";

  LLVM_DEBUG(llvm::dbgs() << "Solving " << x.getValue()
                          << " (fieldID: " << x.getFieldID() << ")]\n");

  // auto find_vertex = [&](const FieldRef &p)
  //     -> std::optional<decltype(result.g)::vertex_descriptor> {
  //   if (auto it = result.vertex_map_.find(p); it != result.vertex_map_.end()) {
  //     return it->second;
  //   }
  //   return std::nullopt;
  // };

  // // 顶点描述符
  // auto var1_vertex = find_vertex(var1);
  // auto var2_vertex = find_vertex(var2);
  // auto x_vertex = find_vertex(x);
  // auto lhs_vertex = find_vertex(c.lhs_var1());

  // // 检查所有顶点是否存在
  // if (!var1_vertex || !var2_vertex || !x_vertex || !lhs_vertex) {
  //   return std::nullopt;
  // }

  // std::vector<FieldRef> path2 =
  //     find_path(result.g, var2_vertex.value(), x_vertex.value());
  // std::vector<FieldRef> path1 =
  //     find_path(result.g, var1_vertex.value(), x_vertex.value());
  // std::vector<FieldRef> path0 =
  //     find_path(result.g, x_vertex.value(), lhs_vertex.value());

  Node* var1_node = graph_.addNode(var1);
  Node* var2_node = graph_.addNode(var2);
  Node* x_node = graph_.addNode(x);
  Node* lhs_node = graph_.addNode(c.lhs_var1());

  if (!var1_node || !var2_node || !x_node || !lhs_node) {
    return std::nullopt;
  }

  std::vector<Node*> node_list2 = findPathBetween(var2_node,x_node);
  std::vector<Node*> node_list1 = findPathBetween(var1_node, x_node);
  std::vector<Node*> node_list0 = findPathBetween(x_node, lhs_node);
  std::vector<FieldRef> path2 = extractFieldRefs(node_list2);
  std::vector<FieldRef> path1 = extractFieldRefs(node_list1);
  std::vector<FieldRef> path0 = extractFieldRefs(node_list0);

  if (path0.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No path found!\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Path found: "
                            << "[" << path0[0].getValue()
                            << " (fieldID: " << path0[0].getFieldID() << ")]");

    for (size_t i = 1; i < path0.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << " -> "
                              << "[" << path0[i].getValue() << " (fieldID: "
                              << path0[i].getFieldID() << ")]");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  if (path1.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No path found!\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Path found: "
                            << "[" << path1[0].getValue()
                            << " (fieldID: " << path1[0].getFieldID() << ")]");
    for (size_t i = 1; i < path1.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << " -> "
                              << "[" << path1[i].getValue() << " (fieldID: "
                              << path1[i].getFieldID() << ")]");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  if (path2.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No path found!\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Path found: "
                            << "[" << path2[0].getValue()
                            << " (fieldID: " << path2[0].getFieldID() << ")]");
    for (size_t i = 1; i < path2.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs() << " -> "
                              << "[" << path2[i].getValue() << " (fieldID: "
                              << path2[i].getFieldID() << ")]");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  // 按顺序找出约束
  auto conslist0 = orderConstraints(path0, constraints);
  auto conslist1 = orderConstraints(path1, constraints);
  auto conslist2 = orderConstraints(path2, constraints);

  for (size_t i = 0; i < conslist0.size(); ++i) {
    LLVM_DEBUG(llvm::dbgs() << conslist0[i] << "\n");
  }
  for (size_t i = 0; i < conslist1.size(); ++i) {
    LLVM_DEBUG(llvm::dbgs() << conslist1[i] << "\n");
  }
  for (size_t i = 0; i < conslist2.size(); ++i) {
    LLVM_DEBUG(llvm::dbgs() << conslist2[i] << "\n");
  }

  // 依次带入不等式
  std::vector<Constraint1> conslist;
  conslist.reserve(conslist0.size() + 1 + conslist1.size() + conslist2.size());
  // 按顺序添加元素
  conslist.insert(conslist.end(), conslist0.begin(),
                  conslist0.end()); // 添加 conslist0
  conslist.push_back(c);            // 添加单个元素 c
  conslist.insert(conslist.end(), conslist1.begin(),
                  conslist1.end()); // 添加 conslist1
  conslist.insert(conslist.end(), conslist2.begin(),
                  conslist2.end()); // 添加 conslist2

  for (size_t i = 0; i < conslist.size(); ++i) {
    LLVM_DEBUG(llvm::dbgs() << "按顺序的约束：" << conslist[i] << "\n");
  }

  auto ConResult = substitute_cs(conslist);
  if (ConResult.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "成功获取约束: " << ConResult.value() << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "错误：无法获取约束\n");
  }

  // 求出上界
  auto ub = compute_ub(ConResult.value());
  if (ub.has_value()) {
    return ub;
  } else {
    return 0;
  }
}

Valuation solve_ubs_case2(const std::vector<FieldRef> &tbsolved,
                          const FieldRef &var1, const FieldRef &var2,
                          const Constraint1 &c,
                          const std::vector<Constraint1> &constraints,
                          //const DependencyGraphResult &result,
                          FieldRefGraph &graph_) {

  Valuation v;
  for (const FieldRef &x : tbsolved) {
    // 求解当前变量
    auto ub = solve_ub_case2(x, var1, var2, c, constraints, graph_);
    if (ub.has_value()) {
      v[x] = ub.value();
      LLVM_DEBUG(llvm::dbgs() << "成功求解"
                              << "[" << x.getValue()
                              << " (fieldID: " << x.getFieldID() << ")]"
                              << "\n的上界: " << ub.value() << "\n\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[" << x.getValue()
                              << " (fieldID: " << x.getFieldID() << ")]"
                              << "求解失败\n");
    }
  };
  return v;
}

// 查找系数大于1项
std::optional<Constraint1>
findCWithCoeGreaterThanOne(const std::vector<Constraint1> &constraints) {
  for (const auto &c : constraints) {
    auto var = c.rhs_terms1().findVarWithCoeGreaterThanOne();
    if (var.has_value()) {
      return c;
    }
  }
  return std::nullopt; // 未找到符合条件的项
}

// 查找多于2项
std::optional<Constraint1>
findCWithTwoVars(const std::vector<Constraint1> &constraints) {
  for (const auto &c : constraints) {
    auto var = c.rhs_terms1().findFirstTwoVars();
    if (var.has_value()) {
      return c;
    }
  }
  return std::nullopt; // 未找到符合条件的项
}

Constraint1 relax_power(const Constraint1 &c) {
  if (c.rhs_power().has_value()) {
    Term relaxed_term(2, c.rhs_power().value());
    Terms combined_terms = c.rhs_terms1();
    combined_terms = combined_terms.combine_term(relaxed_term);
    Constraint1 c_(c.lhs_var1(), c.rhs_const1(), combined_terms, std::nullopt);
    return c_;
  } else
    return c;
}

std::vector<Constraint1>
relax_all_powers(const std::vector<Constraint1> &constraints) {
  std::vector<Constraint1> result;
  result.reserve(constraints.size());
  std::transform(constraints.begin(), constraints.end(),
                 std::back_inserter(result),
                 [](const Constraint1 &c) { return relax_power(c); });
  return result;
}

// 移除 rhs_terms1 为空的约束
std::vector<Constraint1>
remove_only_const(const std::vector<Constraint1> &constraints) {
  std::vector<Constraint1> result;
  result.reserve(constraints.size());
  std::copy_if(constraints.begin(), constraints.end(),
               std::back_inserter(result), [](const Constraint1 &c) {
                 return !c.rhs_terms1().empty(); // 仅保留 rhs_terms1 非空的约束
               });

  return result;
}

std::optional<Valuation> solve_ubs(const std::vector<Constraint1> &constraints,
                                   const std::vector<FieldRef> &tbsolved) {

  auto relaxed_constraints = relax_all_powers(constraints);
  auto processed_constraints = remove_only_const(relaxed_constraints);
  //DependencyGraphResult result = buildDependencyGraph(processed_constraints);
  FieldRefGraph graph_;
  for (const auto &c : constraints) {
    FieldRef lhs = c.lhs_var1();
    Node* lhs_node = graph_.addNode(lhs);

    // 遍历右侧项
    for (const auto &term : c.rhs_terms1()) {
      FieldRef rhs_var = term.var();
      Node* rhs_node = graph_.addNode(rhs_var);
      lhs_node->addSuccessor(rhs_node);
    }
  }

  auto c = findCWithCoeGreaterThanOne(processed_constraints);
  if (c.has_value()) {
    LLVM_DEBUG(llvm::dbgs()
               << "成功获取系数大于1的约束: " << c.value() << "\n");
    auto var = c.value().rhs_terms1().findVarWithCoeGreaterThanOne();
    if (var.has_value()) {
      Valuation ubs = solve_ubs_case1(tbsolved, var.value(), c.value(),
                                      processed_constraints, graph_);
      for (const auto &[pv, value] : ubs) {
        LLVM_DEBUG(llvm::dbgs() << "[" << pv.getValue() << " (fieldID: "
                                << pv.getFieldID() << ")] : " << value << "\n");
      }
      return ubs;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "没有系数大于1的项\n");
      return std::nullopt;
    };
  } else {
    auto c = findCWithTwoVars(processed_constraints);
    if (c.has_value()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "成功获取多于1项的约束: " << c.value() << "\n");
      auto var = c.value().rhs_terms1().findFirstTwoVars();
      if (var.has_value()) {
        Valuation ubs =
            solve_ubs_case2(tbsolved, var.value().first, var.value().second,
                            c.value(), processed_constraints, graph_);
        for (const auto &[pv, value] : ubs) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[" << pv.getValue() << " (fieldID: " << pv.getFieldID()
                     << ")] : " << value << "\n");
        }
        return ubs;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "没有两项\n");
        return std::nullopt;
      };
    } else {
      LLVM_DEBUG(llvm::dbgs() << "都～没有\n");
      return std::nullopt;
    }
  };
}

//===----------------------------------------------------------------------===//
// bab
//===----------------------------------------------------------------------===//

using Bounds =
    std::unordered_map<FieldRef, std::pair<int, int>, FieldRef::Hash>;

Valuation key_value(const Bounds &bounds) {
  Valuation node_val;
  for (const auto &[var, bound] : bounds) {
    nat lb = bound.first;
    nat ub = bound.second;
    node_val[var] = (lb + ub) / 2;
  }
  return node_val;
}

long long product_bounds(const Bounds &bounds) {
  long long sum = 0;
  for (const auto &[var, bound] : bounds) {
    nat lb = bound.first;
    nat ub = bound.second;
    if (ub >= lb) {
      sum += (ub - lb);
    }
  }
  return sum;
}

Bounds update_ub(const Bounds &bounds, const FieldRef &var, nat new_ub) {
  Bounds new_bounds = bounds;
  auto it = new_bounds.find(var);
  if (it != new_bounds.end()) {
    nat lb = it->second.first;
    it->second = std::make_pair(lb, new_ub);
  }
  return new_bounds;
}

Bounds update_lb(const Bounds &bounds, const FieldRef &var, nat new_lb) {
  Bounds new_bounds = bounds;
  auto it = new_bounds.find(var);
  if (it != new_bounds.end()) {
    nat ub = it->second.second;
    it->second = std::make_pair(new_lb, ub);
  }
  return new_bounds;
}

std::optional<Valuation> prioritize_fst(std::optional<Valuation> v1,
                                        std::optional<Valuation> v2) {

  if (v1) {
    return v1;
  }
  return v2;
}

// 获取变量的区间长度
nat length(const Bounds &bounds, const FieldRef &var) {
  auto it = bounds.find(var);
  if (it != bounds.end()) {
    nat lb = it->second.first;
    nat ub = it->second.second;
    if (ub >= lb) {
      return ub - lb;
    }
  }
  return 0;
}

// 获取约束1的右侧变量列表
std::vector<FieldRef> rhs_vars(const Constraint1 &c) {
  std::vector<FieldRef> vars;

  for (const auto &term : c.rhs_terms1().get_terms()) {
    vars.push_back(term.var());
  }

  if (c.rhs_power().has_value())
    vars.push_back(c.rhs_power().value());

  return vars;
}

// 获取约束2的右侧变量列表
std::vector<FieldRef> rhs_vars(const Constraint2 &c) {
  std::vector<FieldRef> vars;

  for (const auto &term : c.rhs_terms2().get_terms()) {
    vars.push_back(term.var());
  }

  return vars;
}

// 主分支定界函数
std::optional<Valuation> bab_bin(const std::vector<FieldRef> &scc,
                                 const Bounds &bounds,
                                 const std::vector<Constraint1> &cs1,
                                 const std::vector<Constraint2> &cs2) {

  Valuation current_node = key_value(bounds);

  // 检查约束1是否满足
  std::optional<Constraint1> unsat1;
  for (const auto &c : cs1) {
    if (!c.satisfies(current_node)) {
      unsat1 = c;
      break;
    }
  }

  // 检查约束2是否满足
  std::optional<Constraint2> unsat2;
  if (!unsat1) {
    for (const auto &c : cs2) {
      if (!c.satisfies(current_node)) {
        unsat2 = c;
        break;
      }
    }
  }

  // 所有约束都满足
  if (!unsat1 && !unsat2) {
    // 所有变量都是单点
    if (product_bounds(bounds) == 0) {
      return current_node;
    }
    // 缩小边界继续搜索
    Bounds new_bounds;
    for (const auto &[var, bound] : bounds) {
      nat lb = bound.first;
      nat ub = bound.second;
      nat mid = (lb + ub) / 2;
      new_bounds[var] = std::make_pair(lb, mid);
    }
    return bab_bin(scc, new_bounds, cs1, cs2);
  }

  // Constraint1不满足
  if (unsat1) {

    Constraint1 c1 = *unsat1;

    if (product_bounds(bounds) == 0) {
      return std::nullopt;
    }

    // 使用列表存储变量
    std::vector<FieldRef> rhs_vars_list = rhs_vars(c1);
    std::optional<FieldRef> v;

    // 遍历列表寻找可分割变量
    for (const auto &var : rhs_vars_list) {
      if (length(bounds, var) != 0) {
        v = var;
        break;
      }
    }

    if (v) {
      auto it = bounds.find(*v);
      if (it != bounds.end()) {
        nat lb = it->second.first;
        nat ub = it->second.second;
        nat mid = (lb + ub) / 2;

        // 优先搜索下半区
        auto left_sol = bab_bin(scc, update_ub(bounds, *v, mid), cs1, cs2);

        // 优先返回左侧解
        if (left_sol) {
          return left_sol;
        }

        // 搜索上半区
        return bab_bin(scc, update_lb(bounds, *v, mid + 1), cs1, cs2);
      }
      return std::nullopt; // 不可能的情况
    }
    // 无右侧可分割变量，检查左侧变量
    else if (length(bounds, c1.lhs_var1()) == 0) {
      return std::nullopt;
    } else {
      auto it = bounds.find(c1.lhs_var1());
      if (it != bounds.end()) {
        nat lb = it->second.first;
        nat ub = it->second.second;
        nat mid = (lb + ub) / 2;

        // 优先搜索上半区
        auto right_sol =
            bab_bin(scc, update_lb(bounds, c1.lhs_var1(), mid + 1), cs1, cs2);

        // 优先返回右侧解
        if (right_sol) {
          return right_sol;
        }

        // 搜索下半区
        return bab_bin(scc, update_ub(bounds, c1.lhs_var1(), mid), cs1, cs2);
      }
      return std::nullopt; // 不可能的情况
    }
  }

  // Constraint2不满足
  if (unsat2) {

    Constraint2 c2 = *unsat2;

    if (product_bounds(bounds) == 0) {
      return std::nullopt;
    }

    // 使用列表存储变量
    std::vector<FieldRef> rhs_vars_list = rhs_vars(c2);
    std::optional<FieldRef> v;

    // 遍历列表寻找可分割变量
    for (const auto &var : rhs_vars_list) {
      if (length(bounds, var) != 0) {
        v = var;
        break;
      }
    }

    if (v) {
      auto it = bounds.find(*v);
      if (it != bounds.end()) {
        nat lb = it->second.first;
        nat ub = it->second.second;
        nat mid = (lb + ub) / 2;

        // 优先搜索下半区
        auto left_sol = bab_bin(scc, update_ub(bounds, *v, mid), cs1, cs2);

        // 优先返回左侧解
        if (left_sol) {
          return left_sol;
        }

        // 搜索上半区
        return bab_bin(scc, update_lb(bounds, *v, mid + 1), cs1, cs2);
      }
      return std::nullopt; // 不可能的情况
    }
    return std::nullopt; // 无解
  }
  return std::nullopt; // 不可能的情况
}

Bounds combineValuations(const std::vector<FieldRef> &vars,
                         const Valuation &val1, const Valuation &val2) {
  Bounds result;

  for (const auto &var : vars) {
    // 确保变量在两个 Valuation 中都存在
    if (val1.find(var) == val1.end() || val2.find(var) == val2.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Variable not found in both valuations\n");
    }

    // 获取两个 Valuation 中的值
    int value1 = val1.at(var);
    int value2 = val2.at(var);

    // 创建 pair 并插入到结果中
    result[var] = std::make_pair(value1, value2);
  }

  return result;
}

Valuation createlb(const std::vector<Constraint1> &constraints) {
  Valuation valuation;

  // 遍历所有约束
  for (const auto &c : constraints) {
    FieldRef lhs_var = c.lhs_var1();                  // 获取左侧变量
    int rhs_const = static_cast<int>(c.rhs_const1()); // 转换为int

    // 检查变量是否已在
    auto it = valuation.find(lhs_var);

    if (it == valuation.end()) {
      // 变量不存在：初始化为0和rhs_const的较大值
      valuation[lhs_var] = std::max<int>(0, rhs_const);
    } else {
      // 变量已存在：更新为当前值和rhs_const的较大值
      it->second = std::max<int>(it->second, rhs_const);
    }
  }

  return valuation;
}

Valuation bab(const std::vector<Constraint1> &constraints,
              const std::vector<FieldRef> &tbsolved) {
  Valuation ubs = solve_ubs(constraints, tbsolved).value();
  Valuation lbs = createlb(constraints);
  Bounds bounds = combineValuations(tbsolved, lbs, ubs);

  std::optional<Valuation> solution = bab_bin(tbsolved, bounds,
                                              constraints, // 类型1约束列表
                                              {} // 类型2约束列表
  );

  // 输出结果
  if (solution.has_value()) {
    // LLVM_DEBUG(llvm::dbgs() << "可行解找到:\n";
    // for (const auto& [var, val] : *solution) {
    //     LLVM_DEBUG(llvm::dbgs() << "[" << var.getValue() << " (fieldID: " <<
    //     var.getFieldID() << ")]"
    //     << "\n = " << val << "\n";
    // }

    // // 验证约束
    // LLVM_DEBUG(llvm::dbgs() << "\n约束验证:\n";
    // for (const auto& c : constraints) {
    //     LLVM_DEBUG(llvm::dbgs() << c << (c.satisfies(*solution) ? " 满足" : "
    //     违反") << "\n";
    // }

    return solution.value();
  } else {
    // LLVM_DEBUG(llvm::dbgs() << "无可行解\n";
    return {};
  }
}

template <typename T>
std::vector<std::vector<T>>
cartesian_product(const std::vector<std::vector<T>> &input) {
  if (input.empty()) {
    return {{}};
  }

  // 递归步骤
  auto sub_product = cartesian_product(
      std::vector<std::vector<T>>(input.begin() + 1, input.end()));

  std::vector<std::vector<T>> result;
  for (const auto &elem : input[0]) {
    for (auto &sub_list : sub_product) {
      std::vector<T> new_list;
      new_list.reserve(sub_list.size() + 1);
      new_list.push_back(elem);
      new_list.insert(new_list.end(), sub_list.begin(), sub_list.end());
      result.push_back(std::move(new_list));
    }
  }

  return result;
}

void smaller_valuation(Valuation &v1, const Valuation &v2) {
  for (const auto &[key, value2] : v2) {
    auto it_v1 = v1.find(key);
    if (it_v1 != v1.end() && it_v1->second > value2) {
      it_v1->second = value2;
    }
  }
}

LogicalResult InferenceMapping::solve() {
  // flat min, generate conjunction
  std::vector<std::vector<Constraint1>> min_list;
  min_list.reserve(constraints_min_.size());
  std::transform(constraints_min_.begin(), constraints_min_.end(),
                 std::back_inserter(min_list), [this](const Constraint_Min &c) {
                   return this->list_Constraint_Min(c);
                 });
  auto flatten_min = cartesian_product(min_list);

  // 分别求解conjunction
  for (auto cs : flatten_min) {
    ConstraintSolver solver(constraints_, /*dependency_graph_, vertex_map_,*/ graph_);
    // 依次把cs添加，并画到图中
    for (auto c : cs) {
      //llvm::errs() << c << "\n";
      solver.addConstraint(c);
    }

  /*auto it = llvm::GraphTraits<FieldRefGraph*>::nodes_begin(&solver.fieldRefGraph());
  auto end = llvm::GraphTraits<FieldRefGraph*>::nodes_end(&solver.fieldRefGraph());
  for (; it != end; ++it) {
    llvm::errs() << it->field << "的后继: \n";
    for (Node* child : it->successors)
      llvm::errs() << "  " << child->field << "; \n";
  }

  // 使用 SCC 迭代器分解强连通分量
  llvm::errs() << "FieldRef Graph SCC Decomposition:\n";
  for (auto sccIter = llvm::scc_begin(&solver.fieldRefGraph()); sccIter != llvm::scc_end(&solver.fieldRefGraph()); ++sccIter) {
    const auto& component = *sccIter;
    llvm::errs() << "SCC (size " << component.size() << "): ";
    for (Node* node : component) 
      llvm::errs() << node->field << "; ";
    llvm::errs() << "\n";
  }*/

    auto solve_result = solver.solve();
    if (failed(solve_result)) {
      continue;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "发现一组解:\n");
      Valuation current_solution = solver.solution();
      for (const auto &[pv, value] : current_solution) {
        LLVM_DEBUG(llvm::dbgs() //<< "[" << pv.getValue() << " (fieldID: " <<
                                //pv.getFieldID() << ")] : \n"
                   << pv.getValue().getLoc() << " : " << value << "\n");
      }

      if (final_solution_.empty()) {
        final_solution_ = current_solution;
      } else {
        smaller_valuation(final_solution_, current_solution);
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Inference Result Application
//===----------------------------------------------------------------------===//

class InferenceTypeUpdate {
public:
  InferenceTypeUpdate(InferenceMapping &mapping) : mapping(mapping) {}

  LogicalResult update(CircuitOp op);
  FailureOr<bool> updateOperation(Operation *op);
  FailureOr<bool> updateValue(Value value);
  FIRRTLBaseType updateType(FieldRef fieldRef, FIRRTLBaseType type);

private:
  const InferenceMapping &mapping;
};

/// Update the types throughout a circuit.
LogicalResult InferenceTypeUpdate::update(CircuitOp op) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Update types") << "\n\n";
  });
  return mlir::failableParallelForEach(
      op.getContext(), op.getOps<FModuleOp>(), [&](FModuleOp op) {
        // Skip this module if it had no widths to be
        // inferred at all.
        if (mapping.isModuleSkipped(op))
          return success();
        auto isFailed = op.walk<WalkOrder::PreOrder>([&](Operation *op) {
                            if (failed(updateOperation(op)))
                              return WalkResult::interrupt();
                            return WalkResult::advance();
                          }).wasInterrupted();
        return failure(isFailed);
      });
}

/// Update the result types of an operation.
FailureOr<bool> InferenceTypeUpdate::updateOperation(Operation *op) {
  bool anyChanged = false;

  for (Value v : op->getResults()) {
    auto result = updateValue(v);
    if (failed(result))
      return result;
    anyChanged |= *result;
  }

  // If this is a connect operation, width inference might have inferred a RHS
  // that is wider than the LHS, in which case an additional BitsPrimOp is
  // necessary to truncate the value.
  if (auto con = dyn_cast<ConnectOp>(op)) {
    auto lhs = con.getDest();
    auto rhs = con.getSrc();
    auto lhsType = type_dyn_cast<FIRRTLBaseType>(lhs.getType());
    auto rhsType = type_dyn_cast<FIRRTLBaseType>(rhs.getType());

    // Nothing to do if not base types.
    if (!lhsType || !rhsType)
      return anyChanged;

    auto lhsWidth = lhsType.getBitWidthOrSentinel();
    auto rhsWidth = rhsType.getBitWidthOrSentinel();
    if (lhsWidth >= 0 && rhsWidth >= 0 && lhsWidth < rhsWidth) {
      OpBuilder builder(op);
      auto trunc = builder.createOrFold<TailPrimOp>(con.getLoc(), con.getSrc(),
                                                    rhsWidth - lhsWidth);
      if (type_isa<SIntType>(rhsType))
        trunc =
            builder.createOrFold<AsSIntPrimOp>(con.getLoc(), lhsType, trunc);

      LLVM_DEBUG(llvm::dbgs()
                 << "Truncating RHS to " << lhsType << " in " << con << "\n");
      con->replaceUsesOfWith(con.getSrc(), trunc);
    }
    return anyChanged;
  }

  // If this is a module, update its ports.
  if (auto module = dyn_cast<FModuleOp>(op)) {
    // Update the block argument types.
    bool argsChanged = false;
    SmallVector<Attribute> argTypes;
    argTypes.reserve(module.getNumPorts());
    for (auto arg : module.getArguments()) {
      auto result = updateValue(arg);
      if (failed(result))
        return result;
      argsChanged |= *result;
      argTypes.push_back(TypeAttr::get(arg.getType()));
    }

    // Update the module function type if needed.
    if (argsChanged) {
      module.setPortTypesAttr(ArrayAttr::get(module.getContext(), argTypes));
      anyChanged = true;
    }
  }
  return anyChanged;
}

/// Resize a `uint`, `sint`, or `analog` type to a specific width.
static FIRRTLBaseType resizeType(FIRRTLBaseType type, uint32_t newWidth) {
  auto *context = type.getContext();
  return FIRRTLTypeSwitch<FIRRTLBaseType, FIRRTLBaseType>(type)
      .Case<UIntType>([&](auto type) {
        return UIntType::get(context, newWidth, type.isConst());
      })
      .Case<SIntType>([&](auto type) {
        return SIntType::get(context, newWidth, type.isConst());
      })
      .Case<AnalogType>([&](auto type) {
        return AnalogType::get(context, newWidth, type.isConst());
      })
      .Default([&](auto type) { return type; });
}

/// Update the type of a value.
FailureOr<bool> InferenceTypeUpdate::updateValue(Value value) {
  // Check if the value has a type which we can update.
  auto type = type_dyn_cast<FIRRTLType>(value.getType());
  if (!type)
    return false;

  // Fast path for types that have fully inferred widths.
  if (!hasUninferredWidth(type))
    return false;

  // If this is an operation that does not generate any free variables that
  // are determined during width inference, simply update the value type based
  // on the operation arguments.
  if (auto op = dyn_cast_or_null<InferTypeOpInterface>(value.getDefiningOp())) {
    SmallVector<Type, 2> types;
    auto res =
        op.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(),
                            op->getAttrDictionary(), op->getPropertiesStorage(),
                            op->getRegions(), types);
    if (failed(res))
      return failure();

    assert(types.size() == op->getNumResults());
    for (auto [result, type] : llvm::zip(op->getResults(), types)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Inferring " << result << " as " << type << "\n");
      result.setType(type);
    }
    return true;
  }

  // Recreate the type, substituting the solved widths.
  auto *context = type.getContext();
  unsigned fieldID = 0;
  std::function<FIRRTLBaseType(FIRRTLBaseType)> updateBase =
      [&](FIRRTLBaseType type) -> FIRRTLBaseType {
    auto width = type.getBitWidthOrSentinel();
    if (width >= 0) {
      // Known width integers return themselves.
      fieldID++;
      return type;
    }
    if (width == -1) {
      // Unknown width integers return the solved type.
      auto newType = updateType(FieldRef(value, fieldID), type);
      fieldID++;
      return newType;
    }
    if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      // Bundle types recursively update all bundle elements.
      fieldID++;
      llvm::SmallVector<BundleType::BundleElement, 3> elements;
      for (auto &element : bundleType) {
        auto updatedBase = updateBase(element.type);
        if (!updatedBase)
          return {};
        elements.emplace_back(element.name, element.isFlip, updatedBase);
      }
      return BundleType::get(context, elements, bundleType.isConst());
    }
    if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      // TODO: this should recurse into the element type of 0 length vectors and
      // set any unknown width to 0.
      if (vecType.getNumElements() > 0) {
        auto updatedBase = updateBase(vecType.getElementType());
        if (!updatedBase)
          return {};
        auto newType = FVectorType::get(updatedBase, vecType.getNumElements(),
                                        vecType.isConst());
        fieldID = save + vecType.getMaxFieldID();
        return newType;
      }
      // If this is a 0 length vector return the original type.
      return type;
    }
    llvm_unreachable("Unknown type inside a bundle!");
  };

  // Update the type.
  auto newType = mapBaseTypeNullable(type, updateBase);
  if (!newType)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "Update " << value << " to " << newType << "\n");
  value.setType(newType);

  // If this is a ConstantOp, adjust the width of the underlying APInt.
  // Unsized constants have APInts which are *at least* wide enough to hold
  // the value, but may be larger. This can trip up the verifier.
  if (auto op = value.getDefiningOp<ConstantOp>()) {
    auto k = op.getValue();
    auto bitwidth = op.getType().getBitWidthOrSentinel();
    if (k.getBitWidth() > unsigned(bitwidth))
      k = k.trunc(bitwidth);
    op->setAttr("value", IntegerAttr::get(op.getContext(), k));
  }

  return newType != type;
}

/// Update a type.
FIRRTLBaseType InferenceTypeUpdate::updateType(FieldRef fieldRef,
                                               FIRRTLBaseType type) {
  assert(type.isGround() && "Can only pass in ground types.");
  auto value = fieldRef.getValue();
  // Get the inferred width.
  Valuation valuation = mapping.final_solution();
  int32_t solution; //*expr->getSolution();
  auto it = valuation.find(fieldRef);
  if (it == valuation.end()) {
    mlir::emitError(value.getLoc(), "width should have been inferred");
    return {};
  }
  solution = it->second;
  assert(solution >= 0); // The solver infers variables to be 0 or greater.
  return resizeType(type, solution);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InferWidthsPass_new
    : public circt::firrtl::impl::InferWidths_newBase<InferWidthsPass_new> {
  void runOnOperation() override;
};
} // namespace

void InferWidthsPass_new::runOnOperation() {
  InferenceMapping mapping(getAnalysis<SymbolTable>(),
                           getAnalysis<hw::InnerSymbolTableCollection>());
  if (failed(mapping.map(getOperation())))
    return signalPassFailure();

  // fast path if no inferrable widths are around
  if (mapping.areAllModulesSkipped())
    return markAllAnalysesPreserved();

  // Solve the constraints.
  if (failed(mapping.solve()))
    return signalPassFailure();

  // Update the types with the inferred widths.
  if (failed(InferenceTypeUpdate(mapping).update(getOperation())))
    return signalPassFailure();

  LLVM_DEBUG(llvm::dbgs() << "最终解:\n");
  for (const auto &[pv, value] : mapping.final_solution()) {
    LLVM_DEBUG(llvm::dbgs() //<< "[" << pv.getValue() << " (fieldID: " <<
                            //pv.getFieldID() << ")] : \n"
               << pv.getValue().getLoc() << " : " << value << "\n");
  }
}
