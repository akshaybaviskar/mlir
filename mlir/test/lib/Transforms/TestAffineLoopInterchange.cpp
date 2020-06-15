#include "iostream"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/Support/Debug.h"
#include <unordered_map>

using namespace mlir;
using namespace std;

#define CACHE_LINE_SIZE 8

namespace {

static LogicalResult getInstIndexSet(Operation *op,
                                     FlatAffineConstraints *indexSet) {
  SmallVector<AffineForOp, 4> loops;
  getLoopIVs(*op, &loops);
  return getIndexSet(loops, indexSet);
}

class ValuePositionMap {
public:
  void addSrcValue(Value value) {
    if (addValueAt(value, &srcDimPosMap, numSrcDims))
      ++numSrcDims;
  }
  void addSymbolValue(Value value) {
    if (addValueAt(value, &symbolPosMap, numSymbols))
      ++numSymbols;
  }
  unsigned getSrcDimOrSymPos(Value value) const {
    return getDimOrSymPos(value, srcDimPosMap, 0);
  }
  unsigned getSymPos(Value value) const {
    auto it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + it->second;
  }

  unsigned getNumSrcDims() const { return numSrcDims; }
  unsigned getNumDims() const { return numSrcDims; }
  unsigned getNumSymbols() const { return numSymbols; }

private:
  bool addValueAt(Value value, DenseMap<Value, unsigned> *posMap,
                  unsigned position) {
    auto it = posMap->find(value);
    if (it == posMap->end()) {
      (*posMap)[value] = position;
      return true;
    }
    return false;
  }
  unsigned getDimOrSymPos(Value value,
                          const DenseMap<Value, unsigned> &dimPosMap,
                          unsigned dimPosOffset) const {
    auto it = dimPosMap.find(value);
    if (it != dimPosMap.end()) {
      return dimPosOffset + it->second;
    }
    it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + it->second;
  }

  unsigned numSrcDims = 0;
  unsigned numSymbols = 0;
  DenseMap<Value, unsigned> srcDimPosMap;
  DenseMap<Value, unsigned> symbolPosMap;
};

static void
buildDimAndSymbolPositionMaps(const FlatAffineConstraints &srcDomain,
                              const AffineValueMap &srcAccessMap,
                              ValuePositionMap *valuePosMap,
                              FlatAffineConstraints *dependenceConstraints) {
  auto updateValuePosMap = [&](ArrayRef<Value> values, bool isSrc) {
    for (unsigned i = 0, e = values.size(); i < e; ++i) {
      auto value = values[i];
      if (!isForInductionVar(values[i])) {
        assert(isValidSymbol(values[i]) &&
               "access operand has to be either a loop IV or a symbol");
        valuePosMap->addSymbolValue(value);
      } else if (isSrc) {
        valuePosMap->addSrcValue(value);
      } else {
      }
    }
  };

  SmallVector<Value, 4> srcValues;
  srcDomain.getIdValues(0, srcDomain.getNumDimAndSymbolIds(), &srcValues);
  // Update value position map with identifiers from src iteration domain.
  updateValuePosMap(srcValues, true);
  // Update value position map with identifiers from src access function.
  updateValuePosMap(srcAccessMap.getOperands(), true);
}

// Sets up dependence constraints columns appropriately.
static void
initDependenceConstraints(const FlatAffineConstraints &srcDomain,
                          const AffineValueMap &srcAccessMap,
                          const ValuePositionMap &valuePosMap,
                          FlatAffineConstraints *dependenceConstraints) {
  // Calculate number of equalities/inequalities and columns required to
  // initialize FlatAffineConstraints for 'dependenceDomain'.
  unsigned numIneq = srcDomain.getNumInequalities();
  AffineMap srcMap = srcAccessMap.getAffineMap();
  unsigned numEq = srcMap.getNumResults();
  unsigned numDims = srcDomain.getNumDimIds();
  unsigned numSymbols = 0; // valuePosMap.getNumSymbols();
  unsigned numLocals = 0;  // srcDomain.getNumLocalIds();
  unsigned numIds = numDims + numSymbols + numLocals;
  unsigned numCols = numIds + 1;

  // Set flat affine constraints sizes and reserving space for constraints.
  dependenceConstraints->reset(numIneq, numEq, numCols, numDims, numSymbols,
                               numLocals);

  // Set values corresponding to dependence constraint identifiers.
  SmallVector<Value, 4> srcLoopIVs;
  srcDomain.getIdValues(0, srcDomain.getNumDimIds(), &srcLoopIVs);

  dependenceConstraints->setIdValues(0, srcLoopIVs.size(), srcLoopIVs);

  // Set values for the symbolic identifier dimensions.
  auto setSymbolIds = [&](ArrayRef<Value> values) {
    for (auto value : values) {
      if (!isForInductionVar(value)) {
        assert(isValidSymbol(value) && "expected symbol");
        dependenceConstraints->setIdValue(valuePosMap.getSymPos(value), value);
      }
    }
  };

  setSymbolIds(srcAccessMap.getOperands());

  SmallVector<Value, 8> srcSymbolValues;
  srcDomain.getIdValues(srcDomain.getNumDimIds(),
                        srcDomain.getNumDimAndSymbolIds(), &srcSymbolValues);
  setSymbolIds(srcSymbolValues);

  for (unsigned i = 0, e = dependenceConstraints->getNumDimAndSymbolIds();
       i < e; i++)
    assert(dependenceConstraints->getIds()[i].hasValue());
}

static LogicalResult addMemRefAccessConstraints(
    const AffineValueMap &srcAccessMap, const ValuePositionMap &valuePosMap,
    FlatAffineConstraints *dependenceDomain, vector<vector<int>> &AccessFun) {
  AffineMap srcMap = srcAccessMap.getAffineMap();
  unsigned numResults = srcMap.getNumResults();

  ArrayRef<Value> srcOperands = srcAccessMap.getOperands();

  std::vector<SmallVector<int64_t, 8>> srcFlatExprs;
  FlatAffineConstraints srcLocalVarCst;
  // Get flattened expressions for the source destination maps.
  if (failed(getFlattenedAffineExprs(srcMap, &srcFlatExprs, &srcLocalVarCst)))
    return failure();

  // Equality to add.
  SmallVector<int64_t, 8> eq(dependenceDomain->getNumCols());
  for (unsigned i = 0; i < numResults; ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);

    // Flattened AffineExpr for src result 'i'.
    const auto &srcFlatExpr = srcFlatExprs[i];
    // Set identifier coefficients from src access function.
    for (unsigned j = 0, e = srcOperands.size(); j < e; ++j)
      eq[valuePosMap.getSrcDimOrSymPos(srcOperands[j])] = srcFlatExpr[j];
    // Set constant term.
    eq[eq.size() - 1] = srcFlatExpr[srcFlatExpr.size() - 1];

    // Add equality constraint.
    dependenceDomain->addEquality(eq);

    vector<int> eq_copy(dependenceDomain->getNumCols());

    for (unsigned long i = 0; i < eq.size(); i++) {
      eq_copy[i] = eq[i];
    }

    AccessFun.push_back(eq_copy);
  }

  return success();
}

struct AffineLoopInterchange
    : public PassWrapper<AffineLoopInterchange, FunctionPass> {
  void runOnFunction() override;

  vector<vector<int>> accfunmatch;
  vector<pair<vector<int>, vector<vector<int>>>> group_list;
  unordered_map<int, bool> parallel_loops;
  vector<pair<unsigned long, vector<int>>> PermMisses;
  vector<vector<int>> spatreuse;
  vector<vector<int>> tempreuse;
  vector<vector<int>> Dependences;
  vector<pair<AffineForOp, vector<unsigned int>>> BestPerm;
  class LoadStoreInfo;
  vector<LoadStoreInfo> ls_vector;
  class LoadStoreInfo {
  public:
    Operation *op;
    MemRefAccess srcAccess;
    AffineValueMap srcAccessMap;
    FlatAffineConstraints srcDomain;
    ValuePositionMap valuePosMap;
    FlatAffineConstraints dependenceConstraints;
    vector<vector<int>> AccessFun;

    LoadStoreInfo(Operation *opn) : srcAccess(opn) {
      op = opn;

      // Get composed access function for 'srcAccess'.
      srcAccess.getAccessMap(&srcAccessMap);

      // Get iteration domain for the 'srcAccess' operation.
      if (failed(getInstIndexSet(srcAccess.opInst, &srcDomain))) {
        cout << " Unable to find iteration domain for: " << endl;
        op->dump();
        exit(1);
      }

      // Build dim and symbol position maps for each access from access operand
      // Value to position in merged constraint system.
      buildDimAndSymbolPositionMaps(srcDomain, srcAccessMap, &valuePosMap,
                                    &dependenceConstraints);
      initDependenceConstraints(srcDomain, srcAccessMap, valuePosMap,
                                &dependenceConstraints);

      // Create memref access constraint by equating src/dst access functions.
      // Note that this check is conservative, and will fail in the future when
      // local variables for mod/div exprs are supported.
      if (failed(addMemRefAccessConstraints(
              srcAccessMap, valuePosMap, &dependenceConstraints, AccessFun))) {
        cout << "Create MemRefAccess constraint failed for:" << endl;
        op->dump();
        exit(1);
      }
    }

    unsigned long cacheMisses(vector<int> perm, AffineForOp &forOp) {
      unsigned long misses = 1;
      int power = 0;
      int div = 0;

      SmallVector<AffineForOp, 4> loops;
      getPerfectlyNestedLoops(loops, forOp);

      SmallVector<unsigned, 4> tripcount(loops.size());
      SmallVector<unsigned, 4> stride(loops.size());

      for (unsigned long i = 0; i < loops.size(); i++) {
        int64_t ub = loops[perm[i]].getConstantUpperBound();
        int64_t lb = loops[perm[i]].getConstantLowerBound();

        int64_t step = loops[perm[i]].getStep();

        tripcount[i] = ((ub - 1) - lb + step) / step;
        stride[i] = step;
      }

      int pivotfound = 0;
      for (int i = perm.size() - 1; i >= 0; i--) {
        if (pivotfound == 1) {
          power++;
          continue;
        }

        int col = perm[i];
        unsigned long row;
        int allZero = 1;
        for (row = 0; row < AccessFun.size() - 1; row++) {
          if (AccessFun[row][col] == 0) {
            continue;
          } else {
            pivotfound = 1;
            allZero = 0;
            power++;
            break;
          }
        }

        if (allZero) {
          if (AccessFun[row][col] == 0) {
            continue;
          } else if ((abs(AccessFun[row][col]) * stride[col]) <
                     CACHE_LINE_SIZE) {
            div = CACHE_LINE_SIZE * abs(AccessFun[row][col]) * stride[col];
            pivotfound = 1;
            power++;
          } else {
            pivotfound = 1;
            power++;
          }
        }
      }

      for (int i = 0; i < power; i++) {
        misses = misses * tripcount[i];
      }
      if (div) {
        misses = misses / div;
      }

      return misses;
    }
  };

  int factorial(int n) { return (n == 1 || n == 0) ? 1 : n * factorial(n - 1); }

  // find dependences between all pairs of load/store operatio and convert them
  // to a direction vector.
  // Also find if two accesses have temporal group locality. Here, two
  // operations are considered to have temporal group locality if they access to
  // same element in two different iterations within the innermost loop of
  // forloop nest. e.g. A[i-1][j][k] and A[i][j][k] will have temporal group
  // locality when i is the innermost loop in for loopnest otherwise not. Since
  // two consecutive iterations will be accessing the same element. matrix
  // tempreuse[i][j] stores the temporal group reuse information. It is set to
  // -1 if no reuse exist. If group reuse exist then it stores the innermost
  // loop id for which temp reuse exist.
  void calculateDependences() {
    tempreuse =
        vector<vector<int>>(ls_vector.size(), vector<int>(ls_vector.size()));

    for (unsigned i = 0, e = ls_vector.size(); i < e; ++i) {
      for (unsigned j = 0; j < e; ++j) {
        if (i == j) {
          tempreuse[i][j] = 1;
          continue;
        }
        tempreuse[i][j] = -1;

        unsigned numCommonLoops =
            getNumCommonSurroundingLoops(*ls_vector[i].op, *ls_vector[j].op);
        for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
          FlatAffineConstraints dependenceConstraints;
          SmallVector<DependenceComponent, 2> dependenceComponents;
          DependenceResult result = checkMemrefAccessDependence(
              ls_vector[i].srcAccess, ls_vector[j].srcAccess, d,
              &dependenceConstraints, &dependenceComponents, true);
          assert(result.value != DependenceResult::Failure);
          bool ret = hasDependence(result);

          if (ret) {
            if (dependenceComponents.empty() || d > numCommonLoops)
              continue;

            vector<int> dep(dependenceComponents.size());

            for (unsigned k = 0, e = dependenceComponents.size(); k < e; ++k) {
              long lb = -1;
              if (dependenceComponents[k].lb.hasValue() &&
                  dependenceComponents[k].lb.getValue() !=
                      std::numeric_limits<int64_t>::min())
                lb = dependenceComponents[k].lb.getValue();

              long ub = 1;
              if (dependenceComponents[k].ub.hasValue() &&
                  dependenceComponents[k].ub.getValue() !=
                      std::numeric_limits<int64_t>::max())
                ub = dependenceComponents[k].ub.getValue();

              if ((lb == 0) && (ub == 0)) {
                dep[k] = 0;
              } else if ((lb >= 0) && (ub >= 0)) {
                dep[k] = 1;
              } else {
                dep[k] = -1;
              }
            }

            // add in dependencies list if not RAR
            if (isa<AffineStoreOp>(ls_vector[i].op) ||
                isa<AffineStoreOp>(ls_vector[j].op))
              Dependences.push_back(dep);

            int dir_exist = 0;
            int dir;
            for (unsigned long l = 0; l < dep.size(); l++) {
              if (dep[l] == 0) {
                continue;
              } else {
                if (dir_exist) {
                  dir_exist = 0;
                  break;
                } else {
                  dir_exist = 1;
                  dir = l;
                }
              }
            }

            if ((dir_exist) && (accfunmatch[i][j] == 1)) {
              tempreuse[i][j] = dir;
            }
          }
        }
      }
    }
  }

  // finds the loops in the for loopnest which can be executed parallely
  void find_par_loops() {
    for (unsigned long i = 0; i < ls_vector[0].AccessFun[0].size() - 1; i++) {
      parallel_loops[i] = true;
    }

    for (auto i : Dependences) {
      for (unsigned long j = 0; j < i.size(); j++) {
        if (i[j] != 0) {
          parallel_loops.erase(j);
          break;
        }
      }
    }
  }

  // the permutation with least cache misses is selected as best permutation
  // if more than two permutation have same cache misses then the permutation
  // which can have outer parallelism is selected as the best permutation
  vector<unsigned> find_best_perm() {
    sort(PermMisses.begin(), PermMisses.end());
    unsigned long last = PermMisses[0].first;
    int best = 0;
    for (unsigned int i = 0; i < PermMisses.size(); i++) {
      if (PermMisses[i].first != last)
        break;
      if (parallel_loops.count(PermMisses[i].second[0]) == 1) {
        best = i;
        break;
      }
    }

    int allSame = 1;
    for (unsigned int i = 0; i < PermMisses.size(); i++) {
      if (PermMisses[i].first != last) {
        allSame = 0;
        break;
      }
    }

    if ((allSame) && (group_list.size() != 0)) {
      vector<vector<int>> last_grp(group_list[0].second);
      for (unsigned int i = 0; i < group_list.size(); i++) {
        // total no of groups for each permutation
        if (group_list[i].second.size() != last_grp.size()) {
          allSame = 0;
          break;
        }
        // each group
        for (unsigned int j = 0; j < group_list[i].second.size(); j++) {
          if (group_list[i].second[j].size() != last_grp[j].size()) {
            allSame = 0;
            break;
          }
          for (unsigned int k = 0; k < group_list[i].second[j].size(); k++) {
            if (group_list[i].second[j][k] != last_grp[j][k]) {
              allSame = 0;
              break;
            }
          }
        }
      }
    }

    if (allSame) {
      double max = 0;
      int maxid = 0;
      unsigned long total_cost = PermMisses[0].first;

      for (unsigned int i = 0; i < group_misses.size(); i++) {
        double sum = 0;
        for (unsigned int j = 0; j < group_misses[i].second.size(); j++) {
          sum += group_list[i].second[j].size() *
                 (1 - (double)group_misses[i].second[j] / total_cost);
        }
        if (sum > max) {
          max = sum;
          maxid = i;
        }
      }
      best = maxid;
    }

    std::vector<unsigned int> permMap(PermMisses[best].second.size());
    for (unsigned inx = 0; inx < PermMisses[best].second.size(); ++inx) {
      permMap[PermMisses[best].second[inx]] = inx;
    }

    return permMap;
  }

  bool isValidPerm(vector<int> perm) {
    if (!Dependences.size())
      return true;
    for (auto i : Dependences) {
      for (unsigned long j = 0; j < i.size(); j++) {
        if (i[perm[j]] > 0)
          break;
        if (i[perm[j]] < 0)
          return false;
      }
    }
    return true;
  }

  // creates a matrix accfunmatch of size nxn where n is the number of
  // load/store operations in the for loopnest. accfunmatch[i][j] is 1 if
  // operation i and j refer to same argument and have same access function else
  // it is 0
  void find_same_acc_fun() {
    accfunmatch =
        vector<vector<int>>(ls_vector.size(), vector<int>(ls_vector.size()));
    for (unsigned long i = 0; i < ls_vector.size(); i++) {
      for (unsigned long j = i; j < ls_vector.size(); j++) {
        if (i == j) {
          accfunmatch[i][j] = 1;
          continue;
        }

        if (ls_vector[i].srcAccess.memref != ls_vector[j].srcAccess.memref) {
          accfunmatch[i][j] = 0;
          accfunmatch[j][i] = 0;
          continue;
        }

        int same_accfun = 1;
        for (unsigned long row = 0; row < ls_vector[i].AccessFun.size();
             row++) {
          for (unsigned long col = 0;
               col < ls_vector[i].AccessFun[0].size() - 1; col++) {
            if (ls_vector[i].AccessFun[row][col] !=
                ls_vector[j].AccessFun[row][col]) {
              same_accfun = 0;
              break;
            }
          }
          if (same_accfun == 0)
            break;
        }

        accfunmatch[i][j] = same_accfun;
        accfunmatch[j][i] = same_accfun;
      }
    }
  }

  // creates a matrix spatreuse of size nxn where n is the number of load/store
  // operations in the for loopnest. spatreuse[i][j] is 1 if accfunmatch[i][j] =
  // 1 AND all the dimensions of the two operations are same except last
  // dimension and last dimension differ by maximum CACHE_LINE_SIZE for two
  // operations. e.g. Assume CACHE_LINE_SIZE to be 8.
  //  A[i-1][j][k] and A[i-1][j][k+6] will have spatial reuse since both the
  //  elements are highly likely to lie in same cache line
  // while A[i][j][k] and A[i][j][k+8] won't lie in same cache line and thus for
  // such a pair spatreuse is set to 0.
  void find_spatial_groups() {
    spatreuse =
        vector<vector<int>>(ls_vector.size(), vector<int>(ls_vector.size()));

    for (unsigned long i = 0; i < ls_vector.size(); i++) {
      for (unsigned long j = i; j < ls_vector.size(); j++) {
        if (i == j) {
          spatreuse[i][j] = 1;
          continue;
        }

        int spat_group = 1;
        if (accfunmatch[i][j]) {
          unsigned long row;
          unsigned long col = ls_vector[i].AccessFun[0].size() - 1;
          for (row = 0; row < ls_vector[i].AccessFun.size() - 1; row++) {
            if ((ls_vector[i].AccessFun[row][col] -
                 ls_vector[j].AccessFun[row][col]) != 0) {
              spat_group = 0;
              break;
            }
          }

          if (spat_group == 1) {
            if (abs(ls_vector[i].AccessFun[row][col] -
                    ls_vector[j].AccessFun[row][col]) >= CACHE_LINE_SIZE) {
              spat_group = 0;
            }
          }

          spatreuse[i][j] = spat_group;
          spatreuse[j][i] = spat_group;
        }
      }
    }
  }

  // Group all the load/stores if they exist either spatial or temporal locality
  // for given permutation. and estimate total cache misses after grouping.
  vector<pair<vector<int>, vector<unsigned long>>> group_misses;
  void find_access_groups(vector<int> perm, AffineForOp &forOp) {
    vector<vector<int>> groups;
    vector<int> visited(ls_vector.size());

    for (unsigned long i = 0; i < ls_vector.size(); i++) {
      for (unsigned long j = i + 1; (j < ls_vector.size() && (visited[j] == 0));
           j++) {
        if (accfunmatch[i][j] &&
            (spatreuse[i][j] || (tempreuse[i][j] == perm[perm.size() - 1]) ||
             (tempreuse[j][i] == perm[perm.size() - 1]))) {
          if (visited[i] == 0) {
            groups.push_back(vector<int>());
            groups[groups.size() - 1].push_back(i);
            visited[i] = 1;
          }

          visited[j] = 1;
          groups[groups.size() - 1].push_back(j);
        }
      }
      if (visited[i] == 0) {
        groups.push_back(vector<int>());
        groups[groups.size() - 1].push_back(i);
      }
    }

    // calcualte total misses for this permutation
    vector<unsigned long> miss_vector;
    unsigned long misses = 0;
    for (auto i : groups) {
      unsigned long miss = ls_vector[i[0]].cacheMisses(perm, forOp);
      misses += miss;
      miss_vector.push_back(miss);
    }

    group_misses.push_back(make_pair(perm, miss_vector));
    group_list.push_back(make_pair(perm, groups));
    PermMisses.push_back(make_pair(misses, perm));
  }

  typedef struct node {
    SmallVector<AffineForOp, 4> loops;
    vector<struct node *> children;
    bool isConvertible;
    bool isPerfect;
  } LoopNode;

  // Takes input as outermost for loop of the for loop nest.
  // And builds tree with inner for loops as children and outermost for loop as
  // root of the tree. Along with that for every node stores the information if
  // that loopnest is perfect. if it is imperfect then finds if it is convertible
  // to perfect. Returns the root of the tree.
  LoopNode *BuildTree(AffineForOp &root) {
    LoopNode *loopnode = new LoopNode();
    getPerfectlyNestedLoops(loopnode->loops, root);

    loopnode->isPerfect = true;
    loopnode->isConvertible = true;

    AffineForOp innermostLoop = loopnode->loops[loopnode->loops.size() - 1];

    Block *block = innermostLoop.getBody();

    for (auto &op : *block) {
      // if innermost forloop contains for loop then not perfect
      if (isa<AffineForOp>(op) && innermostLoop != dyn_cast<AffineForOp>(op)) {
        loopnode->isPerfect = false;
      }

      // if contains 'if-then-else' then niether perfect nor convertible
      if (isa<AffineIfOp>(op)) {
        loopnode->isPerfect = false;
        loopnode->isConvertible = false;
        break;
      }

      // if contains load/store then not convertible
      if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
        loopnode->isConvertible = false;
      }
    }

    // recursive call to build subtrees for it's children for loops
    if ((loopnode->isPerfect == false) && loopnode->isConvertible) {
      for (auto &op : *block) {
        if (auto forOp = dyn_cast<AffineForOp>(op)) {
          if (innermostLoop != forOp) {
            loopnode->children.push_back(BuildTree(forOp));
          }
        }
      }
    }

    return loopnode;
  }

  // Takes input as outermost ForOp of for loopnest and tells if it is a perfect
  // nest.
  bool isPerfect(AffineForOp &root) {
    SmallVector<AffineForOp, 4> loops;
    getPerfectlyNestedLoops(loops, root);

    AffineForOp innermostLoop = loops[loops.size() - 1];

    Block *block = innermostLoop.getBody();

    for (auto &op : *block) {
      // if contains another for loop OR if/else statement then not perfect.
      if ((isa<AffineForOp>(op) &&
           innermostLoop != dyn_cast<AffineForOp>(op)) ||
          (isa<AffineIfOp>(op))) {
        return false;
      }
    }

    // bailout non-rectangular loops
    for (unsigned long i = 0; i < loops.size(); i++) {
      if (!loops[i].hasConstantLowerBound() ||
          !loops[i].hasConstantUpperBound())
        return false;
    }
    return true;
  }

  // Convert the imperfect loops to perfect by traversing the looptree
  void makePerfect(LoopNode *root, AffineForOp &op, int child) {
    AffineForOp newLoop = op;

    if (child != -1) {
      SmallVector<AffineForOp, 4> loops;
      getPerfectlyNestedLoops(loops, op);

      AffineForOp outermostLoop = loops[0];

      OpBuilder opb(outermostLoop.getOperation()->getBlock(),
                    std::next(Block::iterator(outermostLoop.getOperation())));
      newLoop = static_cast<AffineForOp>(opb.clone(*outermostLoop));

      SmallVector<AffineForOp, 4> newLoops;
      getPerfectlyNestedLoops(newLoops, newLoop);

      AffineForOp innermostLoop = newLoops[newLoops.size() - 1];

      int count = 0;
      innermostLoop.walk([&](Operation *op) {
        if (op->getParentOp() == innermostLoop && isa<AffineForOp>(op)) {
          if (count != child) {
            op->erase();
          }
          ++count;
        }
      });
    }

    if (!(root->isPerfect) && (root->isConvertible)) {
      for (int i = root->children.size() - 1; i >= 0; i--) {
        makePerfect(root->children[i], newLoop, i);
      }
      if (child != -1) {
        newLoop.erase();
      }
    }
  }
};
} // end anonymous namespace

void AffineLoopInterchange::runOnFunction() {

  vector<LoopNode *> looptree;

  // Build ForOp n-ary tree for each forloop nest
  for (auto &block : getFunction()) {
    for (auto &op : block) {
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        LoopNode *n = BuildTree(forOp);
        if (!(n->isPerfect)) {
          looptree.push_back(BuildTree(forOp));
        }
      }
    }
  }

  // Iterate over all the looptrees and convert all the convertible imperefect
  // loops into perfect loops by fission
  for (auto i : looptree) {
    if (!(i->isPerfect) && (i->isConvertible)) {
      makePerfect(i, i->loops[0], -1);
      i->loops[0].erase();
    }
  }

  // Find best permutation for all the perfect loopnests.
  for (auto &block : getFunction()) {
    for (auto &op : block) {
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        if (isPerfect(forOp)) {

          // Calculate access function for each load/store operation
          forOp.walk([&](Operation *op) {
            if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op))
              ls_vector.push_back(LoadStoreInfo(op));
          });

          // find all the load/store operations which has same access function
          find_same_acc_fun();

          // find all the load/store operations which exhibit spatial locality
          // in same iteration
          find_spatial_groups();

          // calculate dependences between all pairs of load/store ops
          calculateDependences();

          // find total possible permutations
          int total_perm = factorial(ls_vector[0].AccessFun[0].size() - 1);

          vector<int> perm;
          for (int i = 0, e = ls_vector[0].AccessFun[0].size() - 1; i < e;
               i++) {
            perm.push_back(i);
          }

          // calculate cache misses for each permutation
          for (int i = 0; i < total_perm; i++) {
            if (isValidPerm(perm)) {
              find_access_groups(perm, forOp);
            }
            next_permutation(perm.begin(), perm.end());
          }

          // find iterations which can be executed parallely
          find_par_loops();

          // find best permutation
          BestPerm.push_back(make_pair(forOp, find_best_perm()));

          ls_vector.clear();
          PermMisses.clear();
          spatreuse.clear();
          tempreuse.clear();
          accfunmatch.clear();
          parallel_loops.clear();
          Dependences.clear();
        }
      }
    }
  }

  // perform the interchange as per best perumtation for all the forloop nests
  // in the function.
  for (auto i : BestPerm) {
    SmallVector<AffineForOp, 4> l;
    getPerfectlyNestedLoops(l, i.first);
    permuteLoops(l, i.second);
  }
  BestPerm.clear();
}

namespace mlir {
void registerAffineLoopInterchange() {
  PassRegistration<AffineLoopInterchange> pass(
      "affine-loop-interchange", "Find most efficient loop permutation.");
}
} // namespace mlir
