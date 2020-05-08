//===- TestMemRefDependenceCheck.cpp - Test dep analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to run pair-wise memref access dependence checks.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "iostream"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"

#define DEBUG_TYPE "test-memref-dependence-check1"

using namespace mlir;
using namespace std;

namespace {

// TODO(andydavis) Add common surrounding loop depth-wise dependence checks.
/// Checks dependences between all pairs of memref accesses in a Function.
struct TestMemRefDependenceCheck1
    : public PassWrapper<TestMemRefDependenceCheck1, FunctionPass> {
  SmallVector<Operation *, 4> loadsAndStores;
  SmallVector<Operation *, 4> forOps;
  void runOnFunction() override;
};

} // end anonymous namespace

// Returns a result string which represents the direction vector (if there was
// a dependence), returns the string "false" otherwise.
static std::string
getDirectionVectorStr(bool ret, unsigned numCommonLoops, unsigned loopNestDepth,
                      ArrayRef<DependenceComponent> dependenceComponents) {
  if (!ret)
    return "false";
  if (dependenceComponents.empty() || loopNestDepth > numCommonLoops)
    return "true";
  std::string result;
  for (unsigned i = 0, e = dependenceComponents.size(); i < e; ++i) {
    std::string lbStr = "-inf";
    if (dependenceComponents[i].lb.hasValue() &&
        dependenceComponents[i].lb.getValue() !=
            std::numeric_limits<int64_t>::min())
      lbStr = std::to_string(dependenceComponents[i].lb.getValue());

    std::string ubStr = "+inf";
    if (dependenceComponents[i].ub.hasValue() &&
        dependenceComponents[i].ub.getValue() !=
            std::numeric_limits<int64_t>::max())
      ubStr = std::to_string(dependenceComponents[i].ub.getValue());

    result += "[" + lbStr + ", " + ubStr + "]";
  }
  return result;
}

// Computes the iteration domain for 'opInst' and populates 'indexSet', which
// encapsulates the constraints involving loops surrounding 'opInst' and
// potentially involving any Function symbols. The dimensional identifiers in
// 'indexSet' correspond to the loops surrounding 'op' from outermost to
// innermost.
// TODO(andydavis) Add support to handle IfInsts surrounding 'op'.
static LogicalResult getInstIndexSet(Operation *op,
                                     FlatAffineConstraints *indexSet) {
  // TODO(andydavis) Extend this to gather enclosing IfInsts and consider
  // factoring it out into a utility function.
  SmallVector<AffineForOp, 4> loops;
  getLoopIVs(*op, &loops);
  return getIndexSet(loops, indexSet);
}



// ValuePositionMap manages the mapping from Values which represent dimension
// and symbol identifiers from 'src' and 'dst' access functions to positions
// in new space where some Values are kept separate (using addSrc/DstValue)
// and some Values are merged (addSymbolValue).
// Position lookups return the absolute position in the new space which
// has the following format:
//
//   [src-dim-identifiers] [dst-dim-identifiers] [symbol-identifiers]
//
// Note: access function non-IV dimension identifiers (that have 'dimension'
// positions in the access function position space) are assigned as symbols
// in the output position space. Convenience access functions which lookup
// an Value in multiple maps are provided (i.e. getSrcDimOrSymPos) to handle
// the common case of resolving positions for all access function operands.
//
// TODO(andydavis) Generalize this: could take a template parameter for
// the number of maps (3 in the current case), and lookups could take indices
// of maps to check. So getSrcDimOrSymPos would be "getPos(value, {0, 2})".
class ValuePositionMap1 {
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
// Builds a map from Value to identifier position in a new merged identifier
// list, which is the result of merging dim/symbol lists from src/dst
// iteration domains, the format of which is as follows:
//
//   [src-dim-identifiers, dst-dim-identifiers, symbol-identifiers, const_term]
//
// This method populates 'valuePosMap' with mappings from operand Values in
// 'srcAccessMap'/'dstAccessMap' (as well as those in 'srcDomain'/'dstDomain')
// to the position of these values in the merged list.
static void buildDimAndSymbolPositionMaps1(
    const FlatAffineConstraints &srcDomain,
    const AffineValueMap &srcAccessMap,
    ValuePositionMap1 *valuePosMap,
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
				/*nothing*/
      }
    }
  };

  SmallVector<Value, 4> srcValues;
  srcDomain.getIdValues(0, srcDomain.getNumDimAndSymbolIds(), &srcValues);
  // Update value position map with identifiers from src iteration domain.
  updateValuePosMap(srcValues, /*isSrc=*/true);
  // Update value position map with identifiers from src access function.
  updateValuePosMap(srcAccessMap.getOperands(), /*isSrc=*/true);
}


// Sets up dependence constraints columns appropriately, in the format:
// [src-dim-ids, dst-dim-ids, symbol-ids, local-ids, const_term]
static void initDependenceConstraints1(
    const FlatAffineConstraints &srcDomain,
     const AffineValueMap &srcAccessMap,
     const ValuePositionMap1 &valuePosMap,
    FlatAffineConstraints *dependenceConstraints) {
  // Calculate number of equalities/inequalities and columns required to
  // initialize FlatAffineConstraints for 'dependenceDomain'.
  unsigned numIneq = srcDomain.getNumInequalities();// + dstDomain.getNumInequalities();
  AffineMap srcMap = srcAccessMap.getAffineMap();
  //assert(srcMap.getNumResults() == dstAccessMap.getAffineMap().getNumResults());
  unsigned numEq = srcMap.getNumResults();
  unsigned numDims = srcDomain.getNumDimIds(); 
  unsigned numSymbols = 0;//valuePosMap.getNumSymbols();
  unsigned numLocals = 0;//srcDomain.getNumLocalIds();  
  unsigned numIds = numDims + numSymbols + numLocals;
  unsigned numCols = numIds + 1;

  // Set flat affine constraints sizes and reserving space for constraints.
  dependenceConstraints->reset(numIneq, numEq, numCols, numDims, numSymbols, numLocals);

  // Set values corresponding to dependence constraint identifiers.
  SmallVector<Value, 4> srcLoopIVs;
  srcDomain.getIdValues(0, srcDomain.getNumDimIds(), &srcLoopIVs);

  dependenceConstraints->setIdValues(0, srcLoopIVs.size(), srcLoopIVs);
 // dependenceConstraints->setIdValues(srcLoopIVs.size(), srcLoopIVs.size() + dstLoopIVs.size(), dstLoopIVs);

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
  srcDomain.getIdValues(srcDomain.getNumDimIds(), srcDomain.getNumDimAndSymbolIds(), &srcSymbolValues);
  setSymbolIds(srcSymbolValues);

  for (unsigned i = 0, e = dependenceConstraints->getNumDimAndSymbolIds();
       i < e; i++)
    assert(dependenceConstraints->getIds()[i].hasValue());
}


static LogicalResult
addMemRefAccessConstraints1(const AffineValueMap &srcAccessMap,
                           const ValuePositionMap1 &valuePosMap,
                           FlatAffineConstraints *dependenceDomain) {
  AffineMap srcMap = srcAccessMap.getAffineMap();
  //assert(srcMap.getNumResults() == dstMap.getNumResults());
  unsigned numResults = srcMap.getNumResults();

//  unsigned srcNumIds = srcMap.getNumDims() + srcMap.getNumSymbols();
  ArrayRef<Value> srcOperands = srcAccessMap.getOperands();

  std::vector<SmallVector<int64_t, 8>> srcFlatExprs;
  FlatAffineConstraints srcLocalVarCst;
  // Get flattened expressions for the source destination maps.
  if (failed(getFlattenedAffineExprs(srcMap, &srcFlatExprs, &srcLocalVarCst)))
    return failure();

//  unsigned domNumLocalIds = dependenceDomain->getNumLocalIds();
  unsigned srcNumLocalIds = srcLocalVarCst.getNumLocalIds();
  unsigned numLocalIdsToAdd = srcNumLocalIds;
  for (unsigned i = 0; i < numLocalIdsToAdd; i++) {
    dependenceDomain->addLocalId(dependenceDomain->getNumLocalIds());
  }

//  unsigned numDims = dependenceDomain->getNumDimIds();
//  unsigned numSymbols = dependenceDomain->getNumSymbolIds();
//  unsigned numSrcLocalIds = srcLocalVarCst.getNumLocalIds();
//  unsigned newLocalIdOffset = numDims + numSymbols + domNumLocalIds;

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
    // Local terms.
  /*  for (unsigned j = 0, e = srcNumLocalIds; j < e; j++)
      eq[newLocalIdOffset + j] = srcFlatExpr[srcNumIds + j];*/
    // Set constant term.
    eq[eq.size() - 1] = srcFlatExpr[srcFlatExpr.size() - 1];

    // Add equality constraint.
    dependenceDomain->addEquality(eq);
	cout<<"printing eq"<<endl;
	for(auto i:eq)
	{
		cout<<i<<" ";
	}
	cout<<endl;
  }



/*  // Add equality constraints for any operands that are defined by constant ops.
  auto addEqForConstOperands = [&](ArrayRef<Value> operands) {
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      if (isForInductionVar(operands[i]))
        continue;
      auto symbol = operands[i];
      assert(isValidSymbol(symbol));
      // Check if the symbol is a constant.
      if (auto cOp = dyn_cast_or_null<ConstantIndexOp>(symbol.getDefiningOp()))
        dependenceDomain->setIdToConstant(valuePosMap.getSymPos(symbol),
                                          cOp.getValue());
    }
  };

  // Add equality constraints for any src symbols defined by constant ops.
  addEqForConstOperands(srcOperands);*/

  // By construction (see flattener), local var constraints will not have any
  // equalities.
/*  assert(srcLocalVarCst.getNumEqualities() == 0);

  // Add inequalities from srcLocalVarCst and destLocalVarCst into the
  // dependence domain.
  SmallVector<int64_t, 8> ineq(dependenceDomain->getNumCols());
  for (unsigned r = 0, e = srcLocalVarCst.getNumInequalities(); r < e; r++) {
    std::fill(ineq.begin(), ineq.end(), 0);

    // Set identifier coefficients from src local var constraints.
    for (unsigned j = 0, e = srcOperands.size(); j < e; ++j)
      ineq[valuePosMap.getSrcDimOrSymPos(srcOperands[j])] =
          srcLocalVarCst.atIneq(r, j);
    // Local terms.
    for (unsigned j = 0, e = srcNumLocalIds; j < e; j++)
      ineq[newLocalIdOffset + j] = srcLocalVarCst.atIneq(r, srcNumIds + j);
    // Set constant term.
    ineq[ineq.size() - 1] =
        srcLocalVarCst.atIneq(r, srcLocalVarCst.getNumCols() - 1);
    dependenceDomain->addInequality(ineq);
  }*/

  return success();
}

// Walks the Function 'f' adding load and store ops to 'loadsAndStores'.
// Runs pair-wise dependence checks.
void TestMemRefDependenceCheck1::runOnFunction() {
  // Collect the loads and stores within the function.
  loadsAndStores.clear();
  getFunction().walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op))
      loadsAndStores.push_back(op);
	
    if (isa<AffineForOp>(op))
		forOps.push_back(op);
  });

	for(int k=0;k<loadsAndStores.size();k++)
{
	FlatAffineConstraints dependenceConstraints;
	
	cout<<"load store : "<<k<<endl;
	loadsAndStores[k]->dump();
	cout<<"-----------------------------------------"<<endl;	

	auto *srcOpInst = loadsAndStores[k];
   MemRefAccess srcAccess(srcOpInst);

  // Get composed access function for 'srcAccess'.
   AffineValueMap srcAccessMap;
   srcAccess.getAccessMap(&srcAccessMap);

  // Get iteration domain for the 'srcAccess' operation.
  FlatAffineConstraints srcDomain;
  if (failed(getInstIndexSet(srcAccess.opInst, &srcDomain))) ; 
    // return DependenceResult::Failure;

  // Build dim and symbol position maps for each access from access operand
  // Value to position in merged constraint system.
  ValuePositionMap1 valuePosMap;
 
	buildDimAndSymbolPositionMaps1(srcDomain, srcAccessMap, &valuePosMap, &dependenceConstraints);
  initDependenceConstraints1(srcDomain, srcAccessMap, valuePosMap, &dependenceConstraints);

//  assert(valuePosMap.getNumDims() ==
  //       srcDomain.getNumDimIds() + dstDomain.getNumDimIds());

  // Create memref access constraint by equating src/dst access functions.
  // Note that this check is conservative, and will fail in the future when
  // local variables for mod/div exprs are supported.
  if (failed(addMemRefAccessConstraints1(srcAccessMap, valuePosMap, &dependenceConstraints)));
 //   return DependenceResult::Failure;


	dependenceConstraints.dump();

	cout<<"***************************************************"<<endl;	
}

/*	auto *srcOpInst = loadsAndStores[0];
   MemRefAccess srcAccess(srcOpInst);

  auto *dstOpInst = loadsAndStores[1];
	loadsAndStores[1]->dump();
   MemRefAccess dstAccess(dstOpInst);

   SmallVector<DependenceComponent, 2> depComps;
	int d=2;
   DependenceResult result = checkMemrefAccessDependence(srcAccess, dstAccess, d, &dependenceConstraints, &depComps);*/

	
   

 // checkDependences(loadsAndStores);
}

namespace mlir {
void registerTestMemRefDependenceCheck1() {
  PassRegistration<TestMemRefDependenceCheck1> pass(
      "affine-loop-interchange",
      "Find most efficient loop permutation.");
}
} // namespace mlir

