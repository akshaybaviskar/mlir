#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Transforms/LoopUtils.h"
#include "iostream"
#include <unordered_map>

using namespace mlir;
using namespace std;

#define DEBUG 1
#define CACHE_SIZE 8

namespace {
#if 1
// Computes the iteration domain for 'opInst' and populates 'indexSet', which
// encapsulates the constraints involving loops surrounding 'opInst' and
// potentially involving any Function symbols. The dimensional identifiers in
// 'indexSet' correspond to the loops surrounding 'op' from outermost to
// innermost.
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

static void buildDimAndSymbolPositionMaps(const FlatAffineConstraints &srcDomain, const AffineValueMap &srcAccessMap, ValuePositionMap *valuePosMap, FlatAffineConstraints *dependenceConstraints)
{
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
  updateValuePosMap(srcValues, /*isSrc=*/true);
  // Update value position map with identifiers from src access function.
  updateValuePosMap(srcAccessMap.getOperands(), /*isSrc=*/true);
}

// Sets up dependence constraints columns appropriately, in the format:
// [src-dim-ids, dst-dim-ids, symbol-ids, local-ids, const_term]
static void initDependenceConstraints(
    const FlatAffineConstraints &srcDomain,
     const AffineValueMap &srcAccessMap,
     const ValuePositionMap &valuePosMap,
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

	#ifdef DEBUG
	//cout<<"dims: "<<numDims<<" symbols: "<< numSymbols<<" locals: "<<numLocals<<endl; 
	#endif

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
addMemRefAccessConstraints(const AffineValueMap &srcAccessMap,
                           const ValuePositionMap &valuePosMap,
                           FlatAffineConstraints *dependenceDomain,
									vector<vector<int>> &AccessFun) {
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
    // Set constant term./
    eq[eq.size() - 1] = srcFlatExpr[srcFlatExpr.size() - 1];

    // Add equality constraint.
    dependenceDomain->addEquality(eq);

	 vector<int> eq_copy(dependenceDomain->getNumCols());	

	#ifdef DEBUG
/*	cout<<"dependenceDomain : "<<dependenceDomain->getNumCols()<<endl;
	cout<<"eq size: "<<eq.size()<<endl;*/
	#endif

	for(unsigned long i=0;i<eq.size();i++)
	{
		eq_copy[i] = eq[i];
	}
	
	AccessFun.push_back(eq_copy);
	//cout<<endl;
  }

  return success();
}

struct AffineLoopInterchange : public PassWrapper<AffineLoopInterchange, FunctionPass>
{
  SmallVector<Operation *, 4> loadsAndStores;
  SmallVector<Operation*,4> forOps;
  void runOnFunction() override;

vector<vector<int>> accfunmatch;
unordered_map<int,bool> parallel_loops;
vector<pair<unsigned long ,vector<int>>> PermMisses;
vector<vector<int>> spatreuse;
vector<vector<int>> tempreuse; 
vector<vector<int>> Dependences;
vector<pair<AffineForOp, vector<unsigned int>>> BestPerm;
class LoadStoreInfo;
vector<LoadStoreInfo> ls_vector;
class LoadStoreInfo
{
public:
	Operation* op;
  MemRefAccess srcAccess;
	AffineValueMap srcAccessMap;
	FlatAffineConstraints srcDomain;
   ValuePositionMap valuePosMap;
	FlatAffineConstraints dependenceConstraints; 
	vector<vector<int>> AccessFun;
	
	LoadStoreInfo(Operation *opn):srcAccess(opn)
	{
		op = opn;

	  // Get composed access function for 'srcAccess'.
	   srcAccess.getAccessMap(&srcAccessMap);
	
	  // Get iteration domain for the 'srcAccess' operation.
	  if (failed(getInstIndexSet(srcAccess.opInst, &srcDomain)))  
	  {
			cout<<" Unable to find iteration domain for: "<<endl;
			op->dump();
			exit(1);
		  // return DependenceResult::Failure;
	  }

	  // Build dim and symbol position maps for each access from access operand
     // Value to position in merged constraint system.
   	buildDimAndSymbolPositionMaps(srcDomain, srcAccessMap, &valuePosMap, &dependenceConstraints);
      initDependenceConstraints(srcDomain,srcAccessMap, valuePosMap, &dependenceConstraints);

    // Create memref access constraint by equating src/dst access functions.
	 // Note that this check is conservative, and will fail in the future when
	 // local variables for mod/div exprs are supported.
		if (failed(addMemRefAccessConstraints(srcAccessMap, valuePosMap, &dependenceConstraints, AccessFun)))
		{
			cout<<"Create MemRefAccess constraint failed for:"<<endl;
			op->dump();
			exit(1);
		}
	}

	void print_acc_fun()
	{
		for(auto i:AccessFun)
		{
			for(auto j:i)
			{
				cout<<j<<"  ";
			}
			cout<<endl;
		}
	}

	// TODO : add stride information and read 2048 from loop information
	unsigned long cacheMisses(vector<int> perm)
	{
		unsigned long misses = 1;
		int power = 0;
		int div = 0;

		int pivotfound = 0;
		for(int i=perm.size()-1;i>=0;i--)
		{
			if(pivotfound == 1)
			{
				power++;
				continue;
			}

			int col = perm[i];
			unsigned long row; 
			int allZero = 1;
			for(row = 0; row < AccessFun.size() - 1; row++)
			{
				if(AccessFun[row][col] == 0)
				{
					continue;
				}
				else
				{
					pivotfound = 1;
					allZero = 0;
					power++;
					break;
				}
			}
		
			if(allZero)
			{
				if(AccessFun[row][col] == 0)
				{
					continue;
				}
				else if(abs(AccessFun[row][col]) < CACHE_SIZE)
				{
					div = CACHE_SIZE/abs(AccessFun[row][col]);
					pivotfound = 1;
					power++;
				}
			}	

		}

	//	cout<<"(n^"<<power<<")";
		for(int i=0;i<power;i++)
		{
			misses = misses * 2048;
		}				
		if(div)
		{
			misses = misses/div;
		//	cout<<"/"<<div<<endl;
		}

		return misses;
	}
};



int factorial(int n) 
{ 
    return (n==1 || n==0) ? 1: n * factorial(n - 1);  
}

void calculateDependences()
{
	tempreuse = vector<vector<int>> (ls_vector.size(), vector<int>(ls_vector.size()));

	for (unsigned i = 0, e = ls_vector.size(); i < e; ++i)
	{
		for (unsigned j = 0; j < e; ++j)
		{
			if(i==j)
			{
				tempreuse[i][j] = 1;
				continue;
			}
			tempreuse[i][j] = -1;

			
			unsigned numCommonLoops = getNumCommonSurroundingLoops(*ls_vector[i].op, *ls_vector[j].op);
			for (unsigned d = 1; d <= numCommonLoops + 1; ++d)
			{
				FlatAffineConstraints dependenceConstraints;
				SmallVector<DependenceComponent, 2> dependenceComponents;
				DependenceResult result = checkMemrefAccessDependence(ls_vector[i].srcAccess, ls_vector[j].srcAccess, d, &dependenceConstraints,&dependenceComponents,true);
				assert(result.value != DependenceResult::Failure);
				bool ret = hasDependence(result);

				if(ret)
				{
					if (dependenceComponents.empty() || d > numCommonLoops)
					continue;
		
					vector<int> dep(dependenceComponents.size());
 
					for (unsigned k = 0, e = dependenceComponents.size(); k < e; ++k) 
					{
						 long lb = -1;
						 if (dependenceComponents[k].lb.hasValue() && dependenceComponents[k].lb.getValue() != std::numeric_limits<int64_t>::min())
  						    lb = dependenceComponents[k].lb.getValue();

  						  long ub = 1;
  						  if (dependenceComponents[k].ub.hasValue() && dependenceComponents[k].ub.getValue() != std::numeric_limits<int64_t>::max())
  						    ub = dependenceComponents[k].ub.getValue();

  						 if( (lb == 0) && (ub == 0))
  						 {
  						 	dep[k] = 0;
  						 }
  						 else if((lb>=0) && (ub>=0))
  						 {
  						 	dep[k] = 1;
  						 }
  						 else
  						 {
  						 	dep[k] = -1;
  						 }
  					}

					//add in dependencies list if not RAR
					if (isa<AffineStoreOp>(ls_vector[i].op) || isa<AffineStoreOp>(ls_vector[j].op))
					Dependences.push_back(dep);

					int dir_exist = 0;
					int dir;
					for(unsigned long l = 0; l<dep.size() ; l++)
					{
						if(dep[l] == 0)
						{
							continue;
						}
						else
						{
							if(dir_exist)
							{
								dir_exist = 0;
								break;
							}
							else
							{
								dir_exist = 1;
								dir = l;
							}
						}
					}

					if((dir_exist ) && (accfunmatch[i][j] == 1))
					{
						tempreuse[i][j] = dir;
					}
				}
			}
		}
	}
}

void find_par_loops()
{
	for(unsigned long i=0;i<ls_vector[0].AccessFun[0].size()-1;i++)
	{
		parallel_loops[i] = true;
	}

	for(auto i:Dependences)
	{
		for(unsigned long j=0;j<i.size();j++)
		{
			if(i[j] != 0)
			{
				parallel_loops.erase(j);
				break;
			}
		}
	}
}

vector<unsigned> find_best_perm()
{
	sort(PermMisses.begin(), PermMisses.end());
	unsigned long last = PermMisses[0].first;
	int best = 0;
	for(unsigned int i=0; i<PermMisses.size();i++)
	{
		if(PermMisses[i].first != last) break;
		if(parallel_loops.count(PermMisses[i].second[0]) == 1)
		{
			best = i;
			break;
		}
	}

	std::vector<unsigned int> permMap(PermMisses[best].second.size());
   for (unsigned inx = 0; inx < PermMisses[best].second.size(); ++inx) {
		 permMap[PermMisses[best].second[inx]] = inx;
   }	

	return permMap;
}

bool isValidPerm(vector<int> perm)
{
	if(!Dependences.size()) return true; 
	for(auto i:Dependences)
	{
		for(unsigned long  j=0; j<i.size(); j++)
		{
			if(i[perm[j]]>0) break;
			if(i[perm[j]]<0) return false;
		}
	}
	return true;
}

void find_same_acc_fun()
{
	accfunmatch = vector<vector<int>> (ls_vector.size(), vector<int>(ls_vector.size()));
	for(unsigned long i = 0; i<ls_vector.size();i++)
	{
		for(unsigned long j = i;j<ls_vector.size();j++)
		{
			if(i==j)
			{
				accfunmatch[i][j] = 1;
				continue;
			}

			if (ls_vector[i].srcAccess.memref != ls_vector[j].srcAccess.memref)
			{
				accfunmatch[i][j] = 0;
				accfunmatch[j][i] = 0;
				continue;
			}
			
			int same_accfun = 1;
			for(unsigned long row=0; row<ls_vector[i].AccessFun.size(); row++)
			{
				for(unsigned long col=0; col<ls_vector[i].AccessFun[0].size()-1; col++)
				{
					if(ls_vector[i].AccessFun[row][col] != ls_vector[j].AccessFun[row][col])
					{
						same_accfun = 0;
						break;
					}
				}
				if(same_accfun == 0) break;
			}

			accfunmatch[i][j] = same_accfun;
			accfunmatch[j][i] = same_accfun;
		}
	}
}


void find_spatial_groups()
{
	spatreuse = vector<vector<int>> (ls_vector.size(), vector<int>(ls_vector.size()));

	for(unsigned long i = 0; i<ls_vector.size();i++)
	{
		for(unsigned long j = i	;j<ls_vector.size() ;j++)
		{
			if(i==j)
			{
				spatreuse[i][j] = 1;
				continue;
			}

			int spat_group = 1;
			if(accfunmatch[i][j])
			{
				unsigned long row;
				unsigned long col = ls_vector[i].AccessFun[0].size()-1;
				for(row=0; row<ls_vector[i].AccessFun.size()-1; row++)
				{
						if((ls_vector[i].AccessFun[row][col] - ls_vector[j].AccessFun[row][col]) != 0)
						{
							spat_group = 0;
							break;
						}
				}

				if(spat_group == 1)
				{
					if(abs(ls_vector[i].AccessFun[row][col] - ls_vector[j].AccessFun[row][col]) >= CACHE_SIZE)
					{
						spat_group = 0;
					}
				}
				
				spatreuse[i][j] = spat_group;
				spatreuse[j][i] = spat_group;
			}
		}
	}
}
	
/*finds access groups for given permutation*/
void find_access_groups(vector<int> perm)
{
	vector<vector<int>> groups;
	vector<int> visited(ls_vector.size());

	for(unsigned long i = 0; i<ls_vector.size();i++)
	{
		for(unsigned long j = i+1;(j<ls_vector.size() && (visited[j] == 0));j++)
		{
			if(accfunmatch[i][j] &&(spatreuse[i][j] || (tempreuse[i][j] == perm[perm.size() - 1]) || (tempreuse[j][i] == perm[perm.size() - 1])))
			{
				if(visited[i] == 0)
				{
					groups.push_back(vector<int>());
					groups[groups.size()-1].push_back(i);
					visited[i] = 1;
				}

				visited[j] = 1;
				groups[groups.size()-1].push_back(j);
			}
		}
		if(visited[i] == 0)
		{
			groups.push_back(vector<int>());
			groups[groups.size()-1].push_back(i);	
		}
	}

	//calcualte total misses for this permutation
	unsigned long misses = 0;
	for(auto i:groups)
	{
		misses += ls_vector[i[0]].cacheMisses(perm);
	}

	PermMisses.push_back(make_pair(misses, perm));	
}

typedef struct node
{
	SmallVector<AffineForOp, 4> loops;
	vector<struct node*> children;
	bool isConvertible;
	bool isPerfect;

}LoopNode;

LoopNode* BuildTree(AffineForOp& root)
{
	LoopNode* loopnode = new LoopNode();
	getPerfectlyNestedLoops(loopnode->loops, root);

	loopnode->isPerfect = true;
	loopnode->isConvertible = true;

	AffineForOp innermostLoop = loopnode->loops[loopnode->loops.size() - 1];

	Block* block = innermostLoop.getBody();
 
     for (auto &op : *block)
     {
    	//if contains no for loop then already perfect
        if (isa<AffineForOp>(op) && innermostLoop != dyn_cast<AffineForOp>(op))
		  {
          loopnode->isPerfect = false;
        }
		
		  //if contains if or load store statements then not convertible
        if (isa<AffineIfOp>(op)) 
		  {
				loopnode->isPerfect = false;
				loopnode->isConvertible = false;	
				break;
		  }

		  if(isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op))
		  {
				loopnode->isConvertible = false;
		  }
		}

	if((loopnode->isPerfect == false) && loopnode->isConvertible)
	{
		for(auto &op : *block)
		{
			if(auto forOp = dyn_cast<AffineForOp>(op))
			{
				if(innermostLoop != forOp)
				{
					loopnode->children.push_back(BuildTree(forOp));
				}
			}
		}
	}

	return loopnode;
}

bool isPerfect(AffineForOp& root)
{
	SmallVector<AffineForOp, 4> loops;
	getPerfectlyNestedLoops(loops, root);

	AffineForOp innermostLoop = loops[loops.size() - 1];

	Block* block = innermostLoop.getBody();
 
     for (auto &op : *block)
     {
    	//if contains another for loop OR if/else statement then not perfect.
        if ((isa<AffineForOp>(op) && innermostLoop != dyn_cast<AffineForOp>(op)) || (isa<AffineIfOp>(op)))
		  {
          return false;
        }
		}
	return true;
}

void makePerfect(LoopNode* root, AffineForOp& op, int child)
{
	AffineForOp newLoop = op;

	if(child != -1)
	{
		SmallVector<AffineForOp, 4> loops;
		getPerfectlyNestedLoops(loops, op);
	
		AffineForOp outermostLoop = loops[0];
	
		OpBuilder opb(outermostLoop.getOperation()->getBlock(), std::next(Block::iterator(outermostLoop.getOperation())));
	   newLoop = static_cast<AffineForOp>(opb.clone(*outermostLoop));
	
		SmallVector<AffineForOp, 4> newLoops;
		getPerfectlyNestedLoops(newLoops, newLoop);
	
		AffineForOp innermostLoop = newLoops[newLoops.size() - 1];

		int count = 0;
		innermostLoop.walk([&](Operation *op) {
          if (op->getParentOp() == innermostLoop && isa<AffineForOp>(op)) {
				if(count != child){
              op->erase();
				}
				++count;
          }
        });
	}

	if(!(root->isPerfect) && (root->isConvertible))
	{
		for(int i=root->children.size()-1; i>=0 ; i--)
		{
			makePerfect(root->children[i], newLoop, i);
		}
		if(child != -1)
		{
			newLoop.erase();
		}
	}
}
#endif

vector<SmallVector<AffineForOp, 4>> aksloops1;
void getLoopNests(AffineForOp forOp) {
  auto getPerfectLoopNests = [&](AffineForOp root) {
		SmallVector<AffineForOp, 4> l;
      aksloops1.clear();
		aksloops1.resize(1);
      getPerfectlyNestedLoops(l, root);
		aksloops1[0].resize(l.size());
		for(unsigned long i=0;i<l.size();i++)
		{
			aksloops1[0][i] = l[i];
		}
  };
  getPerfectLoopNests(forOp);
}
};
} // end anonymous namespace

void AffineLoopInterchange::runOnFunction() {

	vector<LoopNode*> looptree;

	//Build ForOp trees 
	for (auto &block : getFunction())
   {
		 for (auto &op : block)
		 {
			if (auto forOp = dyn_cast<AffineForOp>(op))
			{
				LoopNode* n = BuildTree(forOp);
				if(!(n->isPerfect))
				{
					looptree.push_back(BuildTree(forOp));
				}
			}
	    }
	}

	//Try to convert all the imperefect loops into perfect loops.
	for(auto i:looptree)
	{
		if(!(i->isPerfect) && (i->isConvertible))
		{
			makePerfect(i,i->loops[0], -1);
			i->loops[0].erase();
		}
	}
	//Process all the perfect loops.
	for (auto &block : getFunction())
   {
		 for (auto &op : block)
		 {
			if (auto forOp = dyn_cast<AffineForOp>(op))
			{
				if(isPerfect(forOp))
				{
		
					forOp.walk([&](Operation *op) {
						 if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op))
							ls_vector.push_back(LoadStoreInfo(op));
					 });
					
					find_same_acc_fun();
					find_spatial_groups();
					calculateDependences();

					//find total possible permutations
					int total_perm = factorial(ls_vector[0].AccessFun[0].size()-1);
					vector<int> perm;
				
					for(int i=0, e=ls_vector[0].AccessFun[0].size()-1;i<e ; i++)
					{
					//	cout<<"push"<<endl;
						perm.push_back(i);
					}
				
					for(int i = 0; i<total_perm; i++)
					{
						if(isValidPerm(perm))
						{
							find_access_groups(perm);
						}
						next_permutation(perm.begin(), perm.end());
					}	
					
					//find iterations which can be executed parallely
					find_par_loops();

					//find best permutation
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

	for(auto i:BestPerm)
	{
		getLoopNests(i.first);
		permuteLoops(aksloops1[0],i.second);
	}
	BestPerm.clear();
}

namespace mlir {
void registerAffineLoopInterchange() {
  PassRegistration<AffineLoopInterchange> pass(
      "affine-loop-interchange",
      "Find most efficient loop permutation.");
}
} // namespace mlir
