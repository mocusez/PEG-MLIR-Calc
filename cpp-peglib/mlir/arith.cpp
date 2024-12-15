#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

#include "arith.h"

bool EnablePass = true;
bool EnableInlinePass = true;

int arith_work(int first,int second,ArithOp type) {
  mlir::MLIRContext context;
  
  // Register dialects
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create function returning i32
  auto i32Type = builder.getI32Type();
  auto funcType = builder.getFunctionType({}, {i32Type});
  auto func = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), "main", funcType);
      
  // Push function to module immediately
  module.push_back(func);

  // Create entry block and set insertion point
  auto entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto one = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), i32Type,
      builder.getI32IntegerAttr(first));

  auto two = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), i32Type,
      builder.getI32IntegerAttr(second));


  mlir::Value result;
  if(type == ArithOp::ADD){
      result = builder.create<mlir::arith::AddIOp>(
      builder.getUnknownLoc(), one, two);
  } else if(type == ArithOp::MUL){
      result = builder.create<mlir::arith::MulIOp>(
      builder.getUnknownLoc(), one, two);
  }

  builder.create<mlir::func::ReturnOp>(
      builder.getUnknownLoc(), 
      mlir::ValueRange{result}); 


    if(EnablePass){
        mlir::PassManager pm(&context);
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if(EnableInlinePass){
            pm.addPass(mlir::createInlinerPass());
        }
        if (mlir::failed(pm.run(module))) {
            llvm::errs() << "Failed to lower to LLVM.\n";
            return 1;
        }
    }
    

    module.print(llvm::outs());
    return 0;
}