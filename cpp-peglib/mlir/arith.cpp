#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include "arith.h"
#include "Passes.h"

bool EnablePass = true;
bool EnableInlinePass = true && EnablePass;
bool EnableLLVMPass = true && EnablePass;
bool DumpLLVMIR = true;
bool enableOpt = true;

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create target machine and configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(3 , /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::outs() << *llvmModule << "\n";
  return 0;
}


int arith_work(int first,int second,ArithOp type) {
  mlir::MLIRContext context;
  
  // Register dialects
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create function returning i32
  auto i32Type = builder.getI32Type();
  auto funcType = builder.getFunctionType({}, {i32Type});
  auto func = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), "main", funcType);
      
  // Push function to module immediately
  module->push_back(func);

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
        
        if(EnableLLVMPass) {
            pm.addPass(mlir::toy::createLowerToLLVMPass());
        }

        if (mlir::failed(pm.run(*module))) {
            llvm::errs() << "Failed to lower to LLVM.\n";
            return 1;
        }
    }

    if(EnableLLVMPass && DumpLLVMIR){
        dumpLLVMIR(*module);
    } else {
        module->print(llvm::outs());
    }
    return 0;
}