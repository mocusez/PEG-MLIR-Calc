#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include "arith.h"
#include "Passes.h"

#include <vector>
#include <algorithm>
#include <xmmintrin.h> // For __m128i

bool EnablePass = true;
bool EnableInlinePass = true && EnablePass;
bool EnableLLVMPass = true && EnablePass;
bool DumpLLVMIR = true;
bool enableOpt = true;
bool runJIT = true;

int dumpLLVMIR(mlir::ModuleOp module,bool enableJIT=false,bool simd_output=false) {
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

    if(enableJIT){
        llvm::ExitOnError ExitOnErr;
        auto J = ExitOnErr(llvm::orc::LLJITBuilder().create());
        ExitOnErr(J->addIRModule(llvm::orc::ThreadSafeModule(std::move(llvmModule), std::make_unique<llvm::LLVMContext>())));
        auto MainSymbol = ExitOnErr(J->lookup("main"));
        auto normalJIT = [&MainSymbol](){
            auto *main = MainSymbol.toPtr<int()>();
            llvm::outs() << main() << "\n";
        };
        auto simdJIT = [&MainSymbol](){
            using VectorFnType = __m128i (*)();
            auto *mainFn = MainSymbol.toPtr<VectorFnType>();
            const size_t size = 4;
            int32_t result[size];
            __m128i vec_result = mainFn();
            _mm_store_si128((__m128i*)result, vec_result);
            
            llvm::outs() << "[";
            for(size_t i = 0; i < size; i++) {
                llvm::outs() << result[i];
                if (i < size - 1) llvm::outs() << ", ";
            }
            llvm::outs() << "]\n";
        };
        if(simd_output){
            simdJIT();
        } else {
            normalJIT();
        }
    } else{
        llvm::outs() << *llvmModule << "\n";
    }
    return 0;
}

int simd_work(const std::vector<int> &values1,const std::vector<int> &values2){
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::vector::VectorDialect>();
    context.loadDialect<mlir::arith::ArithDialect>(); 

    mlir::OpBuilder builder(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());

    if(values1.size() != values2.size()){
        llvm::errs() << "Wrong Vector Caculator!" << "\n";
    }
    auto vectorType = mlir::VectorType::get(
        {static_cast<int64_t>(values1.size())}, 
        builder.getIntegerType(32));

    auto funcType = builder.getFunctionType({}, {vectorType});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);

    module->push_back(func);
    auto entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto stdVector2Value = [&builder,&vectorType](const std::vector<int> &values) { 
        std::vector<mlir::Attribute> attrs(values.size());
        auto trans = [&builder](int val) { 
            return builder.getI32IntegerAttr(val);
        };
        std::transform(values.begin(), values.end(), attrs.begin(), trans);
        return mlir::DenseElementsAttr::get(vectorType, attrs);
    };

    mlir::Value vecConstant[] = {
        builder.create<mlir::arith::ConstantOp>( // Changed to arith::ConstantOp
            builder.getUnknownLoc(),
            vectorType,
            stdVector2Value(values1)),
        builder.create<mlir::arith::ConstantOp>( // Changed to arith::ConstantOp
            builder.getUnknownLoc(),
            vectorType,
            stdVector2Value(values2))
    };

    mlir::Value result = builder.create<mlir::arith::AddIOp>(
        builder.getUnknownLoc(), vecConstant[0], vecConstant[1]);

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

    if(EnableLLVMPass){
        if(runJIT) dumpLLVMIR(*module,true,true);
        else if(DumpLLVMIR) dumpLLVMIR(*module);
    } else {
        module->print(llvm::outs());
    }
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

    if(EnableLLVMPass){
        if(runJIT) dumpLLVMIR(*module,true);
        else if(DumpLLVMIR) dumpLLVMIR(*module);
    } else {
        module->print(llvm::outs());
    }
    return 0;
}