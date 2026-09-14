#include <onnxruntime_cxx_api.h>
#include <iostream>

int main(int argc,char** argv){
    if(argc <2){std::cout<<"用法: ./ort_probe 模型路径\n"; return 1;}
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "probe");
    Ort::Session session(env, argv[1], Ort::SessionOptions(nullptr));
    Ort::AllocatorWithDefaultOptions alloc;

    for (int i = 0;i < session.GetInputCount(); i++){
        auto name = session.GetInputNameAllocated(i, alloc);
        auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "输入["<<i<<"]"<<name.get() << "形状:";
        for(auto s : shape) std::cout <<s<<" ";
        std::cout <<"\n";
    }
    for(int i= 0; i < session.GetOutputCount(); i++){
        auto name = session.GetOutputNameAllocated(i, alloc);
        auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "输出[" << i << "] " << name.get() << " 形状: ";
        for (auto s : shape) std::cout << s << " ";
        std::cout << "\n";
    }
    std::cout << "模型加载成功\n";
    return 0;
}