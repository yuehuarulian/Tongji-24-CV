#ifndef ONNXRNTIME
#define ONNXRNTIME

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <onnxruntime_cxx_api.h>

#include <windows.h>
#include <iomanip>		//setw（）函数所在库
#include<stdlib.h>
#include<time.h>
using namespace std;
using namespace cv;
using namespace Ort;


//0 up
//1 down
//2 right
//3 left
//4 click
const std::vector<std::string> class_names = {
    "Up", "Down", "Right", "Left", "Click"};

class OnnxRuntime
{
public:
    OnnxRuntime(wstring model_path = L"onnx/2.onnx");
    string detect(Mat& frame);

private:
    int inpWidth; // input图像的大小 416x416
    int inpHeight;
    cv::Mat letter_box(cv::Mat& src, int h, int w);
    float conf = 0.5;
    vector<float> input_image_;
    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "onnx_model");
    Ort::Session* ort_session = nullptr;
    SessionOptions sessionOptions = SessionOptions();
    vector<char*> input_names;
    vector<char*> output_names;
    vector<vector<int64_t>> input_node_dims;  // >=1 outputs
    vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

OnnxRuntime::OnnxRuntime(wstring model_path)
{
    const ORTCHAR_T* modelPath = model_path.c_str();
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    ort_session = new Session(env, modelPath, sessionOptions); // const ORTCHAR_T*
    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++)
    {
        input_names.push_back(ort_session->GetInputName(i, allocator));
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        output_names.push_back(ort_session->GetOutputName(i, allocator));
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }
    //[1,224,224,3]
    this->inpHeight = input_node_dims[0][1];
    this->inpWidth = input_node_dims[0][2];//NULL,224,224,3
}

cv::Mat OnnxRuntime::letter_box(cv::Mat& src, int h, int w)
{
    int in_w = src.cols;
    int in_h = src.rows;
    int tar_w = w;
    int tar_h = h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;
    cv::Mat resize_img;
    resize(src, resize_img, cv::Size(inside_w, inside_h));
    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return resize_img;
}

string OnnxRuntime::detect(Mat& frame)
{
    cv::Mat dstimg;
    // 对输入图像进行大小调整以适应模型输入大小，并返回调整后的图像和填充值
    cv::Mat boxed = letter_box(frame, inpWidth, inpHeight);
    //cv::imshow("boxed", boxed);
    cv::cvtColor(boxed, dstimg, cv::COLOR_BGR2RGB);
    this->input_image_.resize(inpWidth * inpHeight * dstimg.channels());
    // 将图像数据归一化并填充到输入张量中
    for (int h = 0; h < inpHeight; h++)
        for (int w = 0; w < inpWidth; w++)
            for (int c = 0; c < 3; c++)
                //NHWC 格式
                input_image_[h * inpWidth * 3 + w * 3 + c] = float(dstimg.at<cv::Vec3b>(h, w)[c]) / 127.5 - 1.0;//标准化
                //input_image_[h * inpWidth * 3 + w * 3 + c] = float(dstimg.at<cv::Vec3b>(h, w)[c]) / 255.0;//归一化

    array<int64_t, 4> input_shape_{ 1, this->inpHeight, this->inpWidth, 3};

    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

    // 开始推理
    vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());

    //// 获取模型输出节点的数量
    //size_t num_output_nodes = ort_outputs.size();
    //cout << num_output_nodes << endl;
    //// 遍历模型输出节点
    //for (size_t i = 0; i < num_output_nodes; ++i)
    //{
    //    Ort::Value& output_tensor = ort_outputs[i];
    //    // 获取输出节点的名称
    //    const char* output_name = output_names[i];
    //    // 获取输出节点的形状
    //    std::vector<int64_t> output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    //    // 获取输出节点的数据类型
    //    Ort::TensorTypeAndShapeInfo output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    //    ONNXTensorElementDataType output_data_type = output_tensor_info.GetElementType();
    //    // 打印输出节点的信息
    //    std::cout << "Output Node: " << output_name << std::endl;
    //    std::cout << "Output Shape: ";
    //    for (int dim : output_shape)
    //    {
    //        std::cout << dim << " ";
    //    }
    //    std::cout << std::endl;
    //    std::cout << "Output Data Type: " << output_data_type << std::endl;
    //    // 获取输出节点的数据
    //    // 注意：根据输出节点的数据类型和形状，以适当的方式访问输出数据
    //    // 这里的示例假设输出节点是一个 float32 类型的张量
    //    const float* output_data = output_tensor.GetTensorMutableData<float>();
    //    // 打印输出数据
    //    std::cout << "Output Data: ";
    //    for (size_t j = 0; j < output_shape.size(); ++j)
    //    {
    //        std::cout << output_data[j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    const float* result = ort_outputs[0].GetTensorMutableData<float>();
    for (int i = 0; i < 5; i++) {
        std::ostringstream oss;
        oss << class_names[i] << std::fixed << std::setprecision(4) << result[i];
        std::string s = oss.str();
        cv::putText(frame, s, cv::Point(20, 20 * i + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 225), 1.5);
    }
    for (int i = 0; i < 5; i++) {
        if (result[i] > conf)
        {
            cv::putText(frame, class_names[i], cv::Point(20, 250), cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(0,0,225), 4);
            return class_names[i];
        }
    }
    return "";
}


#endif