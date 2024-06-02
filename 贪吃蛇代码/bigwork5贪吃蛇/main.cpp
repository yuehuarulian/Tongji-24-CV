#include<iostream>
#include"gluttonous_snake.h"
#include <Windows.h>
#include <mmsystem.h>

//#include "onnxruntime.h"
//#include <iomanip>		//setw（）函数所在库
//#include<stdlib.h>
//#include<time.h>
#pragma comment (lib, "winmm.lib")
using namespace std;
//using namespace Ort;
int main()
{
    //cv::VideoCapture cap(0);
    //if (!cap.isOpened()) {
    //    std::cerr << "Error: Failed to open camera" << std::endl;
    //    return -1;
    //}
    //cout << cap.get(cv::CAP_PROP_EXPOSURE) << endl;
    //// 设置摄像头属性 4-3 8-6  16-12  32-24  320-240  640-480  960-720 1280-960
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);//480
    //cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);//640
    //cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // 关闭自动曝光
    //double exposureValue = -4; // 曝光度值，根据摄像头不同可能有不同的范围和单位
    //cap.set(cv::CAP_PROP_EXPOSURE, exposureValue);// 设置曝光度值
    //cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "   " << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
    //OnnxRuntime o;
    //cv::Mat frame;
    //namedWindow("frame");
    //// 推理循环
    //while (true) {
    //    //frame = imread("C:/Users/Ruiling/Desktop/click-samples/90.jpg");
    //    //frame = imread("C:/Users/Ruiling/Desktop/down-samples/3.jpg");
    //    //frame = imread("C:/Users/Ruiling/Desktop/right-samples/80.jpg");
    //    //frame = imread("C:/Users/Ruiling/Desktop/left-samples/5.jpg");
    //    // 从摄像头捕获图像
    //    cap >> frame;
    //    flip(frame, frame, 1);//左右反转
    //    o.detect(frame);
    //    cv::imshow("frame", frame);
    //    // 检测键盘输入，如果按下 'q' 键则退出循环
    //    char key = cv::waitKey(100);
    //    if (key == ' ') {
    //        break;
    //    }
    //}
    //cap.release();
    //cv::destroyAllWindows();
    //return 0;

    if (!PlaySound(L"music/backmusic.wav", NULL, SND_FILENAME | SND_ASYNC))
    {
        cout << "音乐打开失败" << endl;
        return -1;
    }
    Game snake_game;
    snake_game.Top_control();
    return 0;
}