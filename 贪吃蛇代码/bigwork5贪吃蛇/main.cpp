#include<iostream>
#include"gluttonous_snake.h"
#include <Windows.h>
#include <mmsystem.h>

//#include "onnxruntime.h"
//#include <iomanip>		//setw�����������ڿ�
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
    //// ��������ͷ���� 4-3 8-6  16-12  32-24  320-240  640-480  960-720 1280-960
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);//480
    //cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);//640
    //cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // �ر��Զ��ع�
    //double exposureValue = -4; // �ع��ֵ����������ͷ��ͬ�����в�ͬ�ķ�Χ�͵�λ
    //cap.set(cv::CAP_PROP_EXPOSURE, exposureValue);// �����ع��ֵ
    //cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "   " << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
    //OnnxRuntime o;
    //cv::Mat frame;
    //namedWindow("frame");
    //// ����ѭ��
    //while (true) {
    //    //frame = imread("C:/Users/Ruiling/Desktop/click-samples/90.jpg");
    //    //frame = imread("C:/Users/Ruiling/Desktop/down-samples/3.jpg");
    //    //frame = imread("C:/Users/Ruiling/Desktop/right-samples/80.jpg");
    //    //frame = imread("C:/Users/Ruiling/Desktop/left-samples/5.jpg");
    //    // ������ͷ����ͼ��
    //    cap >> frame;
    //    flip(frame, frame, 1);//���ҷ�ת
    //    o.detect(frame);
    //    cv::imshow("frame", frame);
    //    // ���������룬������� 'q' �����˳�ѭ��
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
        cout << "���ִ�ʧ��" << endl;
        return -1;
    }
    Game snake_game;
    snake_game.Top_control();
    return 0;
}