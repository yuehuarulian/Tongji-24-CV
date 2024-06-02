#ifndef CONSTA_H
#define CONSTA_H
/*
1.�����ͣ
2.�ո����
3.�Ҽ��˳�������Ϸ
*/
//����
const double M_PI = 3.1415926;
//����
const LOGFONT Font{ 20, 10, 0, 0, FW_DONTCARE, false, false, false, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, PROOF_QUALITY, DEFAULT_PITCH, L"����" };

//game
    //��Ļ
const int WINDOW_X = 1000;
const int WINDOW_Y = 660;
const int GAME_X = WINDOW_Y - 10;
const int GAME_Y = WINDOW_Y - 10;
//mode
const int MODE_NUM = 6;
//menu
const int ICONNUM = 24;
const int BUTTON_W = 200, BUTTON_H = 50;
const TCHAR GAME_MODE_NAME[MODE_NUM][50] = { _T("Basic Version"),_T("Advanced Version"),_T("Premium Edition"),_T("�˻�ɱ�߰�"),_T("Quit Game"),_T("History") };
//�ٶ�
const int Speed_Threshold = 30;//speed_count�����ͼ�һ��
//snake
const int SNAKE_BODY_PIC_SIZE = 30;//����ͼƬ�Ĵ�С
const int SNAKE_PRO_PIC_SIZE = 300;//����ȦͼƬ�Ĵ�С
const int SNAKE_INIT_LEN = 5;//��ʼ�ߵĳ���
const int SNAKE_RADIUS = 12;//һ������İ뾶
const int SNAKE_DIS = 14;//��������֮��ľ���
const int PROTECT_TIME = 5;

const int SNAKE_NUM_MAX = 6;//AI�ߵ��������
const int ADD_AISNAKE_TIME_GAP = 6;//��AI�ߵ�ʱ����
//candy
const int CANDY_COUNT = 40;//��Ļ��candy������
const int CANDY_PICTURE_SIZE = 30;//candy_pic��size

//file
//const char *HIGHEST_SCORE_FILENAME = (const char *)"userdata/highest_score.txt";
const char HIGHEST_SCORE_FILENAME[] = "userdata/highest_score.txt";
const char GAME_HISTORY_SAVE_FILENAME[] = "userdata/game_history.txt";//�汾�����Ű� �û�����root �÷֣�100

//drawtool
#define SDF_WIDTH_CIRCLE 30//����ݴ������ؿ�� circle
#define SDF_WIDTH_LINE 10//����ݴ������ؿ�� line

//����color
#define MY_PURPLE RGB(150,142,185)
#define MY_WHITE RGB(248, 241, 248)
#define MY_RED RGB(245,150,150)
#define MY_LIGHT_RED RGB(255, 186, 201)
#define MY_PINK RGB(249,218,228)
#define MY_WHITE_2 RGB(249, 236, 243)
#define MY_BLUE RGB(239,239,246)
#endif // !CONST

