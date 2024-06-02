#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include"gluttonous_snake.h"
#include<ctime>
#include <graphics.h>

#include "onnxruntime.h"
#include <windows.h>
#include <iomanip>		//setw�����������ڿ�
#include<stdlib.h>
#include<time.h>

//�û�������
extern MOUSEMSG m;
extern bool mouse_change;
Game::Game()
{
    gamestatus = GAME_STATUS(_notstart);
    life = 5;
    score = 0;
    // ��ʼ����ͼ����
    initgraph(WINDOW_X, WINDOW_Y, SHOWCONSOLE);
    //��ʼ��menu
    IMAGE* menupicture = new IMAGE(WINDOW_X, WINDOW_Y);//����imageָ�����
    loadimage(menupicture, L"images/bk2.jpg", WINDOW_X, WINDOW_Y);
    SetWorkingImage(menupicture);
    //gamename
    IMAGE* gametitle = new IMAGE(900, 500);
    loadimage(gametitle, L"images/gamename.png", 900, 500);
    transparentimage(menupicture, 150, 0, gametitle);
    //��button
    for (int i = 0; i < MODE_NUM; i++) {
        button(150, 40 + i * 80, GAME_MODE_NAME[i]);
        title_position.push_back(make_pair(150, 40 + i * 80));
    }
    button(150, 40 + 5 * 80, _T("�鿴��ʷ"));
    title_position.push_back(make_pair(150, 40 + 5 * 80));
    //icon
    string s1 = "images/icon/1-";
    string s2 = ".png";
    IMAGE* iconpicture = new IMAGE(600, 300);//����imageָ�����
    wchar_t* wideFilename = new wchar_t[256];
    for (int i = 1; i <= ICONNUM; i++) {
        IMAGE* picture = new IMAGE(WINDOW_X, WINDOW_Y);//����imageָ�����
        SetWorkingImage(picture);
        putimage(0, 0, menupicture);
        string s = to_string(i);
        string filename = s1 + s + s2;
        const char* cStyleString = filename.c_str();
        int bufferSize = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, NULL, 0);// ���ַ���ת��Ϊ ANSI ����
        MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, wideFilename, bufferSize);
        loadimage(iconpicture, wideFilename, 600, 300);// ʹ��ת������ַ�������ͼƬ
        transparentimage(picture, 400, WINDOW_Y - 300, iconpicture);
        transparentimage(picture, 0, WINDOW_Y - 300, iconpicture);
        menuimage.push_back(picture);
    }
    delete gametitle;
    delete iconpicture;
    delete[] wideFilename; // �ͷ��ڴ�

    // ��ʼ��background
    background = new IMAGE(WINDOW_X, WINDOW_Y);
    loadimage(background, L"images/bk.jpg", WINDOW_X, WINDOW_Y);
    SetWorkingImage(background);//����background���滭
    for (int x = 10; x <= GAME_X; x += 20) {//�߿�
        SDF_circle(x, 10, MY_BLUE, MY_PURPLE, 10, 2, 10);
        SDF_circle(x, GAME_Y, MY_BLUE, MY_PURPLE, 10, 2, 10);
    }
    for (int y = 10; y <= GAME_Y; y += 20) {
        SDF_circle(10, y, MY_BLUE, MY_PURPLE, 10, 2, 10);
        SDF_circle(GAME_X, y, MY_BLUE, MY_PURPLE, 10, 2, 10);
    }
}
Game::~Game()
{
    delete background;
    for (auto i : menuimage) {
        delete i;
    }
}

int mousechanege(string s)
{
    mouse(true);
    int x = m.x;
    int y = m.y;
    //    0 up
    //    1 down
    //    2 right
    //    3 left
    //    4 click
    //cout << x << "   " << y << "   ";
    if (s == class_names[0])
        y -= 20;//up
    else if (s == class_names[1])
        y += 20;//down
    else if (s == class_names[3])
        x -= 20;//left
    else if (s == class_names[2])
        x += 20;//right
    else
        ;
    //cout << s << "    "<<x <<"    " << y << endl;
    
    //// ģ������������
    //INPUT input;                //����һ��INPUT���ͽṹ��
    //input.type = INPUT_MOUSE;   //ָ�������¼�����Ϊ����¼�
    //input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP; // ģ�����������º�̧��Ķ���
    ////mi�ǽṹ��INPUT��һ����Ա���ԣ�mi��MOUSEINPUT����
    ////MOUSEINPUT �ṹ��������������¼�������������������˫�����Ҽ��������ƶ��Ȳ�����
    SetCursorPos(x, y);
    //input.mi.dx = 0;            //��������¼�������x����
    //input.mi.dy = 0;            //��������¼�������y����
    //input.mi.mouseData = 0;     //���������ֵĹ�������
    //input.mi.dwExtraInfo = 0;   //�� dwExtraInfo �ֶ�����Ϊ 0����ʾ�������κθ��ӵ��������ݡ�
    //input.mi.time = 0;          //time�ֶ�����������ʱ����ģ�0��ʾʹ��ϵͳ��ʱ���
    //SendInput(1, &input, sizeof(INPUT));

    /*SendInput������ Win32 API ������ģ�������¼��ĺ���������ģ������¼��������¼���Ӳ���¼�
    �ȡ�
    �亯��ԭ�����£�
        UINT SendInput(UINT nInputs, LPINPUT pInputs, int cbSize);
    ���У�nInputs ����ָ�������¼���������pInputs ����ָ�������¼������ָ�룬cbSize ����ָ����
    ���¼�����Ĵ�С����λΪ�ֽڣ���*/
    return 0;
}

//���������Ϸ����
void Game::Top_control()
{
    while (1) {
        // �˵�
        menu();
        // �˳�
        if (mode == Quit_game)
            break;
        //��ʼ��ÿһ����Ϸ
        init_eachgame();
        //��ʼ������ͷ
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Failed to open camera" << std::endl;
            exit(-1);
        }
        //cout << cap.get(cv::CAP_PROP_EXPOSURE) << endl;
        // ��������ͷ����
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);//640 480
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // �ر��Զ��ع�
        double exposureValue = -4; // �ع��ֵ����������ͷ��ͬ�����в�ͬ�ķ�Χ�͵�λ
        cap.set(cv::CAP_PROP_EXPOSURE, exposureValue);// �����ع��ֵ
        //cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "   " << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
        OnnxRuntime o;
        cv::Mat frame;
        namedWindow("frame",1);
        while (gamestatus != _gameover && gamestatus != _notstart) {//��Ϸ�������˳�
            //����
            cap >> frame;
            flip(frame, frame, 1);//���ҷ�ת
            lastaddAIsnakeTime = currentTime;
            string s = o.detect(frame);
            mousechanege(s);
            cv::imshow("frame", frame);
            cv::waitKey(100);
            //ʱ���
            currentTime = chrono::system_clock::now();
            //��Ϸ
            renew_status();//��ȡ�������ͣ�˳�
            if (mode == AI_snakes) {
                renew_AISnake();
            }
            renew_interface();//�ػ�ҳ��
        }
        //�ر�����ͷ
        cap.release();
        cv::destroyAllWindows();
        //�����ļ�
        renew_file();
    }
    closegraph();		// �˳�ͼ�ν���
}
//�˵�״̬ά��
void Game::menu()
{
    SetWorkingImage(NULL);
    gamestatus = _notstart;
    //��ȡ��Ϣ
    int count = 0;

    while (gamestatus == _notstart && mode != Quit_game) {
        putimage(0, 0, menuimage[count]);
        count = (count + 1) % ICONNUM;
        Sleep(20);
        mouse(); 
        switch (m.uMsg)
        {
            case WM_LBUTTONDOWN:
                // �������
                if ((m.x - title_position[0].first) >= 0 && (m.x - title_position[0].first) <= BUTTON_W
                    && (m.y - title_position[0].second) >= 0 && (m.y - title_position[0].second) <= BUTTON_H)
                {
                    gamestatus = _play;
                    mode = basic_version;
                }
                else if ((m.x - title_position[1].first) >= 0 && (m.x - title_position[1].first) <= BUTTON_W
                    && (m.y - title_position[1].second) >= 0 && (m.y - title_position[1].second) <= BUTTON_H)
                {
                    gamestatus = _play;
                    mode = advanced_version;
                }
                else if ((m.x - title_position[2].first) >= 0 && (m.x - title_position[2].first) <= BUTTON_W
                    && (m.y - title_position[2].second) >= 0 && (m.y - title_position[2].second) <= BUTTON_H)
                {
                    gamestatus = _play;
                    mode = premium_edition;
                }
                else if ((m.x - title_position[3].first) >= 0 && (m.x - title_position[3].first) <= BUTTON_W
                    && (m.y - title_position[3].second) >= 0 && (m.y - title_position[3].second) <= BUTTON_H)
                {
                    gamestatus = _play;
                    mode = AI_snakes;
                }
                else if ((m.x - title_position[4].first) >= 0 && (m.x - title_position[4].first) <= BUTTON_W
                    && (m.y - title_position[4].second) >= 0 && (m.y - title_position[4].second) <= BUTTON_H)
                {
                    mode = Quit_game;
                }
                else if ((m.x - title_position[5].first) >= 0 && (m.x - title_position[5].first) <= BUTTON_W
                    && (m.y - title_position[5].second) >= 0 && (m.y - title_position[5].second) <= BUTTON_H)
                {
                    history_search();
                }
                break;
            case WM_RBUTTONDOWN:
                // �Ҽ�����
                break;
            case WM_MOUSEMOVE:
                // ����ƶ�
                break;
            case WM_LBUTTONDBLCLK:
                // ���˫��
                break;
        }
        m.x = 0; m.y = 0;
    }
}
//��ʷ��ѯ������
void Game::history_search()
{
    // ��ȡ��ѯ����ʷ���
    TCHAR num[256];
    InputBox(num, 50, _T("�������ѯ�ı�ţ���һ��Ϊ1�����ε�����\n������ʷ������0"), _T("��ʷ��ѯ"));
    wstring numStr(num);
    try {
        int intValue = std::stoi(numStr);
        TCHAR infoText[256];
        if (intValue == 0) {
            file.reset_highest_score();
            file.reset_history();
            _stprintf(infoText, _T("������"));
        }
        else if (file.read_history(intValue) == false)
            _stprintf(infoText, _T("û�и���ʷ"));
        else
            _stprintf(infoText, _T("��%d����ʷ����:%s"), intValue, file.history);
        MessageBox(GetHWnd(), infoText, _T("��ʷ"), MB_OK | MB_ICONINFORMATION);
    }
    catch (invalid_argument) {
        cerr << "��Ч�����룬�޷�ת��Ϊ������" << endl;
        exit(-1);
    }
    catch (out_of_range) {
        cerr << "����������Χ��" << endl;
        exit(-1);
    }
}
//��ʼ����Ϸ����
void Game::init_eachgame()
{
    //��ʼ��
    srand(unsigned int(time(NULL)));
    score = 0;
    life = 500;
    kill_num = 0;
    // �����û���������50���ַ�
    do
        InputBox(username, 50, _T("�������û���\n����Ϊ��"), _T("�û�������"));
    while (username[0] == '\0');
    //time
    gamestartTime = chrono::system_clock::now();
    lastaddAIsnakeTime = gamestartTime;
    //����ͼƬ��ʾ
    SetWorkingImage(NULL);
    putimage(0, 0, background);
    //��
    snake.init();
    AIsnake.clear();
    //�ǹ�
    game_map.create_candy(mode, _notstart, snake);//�����ǹ�
}
//1.������Ϸ״̬
void Game::renew_status()
{
    //������
    snake.Top_control(game_map, AIsnake, mode, currentTime, speed_count, score);
    //������û��die,����life��gameover
    handle_die_gameover();
    //�����ǹ�
    game_map.create_candy(mode, gamestatus, snake);

    //���ֵ����status
    renew_pause_nostart();
    //��ͣ���˳�
    if (gamestatus == _pause)
        handle_pause();
    else if (gamestatus == _notstart)//�˳�
        return;
}
void Game::renew_pause_nostart()
{
    if (!mouse_change)
        return;
    switch (m.uMsg)
    {
        case WM_LBUTTONDOWN:
            // ���������ͣ
            // �ٰ���ȡ����ͣ
            if (gamestatus != _pause)
                gamestatus = _pause;
            else
                gamestatus = _play;
            break;
        case WM_RBUTTONDOWN:
            // �Ҽ������˳�
            gamestatus = _notstart;
            //snake.set_notbirth();
            break;
        case WM_MOUSEMOVE:
            // ����ƶ�
            break;
        case WM_LBUTTONDBLCLK:
            // ���˫��
            break;
    }
}
void Game::handle_pause() 
{
    setbkmode(TRANSPARENT);//���ñ���͸��
    settextstyle(&Font);
    settextcolor(MY_RED);
    TCHAR infoText[256];
    _stprintf(infoText, _T("���ٴΰ������ȡ����ͣ"));
    outtextxy(GAME_X / 2 - sizeof(infoText) / 8, GAME_Y / 2, infoText);
    while (gamestatus == _pause) {
        mouse();
        if (mouse_change) {
            //���ֵ����status
            renew_pause_nostart();
        }
    }
}
void Game::handle_die_gameover()
{
    if (snake.get_snakestatus() != _die)
        return;
    else {
        life--;
        if (life == 0 || mode == AI_snakes) {
            gamestatus = _gameover;
            //snake.set_notbirth();
            return;
        }
    }
}

//����AI��
void Game::renew_AISnake()
{
    //�Ƿ����
    if (AIsnake.size() <= SNAKE_NUM_MAX && increase_AISnake()) {
        AISnake aisnake;
        AIsnake.push_back(aisnake);
    }
    //�Ƿ�����
    for (auto it = AIsnake.begin(); it != AIsnake.end(); ) {
        auto& aisnake = *it;
        aisnake.AI_Top_control(game_map,snake,mode,currentTime,life);
        //�����ǹ�
        if (aisnake.get_snakestatus() == _die) {
            game_map.create_candy(mode, gamestatus, aisnake);
            it = AIsnake.erase(it);  // �Ƴ��ߣ���������ָ����һ��Ԫ��
            kill_num++;
            continue;
        }
        ++it;  // ����������
    }

}
bool Game::increase_AISnake()
{
    auto duration = (chrono::duration_cast<std::chrono::seconds>(currentTime - lastaddAIsnakeTime)).count();
    if (duration > ADD_AISNAKE_TIME_GAP) {
        lastaddAIsnakeTime = currentTime;
        return true;
    }
    return false;
}

//2.����ҳ��
void Game::renew_interface()
{
    IMAGE* newimage = new IMAGE(WINDOW_X, WINDOW_Y);//����imageָ�����
    SetWorkingImage(newimage);//������Ⱦ
    putimage(0, 0, background);
    //�ǹ�
    game_map.draw_candy(newimage);//�����ǹ�
    //ǽ
    if (mode == advanced_version)
        game_map.draw_wall();
    //�� AI��
    snake.Drawsnake(newimage);
    if (mode == AI_snakes) {
        for (const auto &aisnake : AIsnake) {
            aisnake.Drawsnake(newimage);
        }
    }
    //����������
    show_gameinf();
    //��ͼƬ����ʾ��
    SetWorkingImage(NULL);
    putimage(0, 0, newimage);
    delete newimage;
}
void Game::show_gameinf()
{
    setbkmode(TRANSPARENT);//���ñ���͸��
    settextstyle(&Font);
    settextcolor(MY_PURPLE);

    TCHAR infoText[256];
    auto duration = (chrono::duration_cast<std::chrono::milliseconds>(currentTime - gamestartTime)).count();
    int gametime = int(duration) / 1000;//������Ϸʱ��
    switch (mode) {
        case basic_version:
            _stprintf(infoText, _T("[������]"));
            break;
        case advanced_version:
            _stprintf(infoText, _T("[���װ�]"));
            break;
        case premium_edition:
            _stprintf(infoText, _T("[�߼���]"));
            break;
        case AI_snakes:
            _stprintf(infoText, _T("[AIɱ�߰�]"));
            break;
        default:
            break;
    }
    const int ygap = 40;
    outtextxy(GAME_X + 20, ygap * 1, infoText);
    _stprintf(infoText, _T("��ʷ��߷֣�%lld"), file.highest_score[mode]);
    outtextxy(GAME_X + 20, ygap * 2, infoText);
    _stprintf(infoText, _T("��Ϸ�÷֣�%lld"), score);
    outtextxy(GAME_X + 20, ygap * 3, infoText);
    _stprintf(infoText, _T("�ߵĳ��ȣ�%zd"), snake.get_snakesize());
    outtextxy(GAME_X + 20, ygap * 4, infoText);
    if (mode != AI_snakes)
        _stprintf(infoText, _T("����������%d"), life);
    else
        _stprintf(infoText, _T("��ɱ������%d"), kill_num);
    outtextxy(GAME_X + 20, ygap * 5, infoText);
    _stprintf(infoText, _T("��Ϸʱ�䣺%02d:%02d"), gametime / 60, gametime % 60);
    outtextxy(GAME_X + 20, ygap * 6, infoText);

    settextcolor(MY_RED);
    _stprintf(infoText, _T("��ʾ��"));
    outtextxy(GAME_X + 20, ygap * 8, infoText);
    _stprintf(infoText, _T("��������ͣ���ٴ����ȡ����ͣ"));
    outtextxy(GAME_X + 20, ygap * 9, infoText);
    _stprintf(infoText, _T("����Ҽ��˳�������Ϸ"));
    outtextxy(GAME_X + 20, ygap * 10, infoText);
    _stprintf(infoText, _T("�ո���٣�����"));
    outtextxy(GAME_X + 20, ygap * 11, infoText);
    _stprintf(infoText, _T("���ǻ��ȥ���ȣ�����ʹ��Ŷ~~~"));
    outtextxy(GAME_X + 20, ygap * 12, infoText);
    _stprintf(infoText, _T("���������޵�ʱ�䣬�޷�����ɱ����"));
    outtextxy(GAME_X + 20, ygap * 13, infoText);
    if (snake.get_snakesize() <= 5) {
        _stprintf(infoText, _T("�ߵĳ���̫�������ܲ�����������"));
        outtextxy(GAME_X + 20, ygap * 14, infoText);
    }

}
void Game::renew_file()
{
    file.update_highest_score(mode, score);//������߷�
    file.save_history(mode, username, score);
}

