#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include"gluttonous_snake.h"
#include<ctime>

/// <summary>
/// Map Fuction
/// </summary>
//candyͼƬ����
Map::Map()
{
    string s1 = "images/candy";
    string s2 = ".png";
    for (int i = 1; i <= 6; i++) {
        IMAGE* candy_picture = new IMAGE(CANDY_PICTURE_SIZE, CANDY_PICTURE_SIZE);//����imageָ�����
        string s = to_string(i);
        string filename = s1 + s + s2;
        const char* cStyleString = filename.c_str();
        // ���ַ���ת��Ϊ ANSI ����
        int bufferSize = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, NULL, 0);
        wchar_t* wideFilename = new wchar_t[bufferSize];
        MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, wideFilename, bufferSize);
        // ʹ��ת������ַ�������ͼƬ
        loadimage(candy_picture, wideFilename, CANDY_PICTURE_SIZE, CANDY_PICTURE_SIZE);
        delete[] wideFilename; // �ͷ��ڴ�
        candy_pic.push_back(candy_picture);
    }
}
//�ͷ�ͼƬ
Map::~Map()
{
    for (auto i : candy_pic) {
        delete i;
    }
}
//����candy��������
void Map::create_candy(const GAME_MODES& mode, const GAME_STATUS& gamestatus, const Snake& snake)
{
    if (gamestatus == _notstart) {
        candy.clear();
        wall.clear();
    }
    while (candy.size() <= CANDY_COUNT) {
        int x, y, num;
        x = rand() % (GAME_X - 50) + 30;
        y = rand() % (GAME_Y - 50) + 30;
        if (candy.find(make_pair(x, y)) != candy.end())//�ҵ���
            continue;
        num = rand() % (candy_pic.size());
        //����
        candy[make_pair(x, y)] = num;
    }
    if (mode == advanced_version && snake.snakestatus == _die) {
        for (auto i : snake.snake_body) {
            wall.insert(i);
        }
    }
    else if ((mode == premium_edition || mode == AI_snakes) && snake.snakestatus == _die) {
        for (auto i : snake.snake_body) {
            candy[i] = rand() % (candy_pic.size());
        }
    }

}
//����
void Map::draw_candy(IMAGE* newimage)
{
    for (auto element : candy) {
        const std::pair<int, int>& key = element.first;
        int value = element.second;
        transparentimage(newimage, key.first - CANDY_PICTURE_SIZE / 2, key.second - CANDY_PICTURE_SIZE / 2, candy_pic[value]);
    }
}
//��ǽ
void Map::draw_wall()
{
    for (auto i : wall) {
        SDF_circle(i.first, i.second, MY_BLUE, MY_PURPLE, 10, 2, 10);
        SDF_circle(i.first, i.second, MY_BLUE, MY_PURPLE, 10, 2, 10);
    }
}
//�ж�ײǽ�ͳ��ǹ�
bool Map::judge_hitwall(const int &x, const int &y, const GAME_MODES &mode, const int die_dis) const//ײǽ����true
{
    if (mode == advanced_version) {
        for (auto i : wall) {
            double disdance = sqrt((x - i.first) * (x - i.first) + (y - i.second) * (y - i.second));
            if (disdance <= die_dis)
                return true;
        }
    }
    return (abs(x - 10) <= die_dis || abs(x - GAME_X) <= die_dis || abs(y - 0) < die_dis || abs(y - GAME_Y) <= die_dis);//10-GAME_X
}
bool Map::judge_eatcandy(const int& x, const int& y)
{
    for (auto element : candy) {
        const pair<int, int> key = element.first;
        int value = element.second;
        if (sqrt(double(x - key.first) * (x - key.first) + (y - key.second) * (y - key.second)) < CANDY_PICTURE_SIZE)
        {
            candy.erase(key);
            return true;
        }
    }
    return false;
}
