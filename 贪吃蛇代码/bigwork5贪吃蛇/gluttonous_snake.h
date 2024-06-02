#ifndef GAME_H
#define GAME_H
#include <stdio.h>
#include <easyx.h>
#include <conio.h>
#include <time.h>
#include <vector>
#include <map>
#include <set>
#include <deque>
#include <math.h>
#include <string>
#include <windows.h>
#include <chrono>
#include <fstream>
#include "consta.h"
using namespace std;
//��Ϸģʽ
enum GAME_MODES {
	basic_version,//������
	advanced_version,//���װ�
	premium_edition,//�߼���
	AI_snakes,//AIɱ�߰�
	Quit_game,//�˳���Ϸ����
	History//��ʷ��¼
};
//��Ϸ״̬
enum GAME_STATUS {
	_notstart,
	_play,
	_pause,
	_gameover
};
//�ߵ�״̬
enum SNAKE_STATUS {
	_notbirth,
	_move,
	_die,
};
//Snakespeed
enum SNAKE_SPEED {
	slow_speed = 90,
	middle_speed = 50,
	fast_speed = 30
};
//AIsnake��ģʽ
enum AISNAKE_MODES {
	kill_player,
	eat_candy
};
class Snake;
class AISnake;
class Map;
class Game;
class File;

class Snake {
protected:
	IMAGE* snake_body_icon, * snake_protect;//ͼƬ
	chrono::time_point<std::chrono::system_clock>lastmovetime, birthtime;//�ϴ�move��ʱ�� ����ʱ��
	bool isprotect;//�Ƿ񱣻�ʱ��
	//pair<int, int> head_toward;//��ͷ����
	SNAKE_SPEED speed;//�ߵ��ٶ�
	SNAKE_STATUS snakestatus;//s�ߵ�״̬
	deque<pair<int, int>>snake_body;//endΪ��ͷ��0Ϊ��β
	//pair<int, int> snake_head;//�߿�ʼ�ͽ���
	int add_lenth;//�������ӳ���
	//�߳���
	void snake_birth(const int& birth_lenth);
	//���·���
	void renew_direction();
	//�����ٶ�
	void renew_speed(int& speed_count);
	//���³���
	void renew_addlenth(Map& game_map, long long& score, int& speed_count);
	//��������
	void renew_body();
	//�ж��Ƿ�Ҫ�ƶ��Ƿ��ڱ���ʱ��
	bool judgemove_pro(const chrono::time_point<std::chrono::system_clock>& currenttime);//�Ƿ��ƶ�һ��
public:
	//friend AISnake;
	friend Map;
	Snake();
	~Snake();
	pair<int, int> head_toward;//��ͷ����
	pair<int, int> snake_head;//�߿�ʼ
	//ÿ����Ϸ��ʼʱ��ʼ��
	void init();
	//����߳�
	int get_snakesize();
	//�����״̬
	SNAKE_STATUS get_snakestatus();
	//
	void Top_control(Map& game_map, const vector <AISnake>& AIsnake, const GAME_MODES &mode, const chrono::time_point<std::chrono::system_clock>& currenttime,int& speed_count, long long& score);//�ı�snakebody�����ߵ�״̬
	void Drawsnake(IMAGE* newimage) const;//����
	bool judge_hitsnake(const int &x, const int &y, const int die_dis = 20) const;//�Ƿ�ײ���˱���
};
class AISnake : public Snake {
private:
	pair<int, int> target;//Ŀ��
	AISNAKE_MODES aisnake_mode;
	void makeDecision(const Snake& snake, const Map& game_map);//�����������滮����
public:
	AISnake();
	explicit AISnake(const AISnake& AIs);
	AISnake& operator=(const AISnake& AIs);
	~AISnake() {};
	void AI_Top_control(Map& game_map, const Snake& snake, const GAME_MODES& mode, const chrono::time_point<std::chrono::system_clock>& currenttime, int& life);//�ı���
};
class Map {
public:
	Map();
	~Map();
	void create_candy(const GAME_MODES &mode, const GAME_STATUS &gamestatus, const Snake& snake);
	void draw_candy(IMAGE* newimage);
	void draw_wall();
	bool judge_hitwall(const int &x, const int &y, const GAME_MODES &mode, const int die_dis = 20) const;
	bool judge_eatcandy(const int &x, const int &y);
private:
	map<pair<int, int>, int> candy;
	set<pair<int, int>> wall;
	vector<IMAGE*> candy_pic;
	friend class AISnake;
	friend class Snake;
};
class File {
public:
	File();
	void update_highest_score(const GAME_MODES &mode, const long long &score);
	void read_highest_score();
	void reset_highest_score();

	void save_history(const GAME_MODES &mode, const TCHAR* username, const long long &score);
	bool read_history(int history_num);
	void reset_history();
	long long int highest_score[MODE_NUM - 2];
	WCHAR history[256];
private:
	void save_to_file(const char* filename, long long int* array, const size_t &size);
	void save_to_file(const char* filename, TCHAR* array, const size_t &size);
	streamsize read_from_file(const char* filename, long long int* array, const size_t &size);
	streamsize read_from_file(const char* filename, TCHAR* array, const int &history_num);
	void clear_file(const char* filename);
};

class Game {
public:
	Game();//��ʼ��������Ϸ
	~Game();
	void Top_control();//������ƺ���
private:
	//ʱ��
	chrono::time_point<std::chrono::system_clock>currentTime;//��¼���ڵ�ʱ��
	chrono::time_point<std::chrono::system_clock>gamestartTime;//��¼һ����Ϸ��ʼ��ʱ��
	chrono::time_point<std::chrono::system_clock>lastaddAIsnakeTime;
	int speed_count;
	//menu�˵���5��ģʽ
	vector<pair<int, int>> title_position;
	//����ͼƬ �˵�ͼƬ
	IMAGE* background;
	vector<IMAGE*>menuimage;
	//״̬��¼ģʽ��¼
	GAME_STATUS gamestatus;
	GAME_MODES mode;
	//��
	Snake snake;
	vector <AISnake> AIsnake;
	//��ͼ
	Map game_map;
	//�ļ�
	File file;
	//һЩ��Ϸ�Ĳ���
	int life;//����ֵ
	long long int score;//���ַ���
	int kill_num;//��ɱ����
	TCHAR username[256] = _T("");//�û���
	//����
	void menu();//�˵�
	void history_search();//��ʷ��¼

	void init_eachgame();//��ʼ��һ����Ϸ������snake��map�ĳ�ʼ��

	void renew_status();//����status
	//bool mouse();//��ȡ���ṹ��
	//char keyboard();//��ȡ����
	//void renew_direction();//��ȡ����
	//void renew_speed();
	void renew_pause_nostart();
	void handle_pause();
	void handle_die_gameover();
	//void handle_move();
	//void Judge_status();//�ж�״̬������status
	void renew_AISnake();
	bool increase_AISnake();
	
	void renew_interface();//���»��ߺ��ǹ�
	void show_gameinf();

	void renew_file();
};

bool mouse(bool forceget = false);
//draw tool
COLORREF mix_color(COLORREF bg, COLORREF color, double alpha);
void SDF_circle(int center_x, int center_y, COLORREF color_1, COLORREF color_2, int  radius_1, int radius_2, int SDF_degree);
void SDF_line(int x1, int y1, int x2, int y2, COLORREF color, int thickness, int SDF_degree);
void button(int x, int y, const TCHAR* text);
void transparentimage(IMAGE* dstimg, int x, int y, IMAGE* srcimg);
#endif