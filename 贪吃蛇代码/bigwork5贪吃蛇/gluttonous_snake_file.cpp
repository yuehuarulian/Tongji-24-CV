#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include"gluttonous_snake.h"

/// <summary>
/// File Function
/// </summary>
File::File()
{
    read_highest_score();//��ȡ��highestScore
}
//���ַ������浽�ļ�
void File::save_to_file(const char* filename, long long int* array, const size_t& size) {
    ofstream file(filename, ios::binary | ios::out);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(array), sizeof(long long int) * size);
        file.close();
    }
    else {
        cerr << "�޷����ļ���" << GAME_HISTORY_SAVE_FILENAME << endl;
        exit(-1);
    }
    /*FILE* file = fopen(filename, "wb");
    if (file == NULL)
        exit(-1);
    fwrite(array, sizeof(T), size, file);
    fclose(file);*/
}
void File::save_to_file(const char* filename, TCHAR* array, const size_t& size) {
    ofstream file(filename, ios::binary | ios::app);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(array), sizeof(TCHAR) * size);
        file.close();
    }
    else {
        cerr << "�޷����ļ���" << GAME_HISTORY_SAVE_FILENAME << endl;
        exit(-1);
    }
}
//���ļ���ȡ�ַ���
streamsize File::read_from_file(const char* filename, long long int* array, const size_t& size) {
    ifstream file(filename, ios::binary | ios::in);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(array), sizeof(long long int) * size);
        streamsize bytesRead = file.gcount();
        file.close();
        return bytesRead;
    }
    else {
        cerr << "�޷����ļ���" << GAME_HISTORY_SAVE_FILENAME << endl;
        exit(-1);
    }
    return NULL;
    //FILE* file = fopen(filename, "rb");
    //if (file == NULL)
    //    exit(-1);
    //fread(array, sizeof(T), size, file);
    //fclose(file);
}
streamsize File::read_from_file(const char* filename, TCHAR* array, const int& history_num) {
    ifstream file(filename, ios::in);
    if (file.is_open()) {
        for (int i = 0; i < history_num; i++) {
            file.getline(reinterpret_cast<char*>(array), 256);
            file.get();///?
            if (file.eof())
                return NULL;
        }
        streamsize bytesRead = file.gcount();
        file.close();
        return bytesRead;
    }
    else {
        cerr << "�޷����ļ���" << GAME_HISTORY_SAVE_FILENAME << endl;
        exit(-1);
    }
    return NULL;
}
//����ļ�
void File::clear_file(const char* filename) {
    // �����ģʽ���ļ����������ļ�����
    ofstream file(filename, ios::out | ios::trunc);
    if (file.is_open()) {
        file.close();
    }
    else {
        cerr << "�޷����ļ���" << filename << endl;
        exit(-1);
    }
}
//������߳ɼ�
void File::update_highest_score(const GAME_MODES& mode, const long long& score) {
    if (highest_score[mode] >= score)
        return;
    highest_score[mode] = score;
    save_to_file(HIGHEST_SCORE_FILENAME, highest_score, MODE_NUM - 2);
}
void File::read_highest_score() {
    streamsize readnum = read_from_file(HIGHEST_SCORE_FILENAME, highest_score, MODE_NUM - 2);
    if (readnum != (MODE_NUM - 2) * sizeof(long long int))//����ļ�����ʲô��û�оͳ�ʼ��һ��
        reset_highest_score();
}
void File::reset_highest_score()
{
    clear_file(HIGHEST_SCORE_FILENAME);
    for (int i = 0; i < MODE_NUM - 2; i++)
        highest_score[i] = { 0 };
}
//������ʷ
void File::save_history(const GAME_MODES& mode, const TCHAR* username, const long long& score)
{
    WCHAR infoText[256];
    _stprintf(infoText, _T("�汾:%s �û���:%s �÷�:%lld\n"), GAME_MODE_NAME[mode], username, score);
    save_to_file(GAME_HISTORY_SAVE_FILENAME, infoText, wcslen(infoText) * 1);
    //�汾�����Ű� �û�����root �÷֣�100
}
bool File::read_history(int history_num)
{
    if (read_from_file(GAME_HISTORY_SAVE_FILENAME, history, history_num) == NULL)
        return false;
    return true;

    //�汾�����Ű� �û�����root �÷֣�100
}
void File::reset_history()
{
    clear_file(GAME_HISTORY_SAVE_FILENAME);
    //// �����ģʽ���ļ����������ļ�����
    //ofstream file(GAME_HISTORY_SAVE_FILENAME, ios::out | ios::trunc);
    //if (file.is_open()) {
    //    file.close();
    //}
    //else {
    //    cerr << "�޷����ļ���" << GAME_HISTORY_SAVE_FILENAME << endl;
    //    exit(-1);
    //}
}