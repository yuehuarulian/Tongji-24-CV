#define _CRT_SECURE_NO_WARNINGS
#include <iostream>  
#include <iomanip>     
#include <graphics.h>  
#include <math.h>  
#include <conio.h>  
#include <time.h>  
#include <stdio.h>
#include "consta.h"
#include"gluttonous_snake.h"
#pragma comment( lib, "MSIMG32.LIB")
using namespace std;
//SDF+alpha blending
/********************************
�������ܣ������ɫ
�βΣ�COLORREF bg������ɫ
	  COLORREF color��ǰ��ɫ
	  double alpha������
����ֵ����
********************************/
COLORREF mix_color(COLORREF bg, COLORREF color, double alpha)//����ɫ ǰ��ɫ alpha=0.5-d ����RGB( , , )
{
	COLORREF result;
	result = RGB(GetRValue(bg) * (1 - alpha) + GetRValue(color) * alpha, GetGValue(bg) * (1 - alpha) + GetGValue(color) * alpha, GetBValue(bg) * (1 - alpha) + GetBValue(color) * alpha);
	return result;
}

/********************************
�������ܣ��������/����ɫ/͸��Բ
�βΣ�int center_x��Բ�ĺ�����
	  int center_y��Բ��������
	  COLORREF color_1����ɫ1������color_2
	  COLORREF color_2����ɫ2��ΪNULLʱ��ʾ�����Ǵ�ɫԲsolidcircle
	  int  radius_1���뾶1
	  int  radius_2���뾶2���뾶���ǲ����ģ������뾶��ȱ�ʾ������circle
	  int SDF_degree������ݵĳ̶ȣ�Ĭ��ֵ��2���������10�����Ͽ��Դﵽ��͸����Ч��
����ֵ����
********************************/
void SDF_circle(int center_x, int center_y, COLORREF color_1, COLORREF color_2, int  radius_1, int radius_2, int SDF_degree)
{
	for (int x = center_x - radius_1 - SDF_WIDTH_CIRCLE; x < center_x + radius_1 + SDF_WIDTH_CIRCLE; x++) {
		for (int y = center_y - radius_1 - SDF_WIDTH_CIRCLE; y < center_y + radius_1 + SDF_WIDTH_CIRCLE; y++) {
			double d;
			d = sqrt((pow(x - center_x, 2) + pow(y - center_y, 2))) - radius_1;//�㵽Բ�ߵľ���
			double alpha = 0.5 - d / SDF_degree;
			if (alpha >= 0 && alpha <= 1) {
				COLORREF bg = getpixel(x, y);
				COLORREF result = mix_color(bg, color_1, alpha);
				putpixel(x, y, result);
			}
		}
	}
	if (radius_1 == radius_2)//ֻ���п����
		;
	else if (color_2 == NULL || radius_2 == 0) { //û�н���
		setfillcolor(color_1);
		setlinecolor(color_1);
		solidcircle(center_x, center_y, radius_1);
	}
	else {  //����ɫ
		for (int i = radius_1 - 1; i > radius_2; i--) {
			COLORREF co;
			co = RGB(
				int(double(radius_1 - i) / (radius_1 - radius_2) * GetRValue(color_2) + double(i - radius_2) / (radius_1 - radius_2) * GetRValue(color_1)),
				int(double(radius_1 - i) / (radius_1 - radius_2) * GetGValue(color_2) + double(i - radius_2) / (radius_1 - radius_2) * GetGValue(color_1)),
				int(double(radius_1 - i) / (radius_1 - radius_2) * GetBValue(color_2) + double(i - radius_2) / (radius_1 - radius_2) * GetBValue(color_1))
			);
			setlinecolor(co);
			setlinestyle(PS_SOLID, 2);
			circle(center_x, center_y, i);
		}
	}
}

/********************************
�������ܣ��������ֱ��
�βΣ�int x1����������
	  int y1�����������
	  int x2���յ������
	  int y2���յ�������
	  COLORREF color����ɫ
	  int thickness���߿�
����ֵ����
********************************/
void SDF_line(int x1, int y1, int x2, int y2, COLORREF color, int thickness, int SDF_degree)
{
	for (int x = min(x1, x2) - SDF_WIDTH_LINE; x < max(x1, x2) + SDF_WIDTH_LINE; x++) {
		for (int y = min(y1, y2) - SDF_WIDTH_LINE; y < max(y1, y2) + SDF_WIDTH_LINE; y++) {
			double d;
			bool b1 = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) >= 0;
			bool b2 = ((x - x2) * (x1 - x2) + (y - y2) * (y1 - y2)) >= 0;
			if (b1 && b2)
				d = fabs(((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))) - thickness / 2;
			else if (!b1)
				d = sqrt(pow(x - x1, 2) + pow(y - y1, 2)) - thickness / 2;
			else
				d = sqrt(pow(x - x2, 2) + pow(y - y2, 2)) - thickness / 2;
			double alpha = 0.5 - d / SDF_degree;
			if (alpha >= 0 && alpha <= 1) {
				COLORREF bg = getpixel(x, y);
				COLORREF result = mix_color(bg, color, alpha);
				putpixel(x, y, result);
			}
		}
	}
	setlinestyle(PS_SOLID | PS_ENDCAP_ROUND, thickness);
	setlinecolor(color);
	line(x1, y1, x2, y2);
}

//x y button��λ��
void button(int x, int y, const TCHAR* text)
{
	setbkmode(TRANSPARENT);
	setfillcolor(MY_PURPLE);
	fillroundrect(x, y, x + BUTTON_W, y + BUTTON_H, 10, 10);
	settextstyle(&Font);
	settextcolor(WHITE);

	int tx = x + (BUTTON_W - textwidth(text)) / 2;
	int ty = y + (BUTTON_H - textheight(text)) / 2;

	outtextxy(tx, ty, text);
}

//͸����ͼ
void transparentimage(IMAGE* dstimg, int x, int y, IMAGE* srcimg)
{
	HDC dstDC = GetImageHDC(dstimg);
	HDC srcDC = GetImageHDC(srcimg);
	int w = srcimg->getwidth();
	int h = srcimg->getheight();

	// �ṹ��ĵ�������Ա��ʾ�����͸���ȣ�0 ��ʾȫ͸����255 ��ʾ��͸����
	BLENDFUNCTION bf = { AC_SRC_OVER, 0, 255, AC_SRC_ALPHA };
	// ʹ�� Windows GDI ����ʵ�ְ�͸��λͼ
	AlphaBlend(dstDC, x, y, w, h, srcDC, 0, 0, w, h, bf);
}