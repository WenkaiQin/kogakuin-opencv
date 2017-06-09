#include <iostream>
#include <array>

using namespace std;

int mainasdf() {
	// 抽出したい色の指定
	int g_min[] = {30,70,30};
	int g_max[] = {70,255,255};
	// int k[5][5];
	array<int,5> k;

	k.fill(1);

	cout << "k contains:";
	for(int& x:k) {cout << ' ' << x;}
	cout << endl;
}