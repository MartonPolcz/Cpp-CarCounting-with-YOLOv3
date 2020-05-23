#ifndef MY_BOUNDINGBOX
#define MY_BOUNDINGBOX

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// A Bounding Boxokat le�r� oszt�ly
class BoundingBox {
public:
	
	Rect box; // a bounding box kerete

	int classId; // a bounding box oszt�lya

	float confidence; // a bounding box oszt�ly�nak val�sz�n�s�ge

	vector<Point> centerPositions; // megadja a bounding box poz�ci�it az el�z� frame-eken

	double CurrentDiagonalSize; // megadja a bounding box �tm�r�j�nek hossz�t pixelekben

	bool CurrentMatchFoundOrNewBox; // megadja, hogy a bounding box egy �jonnan detekt�lt aut�hoz tartozik, vagy az m�r az el�z� frame-n is szerepelt

	bool StillBeingTracked; // megadja, hogy a bounding box m�g l�tezik-e (az elm�lt 5 frame-n bel�l detekt�ltuk-e valaha)

	Point predictedNextPosition; // megadja, hogy a k�vetkez� frame-n hova v�rhat� az adott aut�hoz tartoz� bounding box

	int NumOfConsecutiveFramesWithoutAMatch; // megadja, hogy ha elt�nt a bounding box, akkor h�ny frame telt el az�ta hogy nem tal�ljuk p�rj�t (ha NumOfConsecutiveFramesWithoutAMatch=5, akkor StillBeingTracked=false lesz)

	bool crossedTheLine; // megadja, hogy a bounding box �thaladt-e m�r a vonalon

	// F�ggv�nyek
	BoundingBox(Rect _box, int _classId, float _confidence); // konstruktor
	void predictNextPosition(void); // megj�vend�li a bounding box k�vetkez� poz�ci�j�t az el�z� frame-eken l�v� poz�ci�ja alapj�n
};

#endif    // MY_BOUNDINGBOX
