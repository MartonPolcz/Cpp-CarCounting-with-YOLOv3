#ifndef MY_BOUNDINGBOX
#define MY_BOUNDINGBOX

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// A Bounding Boxokat leíró osztály
class BoundingBox {
public:
	
	Rect box; // a bounding box kerete

	int classId; // a bounding box osztálya

	float confidence; // a bounding box osztályának valószínûsége

	vector<Point> centerPositions; // megadja a bounding box pozícióit az elõzõ frame-eken

	double CurrentDiagonalSize; // megadja a bounding box átmérõjének hosszát pixelekben

	bool CurrentMatchFoundOrNewBox; // megadja, hogy a bounding box egy újonnan detektált autóhoz tartozik, vagy az már az elõzõ frame-n is szerepelt

	bool StillBeingTracked; // megadja, hogy a bounding box még létezik-e (az elmúlt 5 frame-n belül detektáltuk-e valaha)

	Point predictedNextPosition; // megadja, hogy a következõ frame-n hova várható az adott autóhoz tartozó bounding box

	int NumOfConsecutiveFramesWithoutAMatch; // megadja, hogy ha eltûnt a bounding box, akkor hány frame telt el azóta hogy nem találjuk párját (ha NumOfConsecutiveFramesWithoutAMatch=5, akkor StillBeingTracked=false lesz)

	bool crossedTheLine; // megadja, hogy a bounding box áthaladt-e már a vonalon

	// Függvények
	BoundingBox(Rect _box, int _classId, float _confidence); // konstruktor
	void predictNextPosition(void); // megjövendöli a bounding box következõ pozícióját az elõzõ frame-eken lévõ pozíciója alapján
};

#endif    // MY_BOUNDINGBOX
