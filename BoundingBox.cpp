#include "BoundingBox.h"

using namespace cv;
using namespace dnn;
using namespace std;

// Konstruktor
BoundingBox::BoundingBox(Rect _box, int _classId, float _confidence) {
	
	// Itt módosítjuk a YOLO által detektált bounding boxokat (középpontosan nyújtjuk õket kétszeresen), hogy nagyobb, az egész jármûvet átölelõ kereteket rajzoljunk ki
	box.x = (int)round(_box.x - _box.width / 2);
	box.y = (int)round(_box.y - _box.height / 2);
	box.width = _box.width * 2;
	box.height = _box.height * 2;

	classId = _classId;
	confidence = _confidence;
	Point currentCenter;
	currentCenter.x = (box.x + box.x + box.width) / 2;
	currentCenter.y = (box.y + box.y + box.height) / 2;
	centerPositions.push_back(currentCenter);
	CurrentDiagonalSize = sqrt(pow(box.width, 2) + pow(box.height, 2));
	StillBeingTracked = true;
	CurrentMatchFoundOrNewBox = true;
	NumOfConsecutiveFramesWithoutAMatch = 0;
	crossedTheLine = false;
}

// Megjövendöli a bounding box következõ pozícióját az elõzõ frame-eken lévõ pozíciói alapján
void BoundingBox::predictNextPosition(void) {
	int numPositions = (int)centerPositions.size();
	// Megnézzük, hány elõzõ frame-beli pozíciót tartalmaz a bounding box centerPositions attribútuma, és az eddigi pozíciói alapján súlyozva jósoljuk meg a következõ pozícióját
	if (numPositions == 1) {

		predictedNextPosition.x = centerPositions.back().x;
		predictedNextPosition.y = centerPositions.back().y;

	}
	else if (numPositions >= 2) {

		int deltaX = centerPositions[1].x - centerPositions[0].x;
		int deltaY = centerPositions[1].y - centerPositions[0].y;

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else if (numPositions == 3) {

		int sumOfXChanges = ((centerPositions[2].x - centerPositions[1].x) * 2) +
			((centerPositions[1].x - centerPositions[0].x) * 1);

		int deltaX = (int)std::round((float)sumOfXChanges / 3.0);

		int sumOfYChanges = ((centerPositions[2].y - centerPositions[1].y) * 2) +
			((centerPositions[1].y - centerPositions[0].y) * 1);

		int deltaY = (int)std::round((float)sumOfYChanges / 3.0);

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else if (numPositions == 4) {

		int sumOfXChanges = ((centerPositions[3].x - centerPositions[2].x) * 3) +
			((centerPositions[2].x - centerPositions[1].x) * 2) +
			((centerPositions[1].x - centerPositions[0].x) * 1);

		int deltaX = (int)std::round((float)sumOfXChanges / 6.0);

		int sumOfYChanges = ((centerPositions[3].y - centerPositions[2].y) * 3) +
			((centerPositions[2].y - centerPositions[1].y) * 2) +
			((centerPositions[1].y - centerPositions[0].y) * 1);

		int deltaY = (int)std::round((float)sumOfYChanges / 6.0);

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else if (numPositions >= 5) {

		int sumOfXChanges = ((centerPositions[numPositions - 1].x - centerPositions[numPositions - 2].x) * 4) +
			((centerPositions[numPositions - 2].x - centerPositions[numPositions - 3].x) * 3) +
			((centerPositions[numPositions - 3].x - centerPositions[numPositions - 4].x) * 2) +
			((centerPositions[numPositions - 4].x - centerPositions[numPositions - 5].x) * 1);

		int deltaX = (int)std::round((float)sumOfXChanges / 10.0);

		int sumOfYChanges = ((centerPositions[numPositions - 1].y - centerPositions[numPositions - 2].y) * 4) +
			((centerPositions[numPositions - 2].y - centerPositions[numPositions - 3].y) * 3) +
			((centerPositions[numPositions - 3].y - centerPositions[numPositions - 4].y) * 2) +
			((centerPositions[numPositions - 4].y - centerPositions[numPositions - 5].y) * 1);

		int deltaY = (int)std::round((float)sumOfYChanges / 10.0);

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else {
		// sosem szabad ide jutnia a programnak
	}
}