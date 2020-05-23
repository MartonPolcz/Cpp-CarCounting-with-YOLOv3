// This code is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Példa a használatra: ./object_detection_yolo.out -video=traffic.mp4
//                  ./object_detection_yolo.out -image=city.jpg
//                 ./object_detection_yolo.out -device=0
#include <fstream>
#include <sstream>
#include <iostream>
#include<conio.h> 

#include "BoundingBox.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

const char* keys =
"{help h usage ? | | Pelda a hasznalatra: \n\t\t./object_detection_yolo.out --video=traffic.mp4 \n\t\t./object_detection_yolo.out --image=city.jpg}"
"{image i        |<none>| input image   }"
"{video v        |<none>| input video   }"
"{device d       |<none>| input webcam   }" //device=0 az alapértelmezett webkamera megnyitásához
; 

using namespace cv;
using namespace dnn;
using namespace std;

// Változók inicializálása
float confThreshold = 0.5; // Konfidencia küszöb (default:0.5)
float nmsThreshold = 0.2;  // Non-maximum suppression küszöb (default:0.4)
int inpWidth = 416;  // A neurális háló bemeneti képének szélessége - nagyobb érték esetén pontosabb, kisebb érték esetén gyorsabb eredményt kapunk (default:416)
int inpHeight = 416; // A neurális háló bemeneti képének magassága (default:416)
int carCount = 0; // Változó ami számolja a vonalon áthaladó autókat
vector<string> classes; // Vektor, ami tartalmazza a coco.names-ben tárolt osztályok neveit (car, truck, dog, etc.)

// A YOLO algoritmus lefuttatása után keletkezett bounding boxokat dolgozza fel (alacsony konfidenciájú bounding boxok eltávolítása, utána az átlapolódó bounding boxok eltávolítása non-maximum suppressionnel, összetartozó bounding boxok megkeresése)
void postprocess(Mat& frame, const vector<Mat>& out, int i, vector<BoundingBox>& boundingboxes);

// A neurális háló output layereinek neveinek lekérdezése
vector<String> getOutputsNames(const Net& net);

// Megnézi, hogy az aktuális frame-en detektált bounding boxok szerepeltek-e már az elõzõ frame-en is; ha igen, akkor azokat összepárosítjuk, ha nem akkor új bounding boxot adunk hozzá a vector<BoundingBox> boundingboxes-hoz
void matchCurrentFrameBoundingBoxesToExistingBoundingBoxes(vector<BoundingBox>& boundingboxes, vector<BoundingBox>& currentFrameBoundingBoxes);

// Ha az aktuális frame-en detektált bounding box szerepelt már az elõzõ frame-en is, akkor ez a függvény párosítja azt össze az elõzõ frame-en detektált párjával
void addBoundingBoxToExistingBoundingBoxes(BoundingBox& currentFrameBoundingBox, vector<BoundingBox>& boundingboxes, int& IndexOfLeastDistance);

// Ha az aktuális frame-en detektált bounding boxnak nincsen párja az elõzõ frame-en, akkor ez a függvény mint új bounding boxot adja hozzá a vector<BoundingBox> boundingboxes-hoz
void addNewBoundingBox(BoundingBox& currentFrameBoundingBox, vector<BoundingBox>& boundingboxes);

// Megadja két pont közötti távolságot
double distanceBetweenPoints(Point point1, Point point2);

// Kirajzolja a végleges bounding boxokat az ablakba
void drawBoundingBoxesOnImage(vector<BoundingBox> &boundingboxes, Mat& frame);

// Megnézi, hogy az aktuális frame-en áthaladt-e valamilyen autó a vonalon
bool checkIfBlobsCrossedTheLine(vector<BoundingBox> &boundingboxes, int &HorizontalLinePosition, int &carCount);

// A vonalon áthaladt autók számának kirajzolása az ablak jobb felsõ sarkába
void drawCarCountOnImage(int& carCount, Mat& frame);

int main(int argc, char** argv)
{
	//Változó a feldolgozott frame-ek számolásához
	int i = 0; 

	//Vektor amiben a detektált bounding boxokat tároljuk el
	vector<BoundingBox> boundingboxes;

	//Konstruktor
	CommandLineParser parser(argc, argv, keys); 
	parser.about("Ezzel a programmal a YOLOv3 algoritmus es az OpenCV segitsegevel képfelismerest lehet vegezni egy adott videon/kepen/webkamera felvetelen.");
	if (parser.has("help"))
	{
		parser.printMessage();
		waitKey(10000);
		return 0;
	}

	// Az osztályok neveinek beolvasása (80 darab) a coco.names fájlból
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// A neurális háló konfigurációs- és súlyfájljának beolvasása
	String modelConfiguration = "yolov3-tiny.cfg";
	String modelWeights = "yolov3-tiny.weights";

	// Az elõre betanított neurális háló felépítése, ami a YOLO algoritmust valósítja majd meg
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_OPENCL); //DNN_TARGET_CPU is lehet a target, de ekkor lassabb lesz a program

	// Változók a videó/kép/webkamera megnyitásához
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;

	try {
		outputFile = "yolo.avi";
		if (parser.has("image"))
		{
			// Képfájl megnyitása
			str = parser.get<String>("image");
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo.jpg");
			outputFile = str;
		}
		else if (parser.has("video"))
		{
			// Videófájl megnyitása
			str = parser.get<String>("video");
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo.avi");
			outputFile = str;
		}
		else if (parser.has("device"))
		{
			// Webkamera megnyitása
			cap.open(parser.get<int>("device"));
			outputFile = "webcam_yolo.avi";
		}
		else throw("error");
	}
	catch (...) {
		cout << "Nem sikerult megnyitni a videot/kepet/webkamerat. Valoszinuleg hibas nev lett megadva." << endl;
		waitKey(5000);
		return 0;
	}

	// A video writer inicializálása (ha videó- vagy webkamera felvételt szeretnénk feldolgozni)
	if (!parser.has("image")) {
		video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	}

	// Megjelenítõ ablak elkészítése
	static const string kWinName = "Autószámlálás YOLOv3 algoritmus segítségével";
	namedWindow(kWinName, WINDOW_NORMAL);

	// Szövegfájl létrehozása amibe kiíratjuk a frame feldolgozási idõket
	ofstream myfile;
	myfile.open("times.txt");

	// Feldolgozandó elsõ frame beolvasása
	cap >> frame;

	// Áthaladási vonal elkészítése
	Point crossingLine[2];
	int HorizontalLinePosition = (int)std::round((double)frame.rows * 0.42);
	crossingLine[0].x = 0;
	crossingLine[0].y = HorizontalLinePosition;
	crossingLine[1].x = frame.cols - 1;
	crossingLine[1].y = HorizontalLinePosition;

	// Framek feldolgozása egyesével
	while (waitKey(1) < 0)
	{

		// Hogyha elfogytak a feldolgozandó framek kilépünk a while-ciklusból
		if (frame.empty()) {
			cout << "A fajl feldolgozasa befejezodott!!!\n" << endl;
			cout << "A megszamlalt autokat a " << outputFile << " fajl tartalmazza." << endl;
			waitKey(5000);
			break;
		}
		
		// 4D blob kreálása a framebõl (mean subtraction nélkül, 255-ös scaling factorral)
		blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		// Blob átadása bemenetként a neurális hálónak 
		net.setInput(blob);

		// A YOLO algoritmus lefuttatása a feldolgozandó framen; azaz a blobot áthajtjuk a neurális hálón és megkapjuk a háló által detektált bounding boxokat
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net)); 

		// Az alacsony konfidenciájú bounding boxok eltávolítása, valamint az átlapolódó bounding boxok eltávolítása non-maximum suppressionnel
		postprocess(frame, outs, i, boundingboxes); 

		// Frame feldolgozási idõ kiíratása az ablak tetejére
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq; 
		string label = format("Frame feldolgozasi ideje : %.2f ms", t);
		int FontFace = CV_FONT_HERSHEY_TRIPLEX;
		double FontScale = (frame.rows * frame.cols) / ((frame.rows / 360)*600000.0);
		int FontThickness = (int)std::round(FontScale);
		Size textSize = getTextSize(label, FontFace, FontScale, FontThickness, 0);
		rectangle(frame, Point(5, textSize.height + 15), Point(15 + textSize.width, 5), Scalar(29, 47, 192), FILLED);
		putText(frame, label, Point(10, textSize.height + 10), FontFace, FontScale, Scalar(255, 255, 255), FontThickness);

		// Kiíratjuk a frame-feldolgozási idõket a times.txt szövegfájlba is
		myfile << to_string(i) << "\t" << t << "\n";

		// Megnézi, hogy az aktuális frame-en áthaladt-e valamilyen autó a vonalon
		bool atLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(boundingboxes, HorizontalLinePosition, carCount);

		// Áthaladási vonal kirajzolása az ablakba
		if (atLeastOneBlobCrossedTheLine == true) {
			cv::line(frame, crossingLine[0], crossingLine[1], Scalar(61, 181, 211), 2);
		}
		else {
			cv::line(frame, crossingLine[0], crossingLine[1], Scalar(86, 13, 61), 2);
		}

		// A vonalon áthaladott autók számának kirajzolása az ablak jobb felsõ sarkába
		drawCarCountOnImage(carCount, frame);

		// Feldolgozott frame kiíratása a kimeneti videófájlba/képfájlba.
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		if (parser.has("image")) imwrite(outputFile, detectedFrame);
		else video.write(detectedFrame);

		// A feldolgozott frame megjelenítése az ablakban
		imshow(kWinName, frame);

		// Kiíratjuk a parancsorba éppen hanyadik framet dolgozza fel a YOLO
		i++;
		cout << "A jelenleg feldolgozas alatt allo frame sorszama: " << i << endl; 

		// Következõ feldolgozandó frame beolvasása
		cap >> frame;

	}

	//Bezárjuk a videófájlt és a video writert
	cap.release();
	if (!parser.has("image")) video.release();

	//Bezárjuk a times.txt szövegfájlt amibe a frame feldolgozási idõket írtuk ki
	myfile.close();

	return 0;
}

// Az alacsony konfidenciájú bounding boxok eltávolítása, valamint az átlapolódó bounding boxok eltávolítása non-maximum suppressionnel
void postprocess(Mat& frame, const vector<Mat>& outs, int i, vector<BoundingBox>& boundingboxes)
{

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	// Egyenként végigmegyünk az összes bounding boxon, azon belül megkeressük melyik osztálynak a legnagyobb a 
	// valószínûsége, és ha az nagyobb mint a konfidencia küszöb, akkor megtartjuk a boxot és hozzárendeljük a 
	// legnagyobb valószínûséggel rendelkezõ osztályt
	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Megkeressük a bounding boxhoz tartozó legnagyobb valószínûségû osztályt
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			// Hogyha a legnagyobb valószínûségû osztály valószínûsége nagyobb a konfidencia küszöbnél, akkor 
			// megtartjuk a bounding boxot és a vector<Rect> boxesba elmentjük a kirajzoltatásához szükséges adatait
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Non-maximum suppression elvégzése, hogy eltávolítsuk az alacsonyabb konfidenciájú egymással átlapolódó bounding boxokat
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// Itt fogjuk tárolni a non-maximum suppression utáni bounding boxokat
	vector<BoundingBox> currentFrameBoundingBoxes;

	// Végigmegyünk a non-maximum suppression után megmaradt végleges bounding boxokon és egyenként betöltjük azokat a vector<BoundingBox> currentFrameBoundingBoxes-ba
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];

		BoundingBox newbox(box, classIds[idx], confidences[idx]);
		currentFrameBoundingBoxes.push_back(newbox);

	}

	// Hogyha ez az elsõ frame amit feldolgozunk, akkor egyenként betöltjük a mostani frame-nél detektált bounding boxokat a vector<BoundingBox> boundingboxes-ba
	if (i == 0) {
		for (auto &currentFrameBoundingBox : currentFrameBoundingBoxes) {
			boundingboxes.push_back(currentFrameBoundingBox);
		}
	}
	else {
		// Hogyha már vannak létezõ boundingboxjaink az elõzõ framekbõl, akkor összehasonlítjuk azokkal a mostani framenél detektált bounding boxokat
		matchCurrentFrameBoundingBoxesToExistingBoundingBoxes(boundingboxes, currentFrameBoundingBoxes);
	}

	// Kirajzoljuk a végleges bounding boxokat az ablakba
	drawBoundingBoxesOnImage(boundingboxes, frame);

	currentFrameBoundingBoxes.clear();

}

// A háló output layereinek neveinek lekérdezése
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Az output layerek indexeinek lekérdezése
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//A háló összes layerének a nevének a lekérdezése
		vector<String> layersNames = net.getLayerNames();

		//Az output layerek nevének a lekérdezése
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

// A vonalon áthaladott autók számának kirajzolása az ablak jobb felsõ sarkába
void drawCarCountOnImage(int& carCount, Mat& frame) {
	int FontFace = CV_FONT_HERSHEY_COMPLEX;
	double FontScale = (frame.rows * frame.cols) / ((frame.rows / 360)*80000.0);
	int FontThickness = (int)std::round(FontScale * 1.25);
	Size textSize = cv::getTextSize(std::to_string(carCount), FontFace, FontScale, FontThickness, 0);

	double x = 4.5;
	if (carCount >= 10 && carCount < 100) {
		x = 2.25;
	}
	if (carCount >= 100) {
		x = 1.5;
	}

	Point ptCounterBottomLeftPosition;
	ptCounterBottomLeftPosition.x = frame.cols - 1 - (int)((double)textSize.width * 1.25);
	ptCounterBottomLeftPosition.y = (int)((double)textSize.height * 1.4);
	Point rectangleBottomLeftPosition;
	Point rectangleTopRightPosition;
	rectangleBottomLeftPosition.x = frame.cols - 1 - (int)((double)textSize.width * 1.3);
	rectangleBottomLeftPosition.y = (int)((double)textSize.height * 1.45);
	rectangleTopRightPosition.x = frame.cols - 1 - (int)((double)textSize.width * 0.2);
	rectangleTopRightPosition.y = (int)((double)textSize.height * 0.35);
	Point ptTextBottomLeftPosition;
	ptTextBottomLeftPosition.x = frame.cols - 1 - (int)((double)textSize.width * x);
	ptTextBottomLeftPosition.y = (int)((double)textSize.height * 0.3);
	int ptTextFontThickness = (int)std::round(FontScale * 0.25);

	rectangle(frame, Point(rectangleBottomLeftPosition.x, rectangleBottomLeftPosition.y), Point(rectangleTopRightPosition.x, rectangleTopRightPosition.y), Scalar(29, 47, 192), FILLED);
	putText(frame, "Vonalon athaladott autok:", ptTextBottomLeftPosition, FontFace, FontScale*0.2, Scalar(0, 0, 0), ptTextFontThickness);
	if (carCount < 10) {
		putText(frame, std::to_string(carCount), ptCounterBottomLeftPosition, FontFace, FontScale, Scalar(255, 255, 255), FontThickness);
	}
	else {
		putText(frame, std::to_string(carCount), ptCounterBottomLeftPosition, FontFace, FontScale, Scalar(255, 255, 255), FontThickness);
	}
}

// Megnézi, hogy az aktuális frame-en detektált bounding boxok szerepeltek-e már az elõzõ frame-en is; ha igen, akkor azokat összepárosítjuk õket, ha nem akkor új bounding boxot adunk hozzá a vector<BoundingBox> boundingboxes-hoz
void matchCurrentFrameBoundingBoxesToExistingBoundingBoxes(vector<BoundingBox>& boundingboxes, vector<BoundingBox>& currentFrameBoundingBoxes) {

	for (auto &boundingbox : boundingboxes) {

		boundingbox.CurrentMatchFoundOrNewBox = false;

		boundingbox.predictNextPosition();
	}
	// Minden az aktuális framenél detektált bounding boxra megnézzük, hogy melyik eddigi bounding box-hoz van a legközelebb
	for (auto &currentFrameBoundingBox : currentFrameBoundingBoxes) {

		int IndexOfLeastDistance = 0;
		double ValueOfLeastDistance = 100000.0;

		for (unsigned int i = 0; i < boundingboxes.size(); i++) {

			if (boundingboxes[i].StillBeingTracked == true) {

				double Distance = distanceBetweenPoints(currentFrameBoundingBox.centerPositions.back(), boundingboxes[i].predictedNextPosition);

				if (Distance < ValueOfLeastDistance) {
					ValueOfLeastDistance = Distance;
					IndexOfLeastDistance = i;
				}
			}
		}
		// Hogyha az aktuális framenél detektált bounding box távolsága nagyon kicsi a hozzá legközelebbi már létezõ boundingbox jósolt következõ pontjához képest, akkor a két bounding box igazából ugyanaz
		if (ValueOfLeastDistance < currentFrameBoundingBox.CurrentDiagonalSize * 1.0) {
			addBoundingBoxToExistingBoundingBoxes(currentFrameBoundingBox, boundingboxes, IndexOfLeastDistance);
		}
		else {
			addNewBoundingBox(currentFrameBoundingBox, boundingboxes);
		}

	}

	for (auto &boundingbox : boundingboxes) {
		// Hogyha egy az elõzõ frame-en detektált bounding boxnak nem találtuk meg az aktuális frame-en a párját, növeljük a NumOfConsecutiveFramesWithoutAMatch változóját
		if (boundingbox.CurrentMatchFoundOrNewBox == false) {
			boundingbox.NumOfConsecutiveFramesWithoutAMatch++;
			boundingbox.centerPositions.push_back(boundingbox.predictedNextPosition);
			boundingbox.box.x = boundingbox.centerPositions.back().x - (boundingbox.box.width / 2);
			boundingbox.box.y = boundingbox.centerPositions.back().y - (boundingbox.box.height / 2);
		}
		// Hogyha egy bounding boxnak az elmúlt 5 frame-en belül egyszer sem találtuk meg a párját, akkor azt a bounding boxot töröltnek nyilvánítjuk
		if (boundingbox.NumOfConsecutiveFramesWithoutAMatch >= 7) {
			boundingbox.StillBeingTracked = false;
		}

	}

}

// Megadja két pont közötti távolságot
double distanceBetweenPoints(Point point1, Point point2) {
	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);
	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

// Ha az aktuális frame-en detektált bounding box szerepelt már az elõzõ frame-en is, akkor ez a függvény párosítja azt össze az elõzõ frame-en detektált párjával
void addBoundingBoxToExistingBoundingBoxes(BoundingBox& currentFrameBoundingBox, vector<BoundingBox>& boundingboxes, int& IndexOfLeastDistance) {

	boundingboxes[IndexOfLeastDistance].box = currentFrameBoundingBox.box;
	boundingboxes[IndexOfLeastDistance].classId = currentFrameBoundingBox.classId;
	boundingboxes[IndexOfLeastDistance].confidence = currentFrameBoundingBox.confidence;

	boundingboxes[IndexOfLeastDistance].centerPositions.push_back(currentFrameBoundingBox.centerPositions.back());
	boundingboxes[IndexOfLeastDistance].CurrentDiagonalSize = currentFrameBoundingBox.CurrentDiagonalSize;

	boundingboxes[IndexOfLeastDistance].CurrentMatchFoundOrNewBox = true;
	boundingboxes[IndexOfLeastDistance].StillBeingTracked = true;
	boundingboxes[IndexOfLeastDistance].NumOfConsecutiveFramesWithoutAMatch = 0;
}

// Ha az aktuális frame-en detektált bounding boxnak nincsen párja az elõzõ frame-en, akkor ez a függvény mint új bounding boxot adja hozzá a vector<BoundingBox> boundingboxes-hoz
void addNewBoundingBox(BoundingBox& currentFrameBoundingBox, vector<BoundingBox>& boundingboxes) {

	currentFrameBoundingBox.CurrentMatchFoundOrNewBox = true;
	boundingboxes.push_back(currentFrameBoundingBox);
}

// Kirajzolja a végleges bounding boxokat az ablakba
void drawBoundingBoxesOnImage(vector<BoundingBox> &boundingboxes, Mat& frame) {

	for (unsigned int i = 0; i < boundingboxes.size(); i++) {

		if (boundingboxes[i].StillBeingTracked == true && boundingboxes[i].CurrentMatchFoundOrNewBox == true) {
			rectangle(frame, Point(boundingboxes[i].box.x, boundingboxes[i].box.y), Point(boundingboxes[i].box.x + boundingboxes[i].box.width, boundingboxes[i].box.y + boundingboxes[i].box.height), Scalar(100, 220, 80), 2);


			// A keret feliratozásának definiálása, ami tartalmazza a felismert osztály nevét és annak valószínûségét
			string label = format("%.1f", boundingboxes[i].confidence);
			int classId = boundingboxes[i].classId;
			if (!classes.empty())
			{
				CV_Assert(classId < (int)classes.size());
				label = classes[classId] + ":" + label;
			}

			// A feliratozás megjelenítése a keret tetején
			int baseLine;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			boundingboxes[i].box.y = max(boundingboxes[i].box.y, labelSize.height);
			rectangle(frame, Point(boundingboxes[i].box.x, boundingboxes[i].box.y - round(1.5*labelSize.height)), Point(boundingboxes[i].box.x + round(1.5*labelSize.width), boundingboxes[i].box.y + baseLine), Scalar(255, 255, 255), FILLED);
			putText(frame, label, Point(boundingboxes[i].box.x, boundingboxes[i].box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

			// Az autó ID-jának kiíratása
			int FontFace = CV_FONT_HERSHEY_SIMPLEX;
			double FontScale = boundingboxes[i].CurrentDiagonalSize / 80.0;
			int FontThickness = (int)std::round(FontScale * 1.0);
			cv::putText(frame, std::to_string(i), Point(boundingboxes[i].box.x, boundingboxes[i].box.y + boundingboxes[i].box.height), FontFace, FontScale, Scalar(0, 255, 0), FontThickness);
		}
	}
}

// Megnézi, hogy az aktuális frame-en áthaladt-e valamilyen jármû a vonalon
bool checkIfBlobsCrossedTheLine(vector<BoundingBox> &boundingboxes, int &HorizontalLinePosition, int &carCount) {
	bool atLeastOneBlobCrossedTheLine = false;

	int i = 0;
	vector<int> indexesOfCrossedBoundingBoxes;

	for (auto boundingbox : boundingboxes) {
		
		if (boundingbox.StillBeingTracked == true && (boundingbox.centerPositions.size() >= 2) && boundingbox.crossedTheLine == false && boundingbox.CurrentMatchFoundOrNewBox == true) {

			int currFrameIndex = (int)boundingbox.centerPositions.size() - 1;

			for (int idx = 0; idx < currFrameIndex; idx++) {
				if (((boundingbox.centerPositions[idx].y > HorizontalLinePosition && boundingbox.centerPositions[currFrameIndex].y <= HorizontalLinePosition) || (boundingbox.centerPositions[idx].y < HorizontalLinePosition && boundingbox.centerPositions[currFrameIndex].y >= HorizontalLinePosition)) && (boundingbox.classId == 2 || boundingbox.classId == 3 || boundingbox.classId == 5 || boundingbox.classId == 7)) {
					carCount++;
					atLeastOneBlobCrossedTheLine = true;
					indexesOfCrossedBoundingBoxes.push_back(i);
					break;
				}
			}
		}
		i++;
	}
	for (auto indexOfCrossedBoundingBox : indexesOfCrossedBoundingBoxes) {
		boundingboxes[indexOfCrossedBoundingBox].crossedTheLine = true;
	}

	return atLeastOneBlobCrossedTheLine;
}