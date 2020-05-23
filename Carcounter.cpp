// This code is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// P�lda a haszn�latra: ./object_detection_yolo.out -video=traffic.mp4
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
"{device d       |<none>| input webcam   }" //device=0 az alap�rtelmezett webkamera megnyit�s�hoz
; 

using namespace cv;
using namespace dnn;
using namespace std;

// V�ltoz�k inicializ�l�sa
float confThreshold = 0.5; // Konfidencia k�sz�b (default:0.5)
float nmsThreshold = 0.2;  // Non-maximum suppression k�sz�b (default:0.4)
int inpWidth = 416;  // A neur�lis h�l� bemeneti k�p�nek sz�less�ge - nagyobb �rt�k eset�n pontosabb, kisebb �rt�k eset�n gyorsabb eredm�nyt kapunk (default:416)
int inpHeight = 416; // A neur�lis h�l� bemeneti k�p�nek magass�ga (default:416)
int carCount = 0; // V�ltoz� ami sz�molja a vonalon �thalad� aut�kat
vector<string> classes; // Vektor, ami tartalmazza a coco.names-ben t�rolt oszt�lyok neveit (car, truck, dog, etc.)

// A YOLO algoritmus lefuttat�sa ut�n keletkezett bounding boxokat dolgozza fel (alacsony konfidenci�j� bounding boxok elt�vol�t�sa, ut�na az �tlapol�d� bounding boxok elt�vol�t�sa non-maximum suppressionnel, �sszetartoz� bounding boxok megkeres�se)
void postprocess(Mat& frame, const vector<Mat>& out, int i, vector<BoundingBox>& boundingboxes);

// A neur�lis h�l� output layereinek neveinek lek�rdez�se
vector<String> getOutputsNames(const Net& net);

// Megn�zi, hogy az aktu�lis frame-en detekt�lt bounding boxok szerepeltek-e m�r az el�z� frame-en is; ha igen, akkor azokat �sszep�ros�tjuk, ha nem akkor �j bounding boxot adunk hozz� a vector<BoundingBox> boundingboxes-hoz
void matchCurrentFrameBoundingBoxesToExistingBoundingBoxes(vector<BoundingBox>& boundingboxes, vector<BoundingBox>& currentFrameBoundingBoxes);

// Ha az aktu�lis frame-en detekt�lt bounding box szerepelt m�r az el�z� frame-en is, akkor ez a f�ggv�ny p�ros�tja azt �ssze az el�z� frame-en detekt�lt p�rj�val
void addBoundingBoxToExistingBoundingBoxes(BoundingBox& currentFrameBoundingBox, vector<BoundingBox>& boundingboxes, int& IndexOfLeastDistance);

// Ha az aktu�lis frame-en detekt�lt bounding boxnak nincsen p�rja az el�z� frame-en, akkor ez a f�ggv�ny mint �j bounding boxot adja hozz� a vector<BoundingBox> boundingboxes-hoz
void addNewBoundingBox(BoundingBox& currentFrameBoundingBox, vector<BoundingBox>& boundingboxes);

// Megadja k�t pont k�z�tti t�vols�got
double distanceBetweenPoints(Point point1, Point point2);

// Kirajzolja a v�gleges bounding boxokat az ablakba
void drawBoundingBoxesOnImage(vector<BoundingBox> &boundingboxes, Mat& frame);

// Megn�zi, hogy az aktu�lis frame-en �thaladt-e valamilyen aut� a vonalon
bool checkIfBlobsCrossedTheLine(vector<BoundingBox> &boundingboxes, int &HorizontalLinePosition, int &carCount);

// A vonalon �thaladt aut�k sz�m�nak kirajzol�sa az ablak jobb fels� sark�ba
void drawCarCountOnImage(int& carCount, Mat& frame);

int main(int argc, char** argv)
{
	//V�ltoz� a feldolgozott frame-ek sz�mol�s�hoz
	int i = 0; 

	//Vektor amiben a detekt�lt bounding boxokat t�roljuk el
	vector<BoundingBox> boundingboxes;

	//Konstruktor
	CommandLineParser parser(argc, argv, keys); 
	parser.about("Ezzel a programmal a YOLOv3 algoritmus es az OpenCV segitsegevel k�pfelismerest lehet vegezni egy adott videon/kepen/webkamera felvetelen.");
	if (parser.has("help"))
	{
		parser.printMessage();
		waitKey(10000);
		return 0;
	}

	// Az oszt�lyok neveinek beolvas�sa (80 darab) a coco.names f�jlb�l
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// A neur�lis h�l� konfigur�ci�s- �s s�lyf�jlj�nak beolvas�sa
	String modelConfiguration = "yolov3-tiny.cfg";
	String modelWeights = "yolov3-tiny.weights";

	// Az el�re betan�tott neur�lis h�l� fel�p�t�se, ami a YOLO algoritmust val�s�tja majd meg
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_OPENCL); //DNN_TARGET_CPU is lehet a target, de ekkor lassabb lesz a program

	// V�ltoz�k a vide�/k�p/webkamera megnyit�s�hoz
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;

	try {
		outputFile = "yolo.avi";
		if (parser.has("image"))
		{
			// K�pf�jl megnyit�sa
			str = parser.get<String>("image");
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo.jpg");
			outputFile = str;
		}
		else if (parser.has("video"))
		{
			// Vide�f�jl megnyit�sa
			str = parser.get<String>("video");
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo.avi");
			outputFile = str;
		}
		else if (parser.has("device"))
		{
			// Webkamera megnyit�sa
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

	// A video writer inicializ�l�sa (ha vide�- vagy webkamera felv�telt szeretn�nk feldolgozni)
	if (!parser.has("image")) {
		video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	}

	// Megjelen�t� ablak elk�sz�t�se
	static const string kWinName = "Aut�sz�ml�l�s YOLOv3 algoritmus seg�ts�g�vel";
	namedWindow(kWinName, WINDOW_NORMAL);

	// Sz�vegf�jl l�trehoz�sa amibe ki�ratjuk a frame feldolgoz�si id�ket
	ofstream myfile;
	myfile.open("times.txt");

	// Feldolgozand� els� frame beolvas�sa
	cap >> frame;

	// �thalad�si vonal elk�sz�t�se
	Point crossingLine[2];
	int HorizontalLinePosition = (int)std::round((double)frame.rows * 0.42);
	crossingLine[0].x = 0;
	crossingLine[0].y = HorizontalLinePosition;
	crossingLine[1].x = frame.cols - 1;
	crossingLine[1].y = HorizontalLinePosition;

	// Framek feldolgoz�sa egyes�vel
	while (waitKey(1) < 0)
	{

		// Hogyha elfogytak a feldolgozand� framek kil�p�nk a while-ciklusb�l
		if (frame.empty()) {
			cout << "A fajl feldolgozasa befejezodott!!!\n" << endl;
			cout << "A megszamlalt autokat a " << outputFile << " fajl tartalmazza." << endl;
			waitKey(5000);
			break;
		}
		
		// 4D blob kre�l�sa a frameb�l (mean subtraction n�lk�l, 255-�s scaling factorral)
		blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		// Blob �tad�sa bemenetk�nt a neur�lis h�l�nak 
		net.setInput(blob);

		// A YOLO algoritmus lefuttat�sa a feldolgozand� framen; azaz a blobot �thajtjuk a neur�lis h�l�n �s megkapjuk a h�l� �ltal detekt�lt bounding boxokat
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net)); 

		// Az alacsony konfidenci�j� bounding boxok elt�vol�t�sa, valamint az �tlapol�d� bounding boxok elt�vol�t�sa non-maximum suppressionnel
		postprocess(frame, outs, i, boundingboxes); 

		// Frame feldolgoz�si id� ki�rat�sa az ablak tetej�re
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

		// Ki�ratjuk a frame-feldolgoz�si id�ket a times.txt sz�vegf�jlba is
		myfile << to_string(i) << "\t" << t << "\n";

		// Megn�zi, hogy az aktu�lis frame-en �thaladt-e valamilyen aut� a vonalon
		bool atLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(boundingboxes, HorizontalLinePosition, carCount);

		// �thalad�si vonal kirajzol�sa az ablakba
		if (atLeastOneBlobCrossedTheLine == true) {
			cv::line(frame, crossingLine[0], crossingLine[1], Scalar(61, 181, 211), 2);
		}
		else {
			cv::line(frame, crossingLine[0], crossingLine[1], Scalar(86, 13, 61), 2);
		}

		// A vonalon �thaladott aut�k sz�m�nak kirajzol�sa az ablak jobb fels� sark�ba
		drawCarCountOnImage(carCount, frame);

		// Feldolgozott frame ki�rat�sa a kimeneti vide�f�jlba/k�pf�jlba.
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		if (parser.has("image")) imwrite(outputFile, detectedFrame);
		else video.write(detectedFrame);

		// A feldolgozott frame megjelen�t�se az ablakban
		imshow(kWinName, frame);

		// Ki�ratjuk a parancsorba �ppen hanyadik framet dolgozza fel a YOLO
		i++;
		cout << "A jelenleg feldolgozas alatt allo frame sorszama: " << i << endl; 

		// K�vetkez� feldolgozand� frame beolvas�sa
		cap >> frame;

	}

	//Bez�rjuk a vide�f�jlt �s a video writert
	cap.release();
	if (!parser.has("image")) video.release();

	//Bez�rjuk a times.txt sz�vegf�jlt amibe a frame feldolgoz�si id�ket �rtuk ki
	myfile.close();

	return 0;
}

// Az alacsony konfidenci�j� bounding boxok elt�vol�t�sa, valamint az �tlapol�d� bounding boxok elt�vol�t�sa non-maximum suppressionnel
void postprocess(Mat& frame, const vector<Mat>& outs, int i, vector<BoundingBox>& boundingboxes)
{

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	// Egyenk�nt v�gigmegy�nk az �sszes bounding boxon, azon bel�l megkeress�k melyik oszt�lynak a legnagyobb a 
	// val�sz�n�s�ge, �s ha az nagyobb mint a konfidencia k�sz�b, akkor megtartjuk a boxot �s hozz�rendelj�k a 
	// legnagyobb val�sz�n�s�ggel rendelkez� oszt�lyt
	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Megkeress�k a bounding boxhoz tartoz� legnagyobb val�sz�n�s�g� oszt�lyt
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			// Hogyha a legnagyobb val�sz�n�s�g� oszt�ly val�sz�n�s�ge nagyobb a konfidencia k�sz�bn�l, akkor 
			// megtartjuk a bounding boxot �s a vector<Rect> boxesba elmentj�k a kirajzoltat�s�hoz sz�ks�ges adatait
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

	// Non-maximum suppression elv�gz�se, hogy elt�vol�tsuk az alacsonyabb konfidenci�j� egym�ssal �tlapol�d� bounding boxokat
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// Itt fogjuk t�rolni a non-maximum suppression ut�ni bounding boxokat
	vector<BoundingBox> currentFrameBoundingBoxes;

	// V�gigmegy�nk a non-maximum suppression ut�n megmaradt v�gleges bounding boxokon �s egyenk�nt bet�ltj�k azokat a vector<BoundingBox> currentFrameBoundingBoxes-ba
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];

		BoundingBox newbox(box, classIds[idx], confidences[idx]);
		currentFrameBoundingBoxes.push_back(newbox);

	}

	// Hogyha ez az els� frame amit feldolgozunk, akkor egyenk�nt bet�ltj�k a mostani frame-n�l detekt�lt bounding boxokat a vector<BoundingBox> boundingboxes-ba
	if (i == 0) {
		for (auto &currentFrameBoundingBox : currentFrameBoundingBoxes) {
			boundingboxes.push_back(currentFrameBoundingBox);
		}
	}
	else {
		// Hogyha m�r vannak l�tez� boundingboxjaink az el�z� framekb�l, akkor �sszehasonl�tjuk azokkal a mostani framen�l detekt�lt bounding boxokat
		matchCurrentFrameBoundingBoxesToExistingBoundingBoxes(boundingboxes, currentFrameBoundingBoxes);
	}

	// Kirajzoljuk a v�gleges bounding boxokat az ablakba
	drawBoundingBoxesOnImage(boundingboxes, frame);

	currentFrameBoundingBoxes.clear();

}

// A h�l� output layereinek neveinek lek�rdez�se
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Az output layerek indexeinek lek�rdez�se
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//A h�l� �sszes layer�nek a nev�nek a lek�rdez�se
		vector<String> layersNames = net.getLayerNames();

		//Az output layerek nev�nek a lek�rdez�se
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

// A vonalon �thaladott aut�k sz�m�nak kirajzol�sa az ablak jobb fels� sark�ba
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

// Megn�zi, hogy az aktu�lis frame-en detekt�lt bounding boxok szerepeltek-e m�r az el�z� frame-en is; ha igen, akkor azokat �sszep�ros�tjuk �ket, ha nem akkor �j bounding boxot adunk hozz� a vector<BoundingBox> boundingboxes-hoz
void matchCurrentFrameBoundingBoxesToExistingBoundingBoxes(vector<BoundingBox>& boundingboxes, vector<BoundingBox>& currentFrameBoundingBoxes) {

	for (auto &boundingbox : boundingboxes) {

		boundingbox.CurrentMatchFoundOrNewBox = false;

		boundingbox.predictNextPosition();
	}
	// Minden az aktu�lis framen�l detekt�lt bounding boxra megn�zz�k, hogy melyik eddigi bounding box-hoz van a legk�zelebb
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
		// Hogyha az aktu�lis framen�l detekt�lt bounding box t�vols�ga nagyon kicsi a hozz� legk�zelebbi m�r l�tez� boundingbox j�solt k�vetkez� pontj�hoz k�pest, akkor a k�t bounding box igaz�b�l ugyanaz
		if (ValueOfLeastDistance < currentFrameBoundingBox.CurrentDiagonalSize * 1.0) {
			addBoundingBoxToExistingBoundingBoxes(currentFrameBoundingBox, boundingboxes, IndexOfLeastDistance);
		}
		else {
			addNewBoundingBox(currentFrameBoundingBox, boundingboxes);
		}

	}

	for (auto &boundingbox : boundingboxes) {
		// Hogyha egy az el�z� frame-en detekt�lt bounding boxnak nem tal�ltuk meg az aktu�lis frame-en a p�rj�t, n�velj�k a NumOfConsecutiveFramesWithoutAMatch v�ltoz�j�t
		if (boundingbox.CurrentMatchFoundOrNewBox == false) {
			boundingbox.NumOfConsecutiveFramesWithoutAMatch++;
			boundingbox.centerPositions.push_back(boundingbox.predictedNextPosition);
			boundingbox.box.x = boundingbox.centerPositions.back().x - (boundingbox.box.width / 2);
			boundingbox.box.y = boundingbox.centerPositions.back().y - (boundingbox.box.height / 2);
		}
		// Hogyha egy bounding boxnak az elm�lt 5 frame-en bel�l egyszer sem tal�ltuk meg a p�rj�t, akkor azt a bounding boxot t�r�ltnek nyilv�n�tjuk
		if (boundingbox.NumOfConsecutiveFramesWithoutAMatch >= 7) {
			boundingbox.StillBeingTracked = false;
		}

	}

}

// Megadja k�t pont k�z�tti t�vols�got
double distanceBetweenPoints(Point point1, Point point2) {
	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);
	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

// Ha az aktu�lis frame-en detekt�lt bounding box szerepelt m�r az el�z� frame-en is, akkor ez a f�ggv�ny p�ros�tja azt �ssze az el�z� frame-en detekt�lt p�rj�val
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

// Ha az aktu�lis frame-en detekt�lt bounding boxnak nincsen p�rja az el�z� frame-en, akkor ez a f�ggv�ny mint �j bounding boxot adja hozz� a vector<BoundingBox> boundingboxes-hoz
void addNewBoundingBox(BoundingBox& currentFrameBoundingBox, vector<BoundingBox>& boundingboxes) {

	currentFrameBoundingBox.CurrentMatchFoundOrNewBox = true;
	boundingboxes.push_back(currentFrameBoundingBox);
}

// Kirajzolja a v�gleges bounding boxokat az ablakba
void drawBoundingBoxesOnImage(vector<BoundingBox> &boundingboxes, Mat& frame) {

	for (unsigned int i = 0; i < boundingboxes.size(); i++) {

		if (boundingboxes[i].StillBeingTracked == true && boundingboxes[i].CurrentMatchFoundOrNewBox == true) {
			rectangle(frame, Point(boundingboxes[i].box.x, boundingboxes[i].box.y), Point(boundingboxes[i].box.x + boundingboxes[i].box.width, boundingboxes[i].box.y + boundingboxes[i].box.height), Scalar(100, 220, 80), 2);


			// A keret feliratoz�s�nak defini�l�sa, ami tartalmazza a felismert oszt�ly nev�t �s annak val�sz�n�s�g�t
			string label = format("%.1f", boundingboxes[i].confidence);
			int classId = boundingboxes[i].classId;
			if (!classes.empty())
			{
				CV_Assert(classId < (int)classes.size());
				label = classes[classId] + ":" + label;
			}

			// A feliratoz�s megjelen�t�se a keret tetej�n
			int baseLine;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			boundingboxes[i].box.y = max(boundingboxes[i].box.y, labelSize.height);
			rectangle(frame, Point(boundingboxes[i].box.x, boundingboxes[i].box.y - round(1.5*labelSize.height)), Point(boundingboxes[i].box.x + round(1.5*labelSize.width), boundingboxes[i].box.y + baseLine), Scalar(255, 255, 255), FILLED);
			putText(frame, label, Point(boundingboxes[i].box.x, boundingboxes[i].box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

			// Az aut� ID-j�nak ki�rat�sa
			int FontFace = CV_FONT_HERSHEY_SIMPLEX;
			double FontScale = boundingboxes[i].CurrentDiagonalSize / 80.0;
			int FontThickness = (int)std::round(FontScale * 1.0);
			cv::putText(frame, std::to_string(i), Point(boundingboxes[i].box.x, boundingboxes[i].box.y + boundingboxes[i].box.height), FontFace, FontScale, Scalar(0, 255, 0), FontThickness);
		}
	}
}

// Megn�zi, hogy az aktu�lis frame-en �thaladt-e valamilyen j�rm� a vonalon
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