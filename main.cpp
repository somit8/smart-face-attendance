#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

// Structure to store person data
typedef struct {
    char name[50];
    int id;
    Mat face;   // stored grayscale, resized face image
} Person;

// Global variables
vector<Person> database;
CascadeClassifier face_cascade;
vector<string> marked_names;   // who already got attendance in this run

// Function prototypes
void load_face_database();
int detect_and_recognize_face(Mat &frame);
void mark_attendance(const char* name);
double compare_faces(Mat &face1, Mat &face2);
void capture_new_face(const char* name);
bool already_marked(const char* name);

int main() {
    printf("=== Smart Attendance System ===\n");

    // Load Haar cascade for face detection
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        printf("Error loading face cascade classifier!\n");
        return -1;
    }

    // Load face database from faces/ folder
    load_face_database();

    // Initialize camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Error opening camera!\n");
        return -1;
    }

    Mat frame;
    int choice;

    printf("\n1. Take Attendance\n2. Register New Face\n3. Exit\n");
    printf("Enter choice: ");
    scanf("%d", &choice);

    switch (choice) {
        case 1:
            if (database.empty()) {
                printf("No registered faces found! Please register at least one face first.\n");
                break;
            }

            printf("Starting attendance capture...\n");
            printf("Press 'q' to quit.\n");

            while (true) {
                cap >> frame;
                if (frame.empty()) break;

                // Detect and recognize faces
                int person_id = detect_and_recognize_face(frame);

                if (person_id != -1) {
                    const char* pname = database[person_id].name;

                    if (!already_marked(pname)) {
                        printf("Detected: %s (first time in this session)\n", pname);
                        mark_attendance(pname);
                        marked_names.push_back(string(pname));
                    } else {
                        // Already marked in this run; do nothing
                        // printf("Already marked: %s\n", pname);
                    }
                }

                imshow("Attendance System", frame);

                char key = (char)waitKey(1);
                if (key == 'q' || key == 'Q') break;
            }
            break;

        case 2: {
            char name[50];
            printf("Enter name for new face (no spaces): ");
            scanf("%s", name);
            capture_new_face(name);
            break;
        }

        case 3:
            printf("Exiting...\n");
            break;

        default:
            printf("Invalid choice.\n");
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

// Load faces from faces/*.jpg and build database
void load_face_database() {
    printf("Loading face database from 'faces' folder...\n");
    database.clear();

    vector<String> files;
    glob("faces/*.jpg", files, false);

    for (size_t i = 0; i < files.size(); ++i) {
        Mat img = imread(files[i], CV_LOAD_IMAGE_GRAYSCALE);
        if (img.empty()) {
            printf("Warning: Could not load %s\n", files[i].c_str());
            continue;
        }

        // Resize to a fixed size for comparison
        Mat resized;
        resize(img, resized, Size(100, 100));

        Person p;
        // Extract name from filename
        string path = files[i];
        size_t pos = path.find_last_of("/\\");
        string filename = (pos == string::npos) ? path : path.substr(pos + 1);
        size_t dot = filename.find_last_of('.');
        string name = (dot == string::npos) ? filename : filename.substr(0, dot);

        strncpy(p.name, name.c_str(), sizeof(p.name) - 1);
        p.name[sizeof(p.name) - 1] = '\0';
        p.id = (int)database.size() + 1;
        p.face = resized;

        database.push_back(p);
    }

    printf("Loaded %lu registered faces\n", (unsigned long)database.size());
}

// Check if this name already got attendance in this session
bool already_marked(const char* name) {
    string s(name);
    for (size_t i = 0; i < marked_names.size(); ++i) {
        if (marked_names[i] == s) return true;
    }
    return false;
}

// Detect face, compare with database, return index or -1
int detect_and_recognize_face(Mat &frame) {
    vector<Rect> faces;
    Mat frame_gray;

    // Convert to grayscale
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0, Size(30, 30));

    int bestIndex = -1;
    double bestScore = -1.0;
    Mat face_roi_resized;

    for (size_t i = 0; i < faces.size(); i++) {
        // Draw rectangle around face
        rectangle(frame, faces[i], Scalar(0, 255, 0), 2);

        // Extract face region (grayscale)
        Mat face_roi = frame_gray(faces[i]).clone();
        resize(face_roi, face_roi_resized, Size(100, 100));

        // Compare with each person in database
        for (size_t j = 0; j < database.size(); ++j) {
            double score = compare_faces(face_roi_resized, database[j].face);
            if (score > bestScore) {
                bestScore = score;
                bestIndex = (int)j;
            }
        }

        // Show label based on threshold
        string label = "Unknown";
        double threshold = 0.7; // you can tune this (0.0 - 1.0, higher is stricter)

        if (bestIndex != -1 && bestScore > threshold) {
            label = database[bestIndex].name;
        } else {
            bestIndex = -1; // treat as unknown
        }

        putText(frame, label,
                Point(faces[i].x, faces[i].y - 10),
                FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
    }

    return bestIndex; // -1 if none recognized
}

void mark_attendance(const char* name) {
    FILE *file = fopen("attendance.csv", "a");
    if (file == NULL) {
        printf("Error opening attendance file!\n");
        return;
    }

    // Get current time
    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    fprintf(file, "%s,%02d-%02d-%04d,%02d:%02d:%02d\n",
            name,
            t->tm_mday, t->tm_mon + 1, t->tm_year + 1900,
            t->tm_hour, t->tm_min, t->tm_sec);

    fclose(file);
    printf("Attendance marked for %s\n", name);
}

double compare_faces(Mat &face1, Mat &face2) {
    // Simple histogram comparison
    Mat hist1, hist2;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    calcHist(&face1, 1, 0, Mat(), hist1, 1, &histSize, &histRange);
    calcHist(&face2, 1, 0, Mat(), hist2, 1, &histSize, &histRange);

    // CV_COMP_CORREL for OpenCV 2.4
    return compareHist(hist1, hist2, CV_COMP_CORREL);
}

void capture_new_face(const char* name) {
    printf("Capturing new face for %s...\n", name);
    printf("Look at the camera and press 'c' to capture, 'q' to quit.\n");

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Error opening camera!\n");
        return;
    }

    Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        imshow("Capture Face - Press 'c' to capture", frame);

        char key = (char)waitKey(1);
        if (key == 'c' || key == 'C') {
            // Convert to gray and try to detect face from this frame
            Mat gray;
            cvtColor(frame, gray, CV_BGR2GRAY);
            vector<Rect> faces;
            face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

            Mat face_roi;

            if (!faces.empty()) {
                face_roi = gray(faces[0]).clone();
            } else {
                printf("No face clearly detected, using whole frame as face.\n");
                face_roi = gray.clone();
            }

            Mat resized;
            resize(face_roi, resized, Size(100, 100));

            // Save the full frame as the visible image
            char filename[200];
            sprintf(filename, "faces/%s.jpg", name);
            imwrite(filename, frame);
            printf("Face image saved as %s\n", filename);

            // Add to in-memory database now
            Person p;
            strncpy(p.name, name, sizeof(p.name) - 1);
            p.name[sizeof(p.name) - 1] = '\0';
            p.id = (int)database.size() + 1;
            p.face = resized;
            database.push_back(p);

            printf("Person '%s' added to database (id=%d)\n", p.name, p.id);
            break;
        }
        if (key == 'q' || key == 'Q') {
            printf("Cancelled capture.\n");
            break;
        }
    }

    cap.release();
    destroyAllWindows();
}
