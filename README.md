# Smart Face Attendance System  
A complete Face Detection & Recognition Attendance System built using **C++** and **OpenCV 2.4.13**, designed for academic projects and practical demonstrations.

It allows:
- Registering a new user’s face using a webcam  
- Saving the user image in the `faces/` directory  
- Recognizing the user from webcam during attendance mode  
- Writing attendance to `attendance.csv` with name, date, and time  
- Ensuring each person is marked **only once per session**  

---

# 📌 Features

### ✔ Face Detection  
Uses Haar Cascade classifier (`haarcascade_frontalface_default.xml`).

### ✔ Face Registration  
Captures face via webcam and saves as `faces/<name>.jpg`.

### ✔ Auto Database Loading  
On startup, the system scans `faces/` and loads all registered persons.

### ✔ Recognition  
Compares current face with saved face images using histogram correlation.

### ✔ Attendance Logging  
Saves attendance in:

Compile:
=========
cl main.cpp /EHsc ^
 /I"C:\Users\somit\opencv\build\include" ^
 /link /MACHINE:X86 /LIBPATH:"C:\Users\somit\opencv\build\x86\vc12\lib" ^
 opencv_core2413.lib opencv_highgui2413.lib opencv_imgproc2413.lib opencv_objdetect2413.lib


HOW TO USE THE SYSTEM
1️⃣ Option 2 — Register New Face

Run program

Choose:

2


Enter your name (no spaces):

Somit


Webcam opens

Press c to capture

The system will:

Save face image → faces/Somit.jpg

Add person to in-memory database

Display confirmation:

Face image saved as faces/Somit.jpg
Person 'Somit' added to database (id=1)

2️⃣ Option 1 — Take Attendance

Run main.exe

Choose:

1


System will:

Detect face

Compare with faces in faces/

If match found → display your name on video

Mark attendance ONLY ONCE per session

Example entry in CSV:
Somit,17-11-2025,09:45:33


File saved as:

attendance.csv
