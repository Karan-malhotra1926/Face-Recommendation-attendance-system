import cv2, os, time, smtplib, pickle
import numpy as np
from flask import Flask, render_template, Response, redirect, request, session
from datetime import datetime
from email.mime.text import MIMEText
import mysql.connector

app = Flask(__name__)
app.secret_key = "1926"

# 👥 USERS
USERS = {
    "admin": {"password": "1926", "role": "admin"},
    "user": {"password": "1234", "role": "user"}
}

# 📧 EMAIL (optional)
EMAIL = "km367431@gmail.com"
EMAIL_PASS = "acxawsdhyqeegpnl"
TO_EMAIL = "km367431@gmail.com"

# 🗄 DATABASE
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="karan1926",
    database="face_attendance"
)
cursor = conn.cursor()

# 🤖 MODEL
model = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists("trainer.yml"):
    model.read("trainer.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 🧠 LOAD NAMES
def load_names():
    if os.path.exists("labels.pickle"):
        with open("labels.pickle", "rb") as f:
            label_map = pickle.load(f)
        return {v: k for k, v in label_map.items()}
    return {}

names = load_names()

# 🔧 CONTROL
last_name = ""
count = 0
last_unknown_time = 0
email_sent = False
EMAIL_COOLDOWN = 60 # seconds
marked_names = set()
attendance_message = ""
last_mark_time = 0
frame_count = 0

# 🔐 ADMIN CHECK
def is_admin():
    return session.get("role") == "admin"

# 📧 EMAIL
def send_email():
    try:
        msg = MIMEText("Unknown detected❌")
        msg['Subject'] = "Alert"
        msg['From'] = EMAIL
        msg['To'] = TO_EMAIL

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL, EMAIL_PASS)
        server.sendmail(EMAIL, TO_EMAIL, msg.as_string())
        server.quit()
    except:
        pass

# 📊 ATTENDANCE
def mark_attendance(name):
    global marked_names, attendance_message, last_mark_time

    if name in marked_names:
        return

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    cursor.execute(
        "INSERT INTO attendance (name, date, time) VALUES (%s, %s, %s)",
        (name, date, time_now)
    )
    conn.commit()

    marked_names.add(name)

    attendance_message = f"Marked: {name}"
    last_mark_time = time.time()

# 🧠 TRAIN
def train_model():
    global names, model

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists("dataset"):
        return

    for file in os.listdir("dataset"):
        path = os.path.join("dataset", file)
        person = file.split('.')[0]

        if person not in label_map:
            label_map[person] = current_label
            current_label += 1

        label = label_map[person]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label)

    if len(faces) == 0:
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))
    model.save("trainer.yml")

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_map, f)

    names = {v: k for k, v in label_map.items()}

# 🎥 CAMERA
def generate_frames():
    global names

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200,200))

            try:
                label, confidence = model.predict(face)
            except:
                continue

            if confidence < 100 and label in names:
                name = names[label]
                color = (0,255,0)

                mark_attendance(name)

            else:
                name = "Unknown"
                color = (0,0,255)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            if attendance_message and (time.time() - last_mark_time < 3):
              cv2.putText(frame, attendance_message, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)

# 🌐 ROUTES
@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        u = request.form['username']
        p = request.form['password']

        if u in USERS and USERS[u]["password"] == p:
            session['user'] = u
            session['role'] = USERS[u]["role"]
            return redirect('/camera')

        return "Wrong Login"

    return render_template('login.html')

@app.route('/camera')
def camera():
    if 'user' not in session:
        return redirect('/login')
    return render_template('camera.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
from flask import request
import base64

@app.route('/mobile')
def mobile():
    return render_template('mobile.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.form['image']

    img_data = base64.b64decode(data.split(',')[1])

    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.2,4)

    name = "Unknown"

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(200,200))

        label, confidence = model.predict(face)

        if confidence < 90 and label in names:
            name = names[label]
            mark_attendance(name)

    return name    

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/capture_live')
def capture_live():
    name = request.args.get('name')

    cap = cv2.VideoCapture(0)
    count = 0

    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    while count < 20:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.2,4)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(200,200))

            cv2.imwrite(f"dataset/{name}.{count}.jpg", face)
            count += 1

    cap.release()

    return "DONE"
@app.route('/capture')
def capture():
    name = request.args.get('name')

    cap = cv2.VideoCapture(0)
    count = 0

    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    while count < 20:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.2,4)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(200,200))

            cv2.imwrite(f"dataset/{name}.{count}.jpg", face)
            count += 1
    with open("pending_users.txt", "a") as f:
     f.write(name + "\n")

    return "✅ Register Successful! Ask admin to approve"        

    cap.release()

    return "DONE"

@app.route('/train')
def train():
    if not is_admin():
        return "Access Denied ❌"
    train_model()
    return "Training Done!"

@app.route('/dashboard')
def dashboard():
    if not is_admin():
        return "Access Denied ❌"

    cursor.execute("SELECT * FROM attendance")
    data = cursor.fetchall()

    # graph data
    cursor.execute("SELECT date, COUNT(*) FROM attendance GROUP BY date")
    chart_data = cursor.fetchall()

    dates = [str(row[0]) for row in chart_data]
    counts = [row[1] for row in chart_data]

    # 🔥 LOAD USERS
    pending = open("pending_users.txt").read().splitlines()
    approved = open("approved_users.txt").read().splitlines()

    return render_template('dashboard.html',
                           data=data,
                           dates=dates,
                           counts=counts,
                           pending=pending,
                           approved=approved)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')
@app.route('/admin_panel')
def admin_panel():
    if not is_admin():
        return "Access Denied ❌"

    pending = open("pending_users.txt").read().splitlines()
    approved = open("approved_users.txt").read().splitlines()

    return render_template("admin_panel.html",
                           pending=pending,
                           approved=approved)

@app.route('/approve/<name>')
def approve(name):
    if not is_admin():
        return "Access Denied ❌"

    pending = open("pending_users.txt").readlines()
    with open("pending_users.txt", "w") as f:
        for u in pending:
            if u.strip() != name:
                f.write(u)

    with open("approved_users.txt", "a") as f:
        f.write(name + "\n")

    return redirect('/dashboard')


@app.route('/reject/<name>')
def reject(name):
    if not is_admin():
        return "Access Denied ❌"

    pending = open("pending_users.txt").readlines()
    with open("pending_users.txt", "w") as f:
        for u in pending:
            if u.strip() != name:
                f.write(u)

    return redirect('/dashboard')


@app.route('/delete_user/<name>')
def delete_user(name):
    if not is_admin():
        return "Access Denied ❌"

    approved = open("approved_users.txt").readlines()
    with open("approved_users.txt", "w") as f:
        for u in approved:
            if u.strip() != name:
                f.write(u)

    return redirect('/dashboard')

if __name__ == "__main__":
    app.run(debug=True)