#import eventlet
#eventlet.monkey_patch()

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from flask_socketio import SocketIO, emit, join_room
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import threading
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import PIL.Image

# =============================================================================
# APP CONFIGURATION
# =============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'XsomeComplaintSystem_SuperSecret_2026!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///complaints_v3.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx'}

# Flask-Mail (Gmail SMTP)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'fixitbrotech@gmail.com'
app.config['MAIL_PASSWORD'] = 'gndcjhxumnnbmqon'
app.config['MAIL_DEFAULT_SENDER'] = 'fixitbrotech@gmail.com'

# Gemini AI
app.config['GEMINI_API_KEY'] = 'AIzaSyAvLGZxSbTvUv7SjuTwhZ67lBqpr5KHa00'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
mail = Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# =============================================================================
# DATABASE MODELS
# =============================================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='User')  # 'User', 'Officer', 'Admin'
    department = db.Column(db.String(50), nullable=True)

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    priority = db.Column(db.String(20), default='Low')
    status = db.Column(db.String(20), default='Pending')
    location = db.Column(db.String(200))
    image_file = db.Column(db.String(200), default=None)
    date_posted = db.Column(db.DateTime, default=datetime.utcnow)
    deadline = db.Column(db.DateTime)
    feedback = db.Column(db.Text, default=None)
    rating = db.Column(db.Integer, default=None)
    zone = db.Column(db.String(50), default='Central Zone')
    is_escalated = db.Column(db.Boolean, default=False)
    resolved_date = db.Column(db.DateTime, default=None)
    ai_vision_status = db.Column(db.String(300), default='No Image')
    is_duplicate = db.Column(db.Boolean, default=False)
    duplicate_of = db.Column(db.Integer, db.ForeignKey('complaint.id'), nullable=True)
    notification_sent = db.Column(db.Boolean, default=False)

    user = db.relationship('User', foreign_keys=[user_id], backref='complaints')

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    target_role = db.Column(db.String(20))   # 'Admin', 'Officer', 'User'
    target_dept = db.Column(db.String(50), nullable=True)
    target_user_id = db.Column(db.Integer, nullable=True)
    complaint_id = db.Column(db.Integer, db.ForeignKey('complaint.id'))
    message = db.Column(db.String(500))
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notif_type = db.Column(db.String(30), default='new_complaint')  # new_complaint, escalation, status_update


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# =============================================================================
# AI & HELPER FUNCTIONS
# =============================================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_category(text):
    def local_fallback(t):
        t = t.lower()
        if any(x in t for x in ['light', 'power', 'electricity', 'voltage', 'wire', 'current', 'electric', 'outage']):
            return "Electricity"
        if any(x in t for x in ['water', 'leak', 'pipe', 'drain', 'supply', 'dirty water', 'flood', 'sewage']):
            return "Water"
        if any(x in t for x in ['road', 'pothole', 'street', 'traffic', 'bridge', 'pavement', 'footpath', 'crack']):
            return "Road"
        if any(x in t for x in ['garbage', 'trash', 'dustbin', 'waste', 'smell', 'clean', 'sanitation', 'dump']):
            return "Sanitation"
        return "Other"

    key = app.config.get('GEMINI_API_KEY')
    if not key:
        return local_fallback(text)
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = (
            "Classify this city complaint into exactly one of these categories: "
            "Electricity, Water, Road, Sanitation, Other. "
            "Return ONLY the category name, nothing else.\n"
            f"Complaint: {text}"
        )
        response = model.generate_content(prompt)
        category = response.text.strip()
        for v in ["Electricity", "Water", "Road", "Sanitation", "Other"]:
            if v.lower() in category.lower():
                return v
        return local_fallback(text)
    except Exception:
        return local_fallback(text)

def analyze_image_with_ai(image_path, description):
    key = app.config.get('GEMINI_API_KEY')
    if not key or not os.path.exists(image_path):
        return "Image verification skipped."
    try:
        genai.configure(api_key=key)
        img = PIL.Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = (
            f"As a city inspector, look at this image and tell me if it matches this complaint: '{description}'. "
            "Answer with 'Verified' or 'Potential Mismatch' and a brief 5-word reason."
        )
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return "AI Vision: Technical Issue"

def check_for_duplicates(new_description, location):
    existing = Complaint.query.filter_by(location=location).all()
    if not existing:
        return None
    descriptions = [c.description for c in existing] + [new_description]
    try:
        vectorizer = TfidfVectorizer().fit_transform(descriptions)
        vectors = vectorizer.toarray()
        new_vec = vectors[-1].reshape(1, -1)
        for i in range(len(vectors) - 1):
            sim = cosine_similarity(new_vec, vectors[i].reshape(1, -1))[0][0]
            if sim > 0.70:
                return existing[i]
    except Exception:
        pass
    return None

def determine_priority(category, description):
    desc_lower = description.lower()
    HIGH_KEYWORDS = ['leak', 'power failure', 'fire', 'danger', 'flood', 'explosion', 'accident', 'collapse', 'urgent', 'emergency']
    MEDIUM_KEYWORDS = ['damage', 'broken', 'pothole', 'no supply', 'irregular', 'intermittent', 'overflowing']
    if any(kw in desc_lower for kw in HIGH_KEYWORDS):
        return 'High'
    if any(kw in desc_lower for kw in MEDIUM_KEYWORDS) or category == 'Road':
        return 'Medium'
    return 'Low'

def calculate_deadline(priority):
    now = datetime.utcnow()
    if priority == 'High':
        return now + timedelta(hours=24)
    elif priority == 'Medium':
        return now + timedelta(hours=48)
    return now + timedelta(hours=72)

def determine_zone(location):
    loc_lower = location.lower()
    if any(x in loc_lower for x in ['north', 'anna', 'vyasarpadi', 'tondiarpet']):
        return 'North Zone'
    elif any(x in loc_lower for x in ['south', 'adyar', 'guindy', 'velachery', 'tambaram']):
        return 'South Zone'
    elif any(x in loc_lower for x in ['west', 'ambattur', 'anna nagar', 'koyambedu']):
        return 'West Zone'
    elif any(x in loc_lower for x in ['east', 'mylapore', 'beach', 'triplicane']):
        return 'East Zone'
    return 'Central Zone'

def get_priority_color(priority):
    return {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e'}.get(priority, '#94a3b8')

def send_complaint_email(complaint, user):
    """Send email notification to user and department."""
    try:
        subject = f"[Xsome] Complaint #{complaint.id} Received – {complaint.priority} Priority"
        body = f"""
Dear {user.username},

Your complaint has been successfully received and processed by our AI system.

📋 Complaint Details:
━━━━━━━━━━━━━━━━━━━━
Complaint ID  : #{complaint.id}
Title         : {complaint.title}
Category      : {complaint.category}
Priority      : {complaint.priority} (SLA: {'24 hrs' if complaint.priority == 'High' else '48 hrs' if complaint.priority == 'Medium' else '72 hrs'})
Zone          : {complaint.zone}
Location      : {complaint.location}
Deadline      : {complaint.deadline.strftime('%Y-%m-%d %H:%M')} UTC

Our team will address your complaint within the SLA timeframe. 
You will be notified when the status changes.

Thank you for using Xsome Complaint System.
        """
        msg = Message(subject, recipients=[user.email], body=body)
        mail.send(msg)
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

def send_escalation_email(complaint):
    """Send SLA escalation alert to admin."""
    try:
        admins = User.query.filter_by(role='Admin').all()
        for admin in admins:
            subject = f"🚨 [SLA BREACH] Complaint #{complaint.id} Escalated!"
            body = f"""
ADMIN ALERT – SLA Violation

Complaint #{complaint.id} "{complaint.title}" has breached its SLA deadline.

Category  : {complaint.category}
Priority  : {complaint.priority}
Deadline  : {complaint.deadline.strftime('%Y-%m-%d %H:%M')} UTC
Status    : {complaint.status}

Immediate action required. Please review and reassign if necessary.
            """
            msg = Message(subject, recipients=[admin.email], body=body)
            mail.send(msg)
    except Exception as e:
        print(f"[ESCALATION EMAIL ERROR] {e}")


# =============================================================================
# SLA MONITORING BACKGROUND THREAD
# =============================================================================
def sla_monitor_thread():
    """Background thread: checks for SLA violations every 60 seconds."""
    time.sleep(5)
    while True:
        try:
            with app.app_context():
                now = datetime.utcnow()
                violations = Complaint.query.filter(
                    Complaint.status != 'Resolved',
                    Complaint.deadline < now,
                    Complaint.is_escalated == False
                ).all()
                for c in violations:
                    c.is_escalated = True
                    db.session.commit()

                    # Create escalation notification
                    notif = Notification(
                        target_role='Admin',
                        complaint_id=c.id,
                        message=f"🚨 SLA BREACH: Complaint #{c.id} '{c.title}' ({c.category}) has exceeded its {c.priority} priority deadline!",
                        notif_type='escalation'
                    )
                    dept_notif = Notification(
                        target_role='Officer',
                        target_dept=c.category,
                        complaint_id=c.id,
                        message=f"⚠️ OVERDUE: Complaint #{c.id} '{c.title}' deadline has passed. Please resolve immediately!",
                        notif_type='escalation'
                    )
                    db.session.add(notif)
                    db.session.add(dept_notif)
                    db.session.commit()

                    # Broadcast via Socket.IO
                    payload = {
                        'complaint_id': c.id,
                        'title': c.title,
                        'category': c.category,
                        'priority': c.priority,
                        'priority_color': get_priority_color(c.priority),
                        'message': f"SLA Breach: #{c.id} – {c.title}",
                        'type': 'escalation'
                    }
                    socketio.emit('escalation_alert', payload, room='admin')
                    socketio.emit('escalation_alert', payload, room=f'dept_{c.category}')
                    print(f"[SLA MONITOR] Escalated complaint #{c.id}")

                    # Send escalation email in background
                    threading.Thread(target=send_escalation_email, args=(c,), daemon=True).start()

        except Exception as e:
            print(f"[SLA MONITOR ERROR] {e}")
        time.sleep(60)  # Check every minute


# =============================================================================
# SOCKET.IO EVENTS
# =============================================================================
@socketio.on('connect')
def on_connect():
    print(f"[SOCKET] Client connected: {request.sid}")

@socketio.on('join')
def on_join(data):
    room = data.get('room', 'general')
    join_room(room)
    print(f"[SOCKET] Joined room: {room}")

@socketio.on('disconnect')
def on_disconnect():
    print(f"[SOCKET] Client disconnected: {request.sid}")


# =============================================================================
# AUTHENTICATION ROUTES
# =============================================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        if User.query.filter((User.email == email) | (User.phone == phone)).first():
            flash('Email or phone number already registered.', 'danger')
            return redirect(url_for('register'))
        hashed_pw = generate_password_hash(password, method='scrypt')
        new_user = User(username=username, email=email, phone=phone, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'Admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'Officer':
        return redirect(url_for('officer_dashboard'))
    return redirect(url_for('user_dashboard'))


# =============================================================================
# USER DASHBOARD & COMPLAINT SUBMISSION
# =============================================================================
@app.route('/user_dashboard')
@login_required
def user_dashboard():
    complaints = Complaint.query.filter_by(user_id=current_user.id).order_by(Complaint.date_posted.desc()).all()
    notifications = Notification.query.filter_by(
        target_role='User', target_user_id=current_user.id
    ).order_by(Notification.created_at.desc()).limit(10).all()
    return render_template('user_dashboard.html', complaints=complaints, notifications=notifications)

@app.route('/submit_complaint', methods=['GET', 'POST'])
@login_required
def submit_complaint():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        location = request.form.get('location', '').strip()

        # 1. Image Upload
        filename = None
        full_path = None
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(full_path)

        # 2. Duplicate Detection (cosine similarity)
        duplicate_ref = check_for_duplicates(description, location)
        is_duplicate = duplicate_ref is not None
        duplicate_id = duplicate_ref.id if duplicate_ref else None

        # 3. AI Category Classification (NLP)
        category = request.form.get('category') or predict_category(f"{title} {description}")

        # 4. Priority + SLA Deadline
        priority = determine_priority(category, description)
        deadline = calculate_deadline(priority)

        # 5. Zone Assignment
        zone = determine_zone(location)

        # 6. AI Vision Verification (Multimodal)
        ai_vision = "No Image"
        if filename and full_path:
            ai_vision = analyze_image_with_ai(full_path, description)

        new_complaint = Complaint(
            user_id=current_user.id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            deadline=deadline,
            location=location,
            zone=zone,
            image_file=filename,
            ai_vision_status=ai_vision,
            is_duplicate=is_duplicate,
            duplicate_of=duplicate_id
        )
        db.session.add(new_complaint)
        db.session.commit()

        # 7. Create Notifications
        admin_notif = Notification(
            target_role='Admin',
            complaint_id=new_complaint.id,
            message=f"New {priority} priority complaint: #{new_complaint.id} – {title} ({category})",
            notif_type='new_complaint'
        )
        dept_notif = Notification(
            target_role='Officer',
            target_dept=category,
            complaint_id=new_complaint.id,
            message=f"New complaint assigned to {category} dept: #{new_complaint.id} – {title}",
            notif_type='new_complaint'
        )
        user_notif = Notification(
            target_role='User',
            target_user_id=current_user.id,
            complaint_id=new_complaint.id,
            message=f"Your complaint #{new_complaint.id} '{title}' was received. Priority: {priority}. SLA Deadline set.",
            notif_type='new_complaint'
        )
        db.session.add_all([admin_notif, dept_notif, user_notif])
        db.session.commit()

        # 8. Socket.IO Real-Time Broadcast
        sla_hours = {'High': 24, 'Medium': 48, 'Low': 72}.get(priority, 72)
        image_url = url_for('static', filename=f'uploads/{filename}') if filename else None
        payload = {
            'complaint_id': new_complaint.id,
            'title': title,
            'description': description[:120] + '...' if len(description) > 120 else description,
            'category': category,
            'priority': priority,
            'priority_color': get_priority_color(priority),
            'location': location,
            'zone': zone,
            'deadline': deadline.isoformat(),
            'deadline_hours': sla_hours,
            'image_url': image_url,
            'is_duplicate': is_duplicate,
            'submitter': current_user.username,
            'type': 'new_complaint'
        }
        socketio.emit('new_complaint', payload, room='admin')
        socketio.emit('new_complaint', payload, room=f'dept_{category}')
        socketio.emit('complaint_update', {'complaint_id': new_complaint.id, 'status': 'Pending'}, room=f'user_{current_user.id}')

        # 9. Send Email Notification (async)
        threading.Thread(target=send_complaint_email, args=(new_complaint, current_user), daemon=True).start()

        success_msg = f'Complaint submitted! Category: {category} | Priority: {priority} | SLA: {sla_hours}h'
        if is_duplicate:
            success_msg += f' | ⚠ Possible duplicate of #{duplicate_id}'
        flash(success_msg, 'success')
        return redirect(url_for('user_dashboard'))

    return render_template('submit_complaint.html')


# =============================================================================
# OFFICER DASHBOARD
# =============================================================================
@app.route('/officer_dashboard')
@login_required
def officer_dashboard():
    if current_user.role != 'Officer':
        flash('Access Denied', 'danger')
        return redirect(url_for('dashboard'))
    complaints = Complaint.query.filter_by(category=current_user.department).order_by(
        Complaint.is_escalated.desc(), Complaint.priority.desc(), Complaint.date_posted.desc()
    ).all()
    notifications = Notification.query.filter_by(
        target_role='Officer', target_dept=current_user.department
    ).order_by(Notification.created_at.desc()).limit(15).all()
    now = datetime.utcnow()
    for c in complaints:
        if c.status != 'Resolved' and c.deadline and now > c.deadline and not c.is_escalated:
            c.is_escalated = True
    db.session.commit()
    return render_template('officer_dashboard.html', complaints=complaints, notifications=notifications, now=now)

@app.route('/update_status/<int:complaint_id>', methods=['POST'])
@login_required
def update_status(complaint_id):
    if current_user.role != 'Officer':
        return redirect(url_for('dashboard'))
    complaint = db.session.get(Complaint, complaint_id)
    if not complaint:
        flash('Complaint not found.', 'danger')
        return redirect(url_for('officer_dashboard'))
    new_status = request.form.get('status')
    if new_status in ['Pending', 'In Progress', 'Resolved']:
        old_status = complaint.status
        complaint.status = new_status
        if new_status == 'Resolved':
            complaint.resolved_date = datetime.utcnow()
        db.session.commit()

        # Create notification for user
        user_notif = Notification(
            target_role='User',
            target_user_id=complaint.user_id,
            complaint_id=complaint.id,
            message=f"Your complaint #{complaint.id} '{complaint.title}' status updated: {old_status} → {new_status}",
            notif_type='status_update'
        )
        db.session.add(user_notif)
        db.session.commit()

        # Broadcast real-time status update
        payload = {
            'complaint_id': complaint.id,
            'title': complaint.title,
            'old_status': old_status,
            'new_status': new_status,
            'category': complaint.category,
            'updated_by': current_user.username,
            'type': 'status_update'
        }
        socketio.emit('status_update', payload, room='admin')
        socketio.emit('status_update', payload, room=f'user_{complaint.user_id}')
        socketio.emit('status_update', payload, room=f'dept_{complaint.category}')

        flash(f'Complaint #{complaint.id} status updated to "{new_status}"', 'success')
    return redirect(url_for('officer_dashboard'))

@app.route('/mark_read/<int:notif_id>', methods=['POST'])
@login_required
def mark_read(notif_id):
    notif = db.session.get(Notification, notif_id)
    if notif:
        notif.is_read = True
        db.session.commit()
    return jsonify({'ok': True})

@app.route('/mark_all_read', methods=['POST'])
@login_required
def mark_all_read():
    if current_user.role == 'Admin':
        Notification.query.filter_by(target_role='Admin', is_read=False).update({'is_read': True})
    elif current_user.role == 'Officer':
        Notification.query.filter_by(target_role='Officer', target_dept=current_user.department, is_read=False).update({'is_read': True})
    else:
        Notification.query.filter_by(target_role='User', target_user_id=current_user.id, is_read=False).update({'is_read': True})
    db.session.commit()
    return jsonify({'ok': True})


# =============================================================================
# ADMIN DASHBOARD
# =============================================================================
@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'Admin':
        flash('Access Denied', 'danger')
        return redirect(url_for('dashboard'))

    complaints = Complaint.query.order_by(Complaint.date_posted.desc()).all()
    now = datetime.utcnow()
    sla_violations = 0
    for c in complaints:
        if c.status != 'Resolved' and c.deadline and now > c.deadline:
            if not c.is_escalated:
                c.is_escalated = True
            sla_violations += 1
    db.session.commit()

    categories = ['Electricity', 'Water', 'Road', 'Sanitation', 'Other']
    cat_counts = [Complaint.query.filter_by(category=cat).count() for cat in categories]
    status_labels = ['Pending', 'In Progress', 'Resolved']
    status_counts = [Complaint.query.filter_by(status=s).count() for s in status_labels]
    priority_counts = [Complaint.query.filter_by(priority=p).count() for p in ['High', 'Medium', 'Low']]

    total = len(complaints)
    resolved = status_counts[2]
    resolution_rate = round((resolved / total * 100), 1) if total > 0 else 0
    breach_rate = round((sla_violations / total * 100), 1) if total > 0 else 0

    notifications = Notification.query.filter_by(
        target_role='Admin'
    ).order_by(Notification.created_at.desc()).limit(20).all()
    unread_count = Notification.query.filter_by(target_role='Admin', is_read=False).count()

    return render_template('admin_dashboard.html',
        complaints=complaints,
        sla_violations=sla_violations,
        cat_counts=cat_counts,
        categories=categories,
        status_counts=status_counts,
        status_labels=status_labels,
        priority_counts=priority_counts,
        resolution_rate=resolution_rate,
        breach_rate=breach_rate,
        notifications=notifications,
        unread_count=unread_count,
        now=now
    )


# =============================================================================
# API ENDPOINTS FOR REAL-TIME DATA
# =============================================================================
@app.route('/api/stats')
@login_required
def api_stats():
    complaints = Complaint.query.all()
    total = len(complaints)
    resolved = Complaint.query.filter_by(status='Resolved').count()
    pending = Complaint.query.filter_by(status='Pending').count()
    in_progress = Complaint.query.filter_by(status='In Progress').count()
    now = datetime.utcnow()
    breached = Complaint.query.filter(
        Complaint.status != 'Resolved',
        Complaint.deadline < now
    ).count()
    return jsonify({
        'total': total,
        'resolved': resolved,
        'pending': pending,
        'in_progress': in_progress,
        'breached': breached,
        'resolution_rate': round(resolved / total * 100, 1) if total > 0 else 0
    })

@app.route('/api/notifications')
@login_required
def api_notifications():
    if current_user.role == 'Admin':
        notifs = Notification.query.filter_by(target_role='Admin').order_by(Notification.created_at.desc()).limit(20).all()
    elif current_user.role == 'Officer':
        notifs = Notification.query.filter_by(target_role='Officer', target_dept=current_user.department).order_by(Notification.created_at.desc()).limit(15).all()
    else:
        notifs = Notification.query.filter_by(target_role='User', target_user_id=current_user.id).order_by(Notification.created_at.desc()).limit(10).all()
    return jsonify([{
        'id': n.id,
        'message': n.message,
        'type': n.notif_type,
        'is_read': n.is_read,
        'created_at': n.created_at.isoformat(),
        'complaint_id': n.complaint_id
    } for n in notifs])

@app.route('/api/complaint/<int:cid>')
@login_required
def api_complaint(cid):
    c = db.session.get(Complaint, cid)
    if not c:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({
        'id': c.id,
        'title': c.title,
        'description': c.description,
        'category': c.category,
        'priority': c.priority,
        'status': c.status,
        'location': c.location,
        'zone': c.zone,
        'deadline': c.deadline.isoformat() if c.deadline else None,
        'date_posted': c.date_posted.isoformat(),
        'is_escalated': c.is_escalated,
        'is_duplicate': c.is_duplicate,
        'image_file': c.image_file,
        'ai_vision_status': c.ai_vision_status,
        'priority_color': get_priority_color(c.priority)
    })

# Feedback route
@app.route('/submit_feedback/<int:complaint_id>', methods=['GET', 'POST'])
@login_required
def submit_feedback(complaint_id):
    complaint = db.session.get(Complaint, complaint_id)
    if not complaint or complaint.user_id != current_user.id:
        flash('Permission denied.', 'danger')
        return redirect(url_for('user_dashboard'))
    if request.method == 'POST':
        complaint.feedback = request.form.get('feedback')
        complaint.rating = int(request.form.get('rating', 0))
        db.session.commit()
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('user_dashboard'))
    return render_template('feedback.html', complaint=complaint)

# Setup demo route
@app.route('/setup_demo')
def setup_demo():
    if not User.query.filter_by(email='admin@fixit.com').first():
        admin = User(
            username='Admin',
            email='admin@fixit.com',
            phone='0000000000',
            password=generate_password_hash('admin123', method='scrypt'),
            role='Admin'
        )
        db.session.add(admin)
    departments = ['Electricity', 'Water', 'Road', 'Sanitation', 'Other']
    for idx, dept in enumerate(departments):
        email = f'officer_{dept.lower()}@fixit.com'
        if not User.query.filter_by(email=email).first():
            officer = User(
                username=f'{dept} Officer',
                email=email,
                phone=f'111111111{idx}',
                password=generate_password_hash('officer123', method='scrypt'),
                role='Officer',
                department=dept
            )
            db.session.add(officer)
    db.session.commit()
    return (
        "<h2>✅ Setup Complete!</h2>"
        "<p><b>Admin:</b> admin@fixit.com / admin123</p>"
        "<p><b>Officers:</b> officer_electricity@fixit.com, officer_water@fixit.com, "
        "officer_road@fixit.com, officer_sanitation@fixit.com, officer_other@fixit.com (all: officer123)</p>"
        "<br><a href='/login'>👉 Go to Login</a>"
    )


# =============================================================================
# STARTUP
# =============================================================================
with app.app_context():
    db.create_all()

if __name__ == '__main__':

    # Start SLA monitoring background thread
    monitor = threading.Thread(target=sla_monitor_thread, daemon=True)
    monitor.start()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
