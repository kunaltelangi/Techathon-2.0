from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import re  # For email and password validation

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    username = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home Route
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('world'))
    return redirect(url_for('login'))

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        print("User is already logged in. Redirecting to world.html...")
        return redirect(url_for('world'))  # Redirect if logged in

    if request.method == 'POST':
        full_name = request.form['fname']
        email = request.form['email'].strip()
        username = request.form['uname'].strip()
        password = request.form['psw']
        confirm_password = request.form['cpsw']

        print(f"Register Attempt - Full Name: {full_name}, Email: {email}, Username: {username}")

        # ✅ Check if username or email already exists
        existing_user = User.query.filter((User.email == email) | (User.username == username)).first()
        if existing_user:
            print("Username or Email already exists!")
            flash("Username or Email already exists!", "danger")
            return redirect(url_for('register'))

        # ✅ Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            print("Invalid email format!")
            flash("Invalid email format!", "danger")
            return redirect(url_for('register'))

        # ✅ Validate password strength
        if not re.match(r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', password):
            print("Weak password!")
            flash("Password must be at least 8 characters long and include a number and special character!", "danger")
            return redirect(url_for('register'))

        if password != confirm_password:
            print("Passwords do not match!")
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))

        # ✅ Hash password before storing
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        try:
            new_user = User(full_name=full_name, email=email, username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()  # Save user to DB
            print(f"User {username} added to the database!")

            # ✅ Log in the user
            login_user(new_user)
            print(f"User logged in after registration: {current_user.is_authenticated}")

            flash("Account created successfully! You are now logged in.", "success")
            return redirect(url_for('world'))  # Redirect to world.html

        except Exception as e:
            print(f"Error while registering user: {e}")
            flash("There was an error creating your account. Please try again.", "danger")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('trans'))  # Change redirect to 'trans'

    if request.method == 'POST':
        username = request.form['uname'].strip()
        password = request.form['psw']
        remember = True if request.form.get('remember') else False

        user = User.query.filter_by(username=username).first()
        if user is None:
            flash("Invalid username or password!", "danger")
            return redirect(url_for('login'))

        if not bcrypt.check_password_hash(user.password, password):
            flash("Invalid username or password!", "danger")
            return redirect(url_for('login'))

        login_user(user, remember=remember)
        print(f"User logged in: {current_user.is_authenticated}")
        print(f"Redirecting to trans.html")

        flash("Login successful!", "success")
        return redirect(url_for('trans'))  # Change redirect to 'trans'

    return render_template('login.html')

# Trans Route (New Route for trans.html)
@app.route('/trans')
@login_required
def trans():
    return render_template('trans.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
