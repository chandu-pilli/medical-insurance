# 🏥 Health Insurance Cost Prediction System

> **An AI-powered web application that predicts health insurance premiums based on a person's health profile, family details, and medical history.**

---

# 📖 Table of Contents

* About the Project
* Problem Statement
* Solution
* Features
* How It Works
* System Architecture
* Technologies Used
* Machine Learning Model
* Project Workflow
* Folder Structure
* Installation
* Running the Project
* API Endpoints
* Future Improvements
* Team Contribution
* Conclusion

---

# 📌 About the Project

Health insurance companies calculate insurance premiums using many factors such as:

* Age
* BMI (Body Mass Index)
* Medical history
* Smoking habits
* Family size
* Sum insured
* Policy tenure

Doing this manually is slow and sometimes inconsistent.

This project uses **Machine Learning (XGBoost)** to automatically estimate the insurance premium quickly and accurately.

The system also provides a beautiful web interface where users can enter their details and instantly receive an estimated insurance cost.

---

# ❓ Problem Statement

Many people don't know:

* Which insurance plan is suitable for them
* How much premium they should expect
* Which health conditions increase insurance costs

Insurance companies also spend time manually evaluating customer details.

This project solves that problem by providing an AI-based premium prediction system.

---

# ✅ Solution

The application allows users to:

* Enter personal information
* Add family members
* Enter medical conditions
* Select insurance coverage
* Submit details

The Machine Learning model analyzes the information and predicts an estimated insurance premium within seconds.

---

# ✨ Features

### User Friendly Interface

* Modern responsive website
* Mobile friendly
* Dark Mode support
* Step-by-step form

---

### Customer Details

The system collects:

* Name
* Mobile Number
* Email
* Gender
* Pincode
* Location

---

### Family Information

Users can add:

* Self
* Spouse
* Children
* Parents
* In-laws

---

### Health Information

The application considers:

* Age
* BMI
* Smoking status
* Diabetes
* Hypertension
* Heart Disease
* Thyroid
* Asthma

---

### Insurance Details

Users can choose:

* Sum Insured
* Policy Tenure

---

### AI Prediction

The Machine Learning model predicts:

* Estimated Premium
* Suitable Insurance Plans

---

### Database Storage

Every inquiry is stored safely in an SQLite database for future reference.

---

# ⚙️ How It Works

## Step 1

User opens the website.

↓

## Step 2

User enters personal details.

↓

## Step 3

User enters family information.

↓

## Step 4

User enters medical information.

↓

## Step 5

The frontend sends the data to the Flask backend.

↓

## Step 6

The backend prepares the data.

↓

## Step 7

The data is converted into Machine Learning features.

↓

## Step 8

The trained XGBoost model predicts the insurance premium.

↓

## Step 9

The predicted premium is displayed to the user.

↓

## Step 10

The customer details are stored in the database.

---

# 🏗 System Architecture

```
Frontend (HTML, CSS, JavaScript)
           │
           ▼
     Flask Backend API
           │
           ▼
 Feature Engineering Engine
           │
           ▼
     Data Preprocessing
           │
           ▼
   XGBoost Machine Learning Model
           │
           ▼
 Premium Prediction Result
           │
           ▼
     SQLite Database Storage
```

---

# 🧠 Machine Learning Model

The project uses the **XGBoost Regression Algorithm**.

### Why XGBoost?

Because it provides:

* High prediction accuracy
* Fast performance
* Handles complex relationships
* Works well with structured data

---

## Input Features

The model uses information such as:

* Age
* Gender
* BMI
* Smoking Status
* Number of Family Members
* Diabetes
* Hypertension
* Heart Disease
* Thyroid
* Asthma
* Sum Insured
* Policy Tenure
* City Tier
* Average Family Age

The backend also creates additional calculated features such as:

* Age Group
* BMI Category
* Total Medical Conditions
* Risk Indicators

These improve prediction accuracy.

---

# 💻 Technologies Used

## Frontend

* HTML5
* CSS3
* JavaScript

---

## Backend

* Python
* Flask
* Flask-CORS
* SQLAlchemy

---

## Machine Learning

* XGBoost
* Scikit-Learn
* Pandas
* NumPy
* Joblib

---

## Database

* SQLite

---

## Development Tools

* VS Code
* Git
* GitHub

---

# 📂 Project Structure

```
Health-Insurance-Cost-Prediction/

│
├── app.py                  # Flask Backend
├── train_model.py          # ML Model Training
├── index.html              # Frontend
├── index.css               # Styling
├── app.js                  # Frontend Logic
├── requirements.txt
├── models/
│      ├── xgb_model.pkl
│      ├── scaler.pkl
│      ├── feature_cols.json
│      └── metrics.json
│
├── database/
│      └── leads.db
│
└── README.md
```

---

# 🔄 Project Workflow

```
User

↓

Web Application

↓

User enters details

↓

Flask API

↓

Feature Engineering

↓

Data Scaling

↓

XGBoost Model

↓

Premium Prediction

↓

Display Insurance Plans

↓

Save Customer Record
```

---

# 🚀 Installation

## Step 1

Clone the repository

```bash
git clone <repository-url>
```

---

## Step 2

Open the project folder

```bash
cd Health-Insurance-Cost-Prediction
```

---

## Step 3

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4

Train the Machine Learning model

```bash
python train_model.py
```

This generates:

* Trained Model
* Scaler
* Feature List
* Model Metrics

---

## Step 5

Start the Flask server

```bash
python app.py
```

---

## Step 6

Open the application

```
http://localhost:5000
```

---

# 📡 API Endpoints

### Send OTP

```
POST /api/send-otp
```

Generates a verification OTP.

---

### Verify OTP

```
POST /api/verify-otp
```

Verifies the entered OTP.

---

### Predict Premium

```
POST /api/predict
```

Predicts the insurance premium using the ML model.

---

### Health Check

```
GET /api/health
```

Checks whether the backend and ML model are running properly.

---

# 🎯 Benefits of the Project

### For Customers

* Instant premium estimation
* Easy-to-use interface
* Better understanding of insurance costs
* Compare different coverage options

### For Insurance Companies

* Faster customer processing
* Reduced manual work
* More consistent pricing
* Better customer experience

---

# 🔮 Future Improvements

Some ideas for future versions:

* Support multiple insurance companies
* Real-time policy comparison
* PDF quotation generation
* User login and dashboard
* Cloud deployment
* Email and SMS notifications
* Integration with payment gateway
* AI chatbot for customer support
* Real insurance company APIs

---

# 👥 Team Contribution

Example:

| Team Member | Contribution                                            |
| ----------- | ------------------------------------------------------- |
| Team Leader | Project planning, Machine Learning, Backend integration |
| Member 2    | Frontend Development                                    |
| Member 3    | Backend Development                                     |
| Member 4    | Database Design                                         |
| Member 5    | Testing                                                 |
| Member 6    | Documentation                                           |
| Member 7    | Presentation                                            |

---

# 📊 Key Highlights

* AI-powered insurance premium prediction
* User-friendly responsive web application
* Machine Learning using XGBoost
* Flask REST API
* SQLite database integration
* OTP verification
* Family health analysis
* Automatic feature engineering
* Fast and accurate premium estimation

---

# 🎉 Conclusion

The **Health Insurance Cost Prediction System** is an intelligent web application that helps users estimate health insurance premiums based on their personal and medical information. By combining a simple user interface with a powerful Machine Learning model, the system provides fast, reliable, and easy-to-understand premium estimates.

This project demonstrates the practical use of **Artificial Intelligence, Machine Learning, Web Development, and Database Management** to solve a real-world problem. It can serve as a strong academic project and has the potential to be expanded into a production-ready insurance recommendation platform.
