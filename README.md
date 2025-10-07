# Healthcare Management System

Hey there! This is a healthcare management system I built to help patients and doctors manage medical records, appointments, and health data more easily. It's got a FastAPI backend for the API stuff and a Streamlit frontend for the user interface.

## What This Does

Basically, it's a system where:
- Patients can upload their medical documents, request appointments, and view their health records
- Doctors can search through patients, review records, schedule appointments, and track health metrics
- Everything is secured with JWT tokens and proper authentication
- File uploads are handled safely with size and type validation
- Health data can be analyzed and trends can be visualized

## What's New?

What's New?

### v2:
- Now added database using MYSQL Workbench 8.0. 
- Prerequisites
- MySQL Server: Make sure you have the MySQL Server installed and running on your system.

- MySQL Workbench: You will need MySQL Workbench to connect to the server and import the database.

##### Steps to Set Up the Database
- Open MySQL Workbench: Launch the application and connect to your local MySQL server instance using the localhost connection.
- Create a New Database:
  - In the Schemas section on the left, right-click and select "Create Schema...".
  - Name the new schema healthcare_management_v2 (or your preferred database name).
  - Click the Apply button to create the database.
  - Import the Database Schema and Data:
  - Go to Server > Data Import.
  - Select "Import from Self-Contained File" and browse to the location of the .sql file in your project. This file contains the schema and data for the project.
  - Choose the newly created database healthcare_management_v2 as the "Target Schema".
  - Click Start Import.
  - Verify the Import:
  - Once the import is complete, refresh the Schemas view in the left sidebar.
  - Expand the healthcare_management_v2 schema to confirm that all tables and data have been imported correctly.

### v1: 
- Added persistent signup and login with GET/POST in the backend, storing users in users.json, saving profile pictures to profile_pics/ and serving them via /profile_pics/....
- Compare passwords for confirmation now. after your Submission if the password fields do not match, you will find your appropriate message for any unsuccessful login!
- Updated the Streamlit UI to include a Signup form with all required fields, password confirmation, and login by email or username. Also fixed the Menu section using internal CSS.
- After login, users are routed to role-based dashboards with a new My Profile section that displays the signup details.
##### What you can do now
- Open the UI at PORT = "http://127.0.0.1:8501"
- Switch to Signup, create either a patient or doctor account, then login
- On login, you’ll see My Profile in the sidebar and can proceed to other pages
###### Key endpoints added
- POST /api/auth/signup: fields first_name, last_name, username, email, password, confirm_password, role, address_line1, city, state, pincode, profile_picture (file)
- POST /api/auth/login: accepts either email+password or username+password
- GET /api/auth/user/{user_id}: returns public user details (auth required)
- Static: /profile_pics/{filename} for profile picture serving

### Blog System (New)
- Doctors can create blog posts under categories: Mental Health, Heart Disease, Covid19, Immunization
- Post fields: Title, Image, Category, Summary, Content, Draft flag
- Doctors can see their own posts
- Patients can browse published posts by category; summaries auto-truncate to 15 words with "..."
- Images are served from `/blog_images/`

New API endpoints:
- `POST /api/blogs` (doctors only): create post (multipart form with optional image)
- `GET /api/blogs?category=...` (all): list posts, patients only see non-drafts
- `GET /api/blogs/mine` (doctors only): list author’s posts

UI additions (Streamlit):
- Doctor: "Write Blog", "My Blog Posts" in dashboard
- Patient: "Blogs" page with category filter and truncated summaries

Database:
- MySQL via SQLAlchemy. Table: `blog_posts`.

## Quick Start

If you just want to get this running locally, here's the fastest way:

```bash
# Set up your environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create uploads folder
mkdir uploads

# Run the API (in one terminal)
uvicorn api:app --reload

# Run the UI (in another terminal)
streamlit run app.py
```

Then open http://localhost:8000/docs for the API docs and http://localhost:8501 for the web interface.

## MySQL Setup for Blogs

Create a MySQL database and user, then set environment variables in `.env`:

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=healthcare
MYSQL_USER=root
MYSQL_PASSWORD=yourpassword
# Optionally, full URL:
# DATABASE_URL=mysql+pymysql://root:yourpassword@localhost:3306/healthcare
```

Install new dependencies and run once to auto-create tables:

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

On startup, the API will create the `blog_posts` table if it doesn’t exist.

## Setting Up Environment Variables

Create a `.env` file in your project root:

```env
SECRET_KEY=your-super-secret-key-here
ALLOWED_ORIGINS=http://localhost:8501
```

The SECRET_KEY is used for JWT token generation, so make it something secure and random.

## Demo Users

I've included some demo accounts so you can test things out right away:

- **Patient**: demo@example.com / demo123
- **Doctor**: doctor@example.com / doctor123

Just log in through the Streamlit interface and you'll be able to explore the system.

## Project Structure

Here's how the code is organized:

```
healthcare-management-system/
│
├── api.py                 # Main FastAPI app for local development
├── api/
│   └── index.py          # Serverless entrypoint for Vercel deployment
├── app.py                 # Streamlit user interface
├── utils.py               # Security utilities, JWT helpers, etc.
├── requirements.txt       # Python package dependencies
├── vercel.json            # Vercel configuration
├── uploads/               # Where uploaded files get stored
└── README.md              # This file
```

## Deploying to Vercel

The API part is ready to deploy on Vercel as a serverless function. Here's how:

1. Push your code to GitHub or GitLab
2. Import the project into Vercel (choose "Other" as the framework)
3. Set these environment variables in Vercel:
   - SECRET_KEY (required)
   - ALLOWED_ORIGINS (optional, for CORS)
   - UPLOAD_DIR (optional, defaults to "uploads")
4. Deploy

Your API will be available at your Vercel URL under `/api/...` routes.

Note: The Streamlit app can't run on Vercel, so you'll need to host that separately (like on Streamlit Community Cloud) and point it to your Vercel API URL.

## Main API Endpoints

Here are the key endpoints you'll probably use:

- `POST /api/auth/login` - Log in and get a token
- `POST /api/records/upload` - Upload a medical document
- `GET /api/records/{patient_id}` - Get records for a patient
- `POST /api/appointments` - Create an appointment request
- `GET /api/appointments` - List appointments
- `GET /api/patients/search` - Search for patients (doctors only)

You can explore all the endpoints interactively at `/docs` when the API is running.

## Common Issues and Solutions

**Getting 401 errors?** Make sure you're sending the Authorization header with your JWT token: `Authorization: Bearer <your-token>`

**CORS problems?** Check that your ALLOWED_ORIGINS includes the domain where your frontend is running

**File uploads failing?** Keep files under 10MB and make sure you're using multipart/form-data

**Can't connect to the API?** Double-check your API_URL setting in the Streamlit app

## Dependencies

The main packages this project uses:

**Backend**: FastAPI, Uvicorn, Python-multipart, PyJWT, Passlib, SQLAlchemy, PyMySQL
**Frontend**: Streamlit, Plotly, Pandas, NumPy
**Security**: bcrypt, cryptography

## Contributing

Feel free to contribute! Just keep your changes focused and include a clear description of what you're doing. Small improvements are always welcome.

## License

This is open source under the MIT License, so you can use it however you want.

## Getting Help

If you run into problems:
- Check the API docs at `/docs` for endpoint details
- Look at the error messages in your terminal
- Make sure all environment variables are set correctly
- Verify that your virtual environment is activated

That's about it! The system is pretty straightforward once you get it running. Let me know if you need help with anything specific. 

## What's New?
Changelog:
- Blog system with MySQL storage, images, categories, draft support
- Streamlit pages for writing and browsing blog posts 
