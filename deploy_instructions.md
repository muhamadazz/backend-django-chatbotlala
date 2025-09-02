# GitHub Deployment Instructions

Since Git is not available in the WebContainer environment, you'll need to manually deploy this project to GitHub. Here are the step-by-step instructions:

## Method 1: Download and Upload to GitHub

1. **Download the project files**
   - Download all the project files from the WebContainer
   - Make sure to include all the files and folders

2. **Create a new GitHub repository**
   - Go to [GitHub](https://github.com)
   - Click "New repository"
   - Name it something like "mental-health-lstm-api"
   - Make it public or private as needed
   - Don't initialize with README (we already have one)

3. **Upload files to GitHub**
   - Use GitHub's web interface to upload files
   - Or clone the empty repo locally and copy files there

## Method 2: Use GitHub CLI or Git locally

If you have Git installed on your local machine:

```bash
# Clone your empty GitHub repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Copy all project files to this directory
# (download from WebContainer and copy here)

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Django LSTM mental health prediction API"

# Push to GitHub
git push origin main
```

## Method 3: GitHub Desktop

1. Download GitHub Desktop
2. Create a new repository
3. Copy project files to the repository folder
4. Commit and push using the GUI

## Important Files to Include

Make sure these key files are included:
- `requirements.txt` - Python dependencies
- `Procfile` - For Heroku deployment
- `runtime.txt` - Python version specification
- `manage.py` - Django management script
- All Python files in `mental_health_api/` and `prediction/` directories
- `models/` directory with your trained model files
- `.gitignore` - To exclude unnecessary files
- `README.md` - Documentation

## After Uploading to GitHub

Once your code is on GitHub, you can:

1. **Deploy to Heroku:**
   ```bash
   # Connect to your GitHub repo in Heroku dashboard
   # Or use Heroku CLI:
   heroku create your-app-name
   heroku config:set SECRET_KEY="your-secret-key-here"
   heroku config:set DEBUG=False
   git push heroku main
   ```

2. **Deploy to other platforms:**
   - Railway
   - Render
   - PythonAnywhere
   - DigitalOcean App Platform

## Environment Variables for Production

Set these environment variables in your deployment platform:
- `SECRET_KEY`: A secure Django secret key
- `DEBUG`: Set to `False`
- `ALLOWED_HOSTS`: Your domain name

Your API will be available at:
- `/api/predict/` - Make predictions
- `/api/history/` - View prediction history
- `/api/health/` - Health check