# main.py

from fastapi import FastAPI, HTTPException
import uvicorn
import os

# Initialize the FastAPI application
app = FastAPI(
    title="Test FastAPI Application",
    description="A simple FastAPI application for CI/CD testing purposes.",
    version="1.0.0"
)

@app.get("/", summary="Root endpoint", response_description="A welcome message")
async def read_root():
    """
    Root endpoint that returns a simple welcome message.
    """
    return {"message": "Welcome to the Test FastAPI Application!"}

@app.get("/health", summary="Health check endpoint", response_description="Status of the application")
async def health_check():
    """
    Health check endpoint to verify the application is running.
    Returns a 200 OK status with a simple message.
    """
    return {"status": "ok", "message": "Application is healthy!"}

@app.get("/info", summary="Application Information", response_description="Returns basic application environment info")
async def get_info():
    """
    Returns some basic information about the application's environment,
    including an example of reading an environment variable.
    """
    # Example of reading an environment variable
    # In a real application, you would load these from a .env file or secrets manager
    example_env_var = os.getenv("TEST_ENV_VAR", "TEST_ENV_VAR not set")
    return {
        "app_name": app.title,
        "app_version": app.version,
        "environment_variable_example": example_env_var,
        "message": "This is a test endpoint to show environment variable access."
    }