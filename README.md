# Titanic Dataset Chat Agent

An interactive chatbot that allows users to explore and analyze the
Titanic dataset using natural language. The system combines a FastAPI
backend, a LangChain-based tool-using agent, and a Streamlit frontend to
provide accurate answers and real visual insights.

The goal of this project is to demonstrate a clean, reliable approach to
building an LLM-powered data assistant while ensuring correctness,
transparency, and maintainability.

------------------------------------------------------------------------

# Overview

This application enables users to ask questions such as:

-   What percentage of passengers were male?
-   Show a histogram of passenger ages
-   What was the average ticket fare?
-   How many passengers embarked from each port?

The system responds with:

-   A clear text answer
-   A visualization when relevant
-   Deterministic results computed directly from the dataset

Instead of relying on the language model for calculations, the model
uses structured tools that execute verified pandas operations. This
approach prevents hallucinations and ensures consistent, correct
outputs.

------------------------------------------------------------------------

# Architecture

The project is split into two main components.

## Backend (FastAPI)

Responsible for:

-   Handling chat requests
-   Running the LangChain agent
-   Executing dataset analysis tools
-   Returning text responses and visualization data

## Frontend (Streamlit)

Provides a clean chat interface where users can:

-   Enter questions in plain English
-   View responses
-   See generated charts
-   Interact with the dataset conversationally

------------------------------------------------------------------------

# Technology Stack

Backend: - Python - FastAPI - LangChain - pandas - matplotlib - Ollama
(local LLM runtime)

Frontend: - Streamlit

------------------------------------------------------------------------

# Setup Instructions

## Install Ollama

Download from: https://ollama.com

Pull model: ollama pull mistral

Run: ollama run mistral

## Run Backend

cd backend pip install -r requirements.txt uvicorn main:app --reload

## Run Frontend

cd frontend streamlit run app.py

------------------------------------------------------------------------

# Evaluation

Run:

python evaluation.py

This verifies correctness using predefined test queries.

------------------------------------------------------------------------

# Conclusion

This project demonstrates a reliable and deterministic approach to
building an LLM-powered dataset assistant using structured tools and
clean architecture.
