# NexaMatch
### AI-Based Resume and Job Description Matching Platform

NexaMatch is an AI-powered web application that analyzes a candidate's resume against a job description. It uses Natural Language Processing (NLP), semantic similarity, and skill matching techniques to help users understand how closely their resume aligns with a target role.

The application provides similarity scores, identifies matching and missing skills, and generates a downloadable analysis report.

## Problem Statement

Candidates may find it difficult to determine whether their resumes align with the skills and requirements mentioned in job descriptions. Manually comparing resumes with job requirements can be time-consuming and may overlook important skills.

NexaMatch aims to simplify this process by providing an automated resume and job description comparison system.

## Objectives

- Automate resume text extraction from PDF files.
- Compare resume content with job descriptions.
- Calculate semantic and text-based similarity scores.
- Identify matching and missing skills.
- Display the analysis through an interactive web interface.
- Generate a downloadable report of the results.

## Features

- **PDF Resume Upload:** Upload a resume in PDF format for analysis.
- **Job Description Analysis:** Enter the target job description.
- **Semantic Similarity:** Compare the contextual meaning of resume and job description text.
- **TF-IDF Similarity:** Measure text similarity using TF-IDF vectorization and cosine similarity.
- **Skill Matching:** Identify recognized skills that appear in both the resume and job description.
- **Missing Skill Identification:** Highlight recognized job-related skills that were not detected in the resume.
- **Visual Results:** View similarity scores and skill coverage through the web interface.
- **Downloadable Report:** Export the analysis results as a text report.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Web application interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Semantic text representation |
| all-MiniLM-L6-v2 | Pre-trained model for semantic similarity |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| Pandas | Data handling where required |
| NumPy | Numerical operations where required |

## How It Works

1. The user uploads a resume in PDF format.
2. The application extracts text from the resume using PyMuPDF.
3. The user provides a job description.
4. The text is processed for comparison.
5. TF-IDF and cosine similarity are used to calculate text-based similarity.
6. A Sentence Transformers model generates embeddings to calculate semantic similarity.
7. A predefined skill vocabulary and alias-matching rules identify matching and missing skills.
8. The application calculates skill coverage and displays the results.
9. The user can download a report containing the analysis.

## System Workflow

Resume PDF + Job Description  
↓  
Resume Text Extraction  
↓  
Text Processing and Skill Detection  
↓  
TF-IDF Similarity + Semantic Similarity  
↓  
Skill Matching and Coverage Calculation  
↓  
Results Dashboard  
↓  
Downloadable Analysis Report

## Installation and Setup

### Prerequisites

- Python installed on your system
- Visual Studio Code or another code editor
- Internet connection for the initial download of the semantic model, if it is not already cached

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Resume-Job-Matcher
```

Replace the placeholder with your actual GitHub repository URL.

Alternatively, download the project and open its folder in VS Code.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access NexaMatch.

If you use the virtual environment's Python executable directly, you can also run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Output

The application displays:

- Semantic similarity score
- TF-IDF similarity score
- Required skill coverage
- Matching skills
- Missing skills
- Downloadable analysis report

## Limitations

- Skill detection depends on the predefined skill vocabulary and matching rules.
- Results may vary depending on resume formatting, extracted text quality, and job description wording.
- Similarity scores indicate textual or semantic alignment, not the probability of getting hired.
- The system does not replace human judgment in recruitment.
- The current implementation does not provide a validated prediction of candidate suitability.

## Future Scope

- Support for additional resume formats.
- Improved skill extraction using advanced NLP techniques.
- Better recognition of related skills and technical concepts.
- Resume improvement suggestions based on missing job requirements.
- Comparison of multiple job descriptions.
- Candidate ranking for recruiter workflows.
- Database integration and user authentication, if required in future versions.

## Academic Project

**Project Title:** AI-Based Resume and Job Description Matching Platform

**Application Name:** NexaMatch

**Project Domain:** Artificial Intelligence, Natural Language Processing, and Data Science

**Project Type:** MCA Minor Project

---

*Note: NexaMatch is intended to support resume analysis and career preparation. Its scores should not be interpreted as hiring probabilities or definitive measures of candidate quality.*