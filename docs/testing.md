
# NexaMatch — Testing Documentation

## 1. Purpose

The purpose of testing is to verify that NexaMatch accepts valid inputs, processes resumes and job descriptions, and presents meaningful analysis results.

## 2. Test Environment

- Operating System: Windows
- Programming Language: Python
- User Interface: Streamlit
- PDF Processing: PyMuPDF
- Similarity Analysis: Scikit-learn and Sentence Transformers
- Input Format: PDF resume and text job description

## 3. Test Cases

| Test ID | Test Scenario | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC01 | Upload a valid PDF resume | Resume text is extracted | To be recorded | Pending |
| TC02 | Enter a valid job description | Job description is accepted | To be recorded | Pending |
| TC03 | Analyze a resume and job description | Similarity scores and skill analysis are displayed | To be recorded | Pending |
| TC04 | Download the analysis report | A report is downloaded | To be recorded | Pending |
| TC05 | Submit an empty job description | The application handles the input appropriately | To be recorded | Pending |
| TC06 | Upload an invalid or unsupported file | The application handles the error appropriately | To be recorded | Pending |

## 4. Testing Notes

Execute each test case using the running application. Record the actual result and mark the status as Pass or Fail based on the observed behavior.

Do not mark a test as passed until it has been performed.

## 5. Limitations

- Skill recognition depends on the configured skill vocabulary and matching rules.
- Similarity scores are not hiring probabilities.
- Results may vary depending on the quality and formatting of the resume text.
