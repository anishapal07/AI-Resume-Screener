from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_score(resume_text, job_desc):
    # This part converts text into numbers that the AI can understand
    content = [resume_text, job_desc]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(content)
    
    # This part compares the two sets of numbers
    similarity = cosine_similarity(matrix)
    return similarity[0][1] * 100

# 1. Load your resume file
with open("testresume.txt", "r") as file:
    my_cv_data = file.read()

# 2. Define the Job you are applying for
job_requirement = "We need a student who knows Python, Data Structures, and Java."

# 3. Get the result
score = calculate_score(my_cv_data, job_requirement)

print("---------------------------------")
print(f"AI Resume Match Score: {score:.2f}%")
print("---------------------------------")