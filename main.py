from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

import google.generativeai as genai

import mysql.connector

from interface import Authentication, CandidateData

import pandas as pd
import numpy as np
import os
import json
import random
from dotenv import load_dotenv


prompts = {
    "administration": [
        """
    Imagine yourself as a HR recruiter for to recruit new employee in your new company, Rate this CV from the
    range from 1 to 5 based on these criteria: his expericence based on the document, the projects he has done, and
    skill expertise in the field and where he or she used it and average the score and normalize it to the range from 1 to 5. Return the response 
    for the score and description about the resulting score prompt wiht a json format
  """,
        """
    Imagine yourself as a HR recruiter for to recruit new employee in your new company, Analyze this transcript give an overall score from the
    range from 1 to 5 And to make sure the the files is
    a transcript, if other files is received give it a 0. A motivational letter only consist of score and it's corresponding courses.
    Return the response for the score and description about the resulting score prompt wiht a json format
  """,
        """
    Imagine yourself as a HR recruiter for to recruit new employee in your new company, Rate this motivational letter from the
    range from 1 to 5 based on these criteria: his willingnes based on the document, the writing, and
    his clear motivation and average the score and normalize it to the range from 1 to 5. And to make sure the the files is
    a motivational letter, if other files is received give it a 0. A motivational letter only consist of text nothing ekse.
    Return the response for the score and description about the resulting score prompt wiht a json format
  """,
    ],
    "hr": """

  """,
    "tech": """

  """
}

load_dotenv()

genai.configure(api_key=os.getenv("LLM_API_KEY"))
model = genai.GenerativeModel(os.getenv("LLM_MODEL_NAME"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_NAME = os.getenv("DATABASE_NAME")
DB_HOST = os.getenv("DATABASE_HOST")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD")
DB_USER = os.getenv("DATABASE_USER")


mydb = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)

mycursor = mydb.cursor()


@app.get("/")
def handle_root():
    return JSONResponse({"message": "This is root endpoint"})


@app.post("/login")
async def handle_user_login(auth_data: Authentication):
    user = "hr" if "hr" in auth_data.email else "candidates"
    sql = f"SELECT * FROM {user} WHERE email=%s AND password=%s"
    value = (auth_data.email, auth_data.password)
    mycursor.execute(sql, value)

    myresult = mycursor.fetchone()

    if len(myresult) != 0 and user != "hr":
        id, full_name, doa, last_education_level, last_education_name, *_ = myresult

        data = {
            "id": id,
            "name": full_name,
            "date_of_birth": doa.strftime('%d-%m-%Y %H:%M:%S'),
            "last_education_level": last_education_level,
            "last_education_name": last_education_name,
        }

        return JSONResponse(content={"message": "User is found", "data": data, "is_login": 1})

    elif len(myresult) != 0 and user == "hr":
        id, name, *_ = myresult

        data = {
            "id": id,
            "name": name,
        }

        return JSONResponse(content={"message": "User is found", "data": data, "is_login": 1})

    return JSONResponse(content={"message": "No user is found"})


@app.get("/recruitments/result/{hr_id}")
async def read_item(hr_id: int):
    sql = """
SELECT 
    hr_interview.self_acknowledgement_weakness_score,
    hr_interview.communication_skill_score,
    hr_interview.salary,
    hr_interview.self_acknowledgement_strength_score,
    hr_interview.vision_mission_score,
    hr_interview.stress_management_score,
    candidates.full_name,
    candidates.email
FROM 
    hr_interview
JOIN 
    candidates ON hr_interview.candidate_id = candidates.candidate_id
WHERE 
    hr_interview.hr_id = %s;
"""
    value = (hr_id,)
    mycursor.execute(sql, value)

    myresult = mycursor.fetchall()

    # Define the column names
    columns = ["self_acknowledgement_weakness_score", "communication_skill_score", "salary", "self_acknowledgement_strength_score",
               "vision_mission_score", "stress_management_score", "full_name", "email"]

    res = []
    # Convert the result to a dictionary
    for data in myresult:
        result_dict = dict(zip(columns, data))
        res.append(result_dict)

    return JSONResponse(content={"data": res})


@app.get("/result/administration/{candidate_id}")
async def read_item(candidate_id: int):
    sql = """
    SELECT candidates.full_name, candidates.date_of_birth, candidates.last_education_level, candidates.last_education_name, 
    administration.cv_score, administration.transcript_score, administration.motivational_letter_score,
    administration.cv_description, administration.transcript_description, administration.motivational_letter_description
    FROM candidates 
    INNER JOIN administration 
    ON candidates.candidate_id = administration.candidate_id 
    WHERE candidates.candidate_id = %s
"""
    value = (candidate_id,)
    mycursor.execute(sql, value)

    myresult = mycursor.fetchone()

    # Define the column names
    columns = ["full_name", "date_of_birth", "last_education_level", "last_education_name",
               "cv_score", "transcript_score", "motivational_letter_score", "cv_description",
               "transcript_description", "motivational_letter_description"]

    # Convert the result to a dictionary
    result_dict = dict(zip(columns, myresult))
    result_dict = {
        **result_dict, "date_of_birth": result_dict['date_of_birth'].strftime('%d-%m-%Y %H:%M:%S'), }
    return JSONResponse(content={"data": result_dict})


@app.post("/recruitment/administration")
async def handle_recruitment_administration(candidate_id: int = Form(...), files: List[UploadFile] = File(...)):
    res = []
    for index, file in enumerate(files):
        content = await file.read()
        file_path = f"documents/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)

        sample_file = genai.upload_file(
            path=file_path, display_name=file.filename)
        response = model.generate_content(
            [sample_file, prompts["administration"][index]])
        res.append(json.loads(response.text.replace(
            "`", "").strip('json').strip("\n")))

    sql = "INSERT INTO administration (administration_id, candidate_id, cv_score, cv_description, transcript_score, transcript_description, motivational_letter_score, motivational_letter_description) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    value = (random.randint(0, 2**32 - 1), candidate_id,
             res[0]["score"], res[0]["description"], res[1]["score"], res[1]["description"], res[2]["score"], res[2]["description"])
    mycursor.execute(sql, value)
    mydb.commit()

    return JSONResponse(content={"count": len(res), "data": res})


@app.post("/rank_candidates")
def rank_candidates(candidates: List[CandidateData]):
    data = [candidate.dict() for candidate in candidates]
    df = pd.DataFrame(data)

    weights = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])

    columns = [
        'self_acknowledgement_weakness_score',
        'communication_skill_score',
        'salary',
        'self_acknowledgement_strength_score',
        'vision_mission_score',
        'stress_management_score'
    ]

    def normalize_matrix(df):
        norm_df = pd.DataFrame(index=df.index)
        for column in columns:
            if column in ['self_acknowledgement_weakness_score', 'salary']:
                norm_df[column] = df[column].min() / df[column].astype(float)
            else:
                norm_df[column] = df[column].astype(float) / df[column].max()
        return norm_df

    norm_df = normalize_matrix(df)
    weighted_norm_df = norm_df * weights
    df['final_score'] = weighted_norm_df.sum(axis=1)
    df['Rank'] = df['final_score'].rank(ascending=False).astype(int)
    df = df.sort_values(by='Rank')

    result = df[['full_name', 'email', 'final_score', 'Rank']
                ].to_dict(orient='records')
    return {"ranked_candidates": result}
