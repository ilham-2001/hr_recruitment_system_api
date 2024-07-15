from pydantic import BaseModel

class Authentication(BaseModel):
    email: str
    password: str

class CandidateData(BaseModel):
    self_acknowledgement_weakness_score: int
    communication_skill_score: int
    salary: int
    self_acknowledgement_strength_score: int
    vision_mission_score: int
    stress_management_score: int
    full_name: str
    email: str