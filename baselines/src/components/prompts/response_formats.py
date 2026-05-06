from pydantic import BaseModel
from typing import Literal

class CounsellorUtterance_t1(BaseModel):
    explanation: str
    label: Literal["CRL", "SRL", "IMC", "IMI", "Q", "O"]

class CounsellorUtterance_t2(BaseModel):
    explanation: str
    label: Literal["CR", "AF", "SU", "RF", "EC",
                   "SR", 
                   "ADP", "RCP", "GI",
                   "ADW", "CO", "DI", "RCW", "WA",
                   "OQ", "CQ",
                   "FA", "FI", "ST"]
    
class CounsellorUtterance_flat(BaseModel):
    explanation: str
    label: Literal["CR", "AF", "SU", "RF", "EC",
                   "SR", 
                   "ADP", "RCP", "GI",
                   "ADW", "CO", "DI", "RCW", "WA",
                   "OQ", "CQ",
                   "FA", "FI", "ST"]
    
class ClientUtterance_t1(BaseModel):
    explanation: str
    label: Literal["C", "S", "N"]

class ClientUtterance_t2(BaseModel):
    explanation: str
    label: Literal["O+", "D+", "AB+", "R+", "N+", "C+", "AC+", "TS+",
                   "O-", "D-", "AB-", "R-", "N-", "C-", "AC-", "TS-",
                   "N"]        

class ClientUtterance_flat(BaseModel):
    explanation: str
    label: Literal["O+", "D+", "AB+", "R+", "N+", "C+", "AC+", "TS+",
                   "O-", "D-", "AB-", "R-", "N-", "C-", "AC-", "TS-",
                   "N"]