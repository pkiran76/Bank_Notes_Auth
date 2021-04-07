from pydantic import BaseModel
#2-Create a class which desribes the i/p features of a bank note (taken from the i/p file)
class BankNote(BaseModel):
    variance:float
    skewness:float
    curtosis:float
    entropy:float
