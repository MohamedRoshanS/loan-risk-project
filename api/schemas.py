from pydantic import BaseModel, conint, confloat, constr

class Borrower(BaseModel):
    Age: conint(ge=18, le=69)
    Income: confloat(ge=0)
    LoanAmount: confloat(ge=0)
    CreditScore: conint(ge=300, le=850)
    MonthsEmployed: conint(ge=0)
    NumCreditLines: conint(ge=1)
    LoanTerm: conint(ge=1)
    Education: constr()
    EmploymentType: constr()
    MaritalStatus: constr()
    HasMortgage: constr()
    HasDependents: constr()
    LoanPurpose: constr()
    HasCoSigner: constr()
