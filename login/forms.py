from django import forms
from .models import (
    Login,
    historyofloanstatus
)

class historyofloanstatusForm(forms.ModelForm):
    class Meta:
        model = historyofloanstatus
        fields = [
            "user",
            "loan_amount",
            "term",
            "credit_score",
            "annual_income",
            "years_in_current_job",
            "home_ownership",
            "purpose",
            "monthly_debt",
            "years_of_credit_history",
            "number_of_open_accounts",
            "number_of_credit_problems",
            "current_credit_balance",
            "maximum_open_credit",
            "bankruptcies",
            "tax_liens"
        ]