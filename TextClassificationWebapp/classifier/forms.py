# python
from django import forms

class TextForm(forms.Form):
    text = forms.CharField(
        label='Input text',
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 60}),
        max_length=10000
    )