# python
from django.http import HttpResponse
from django.shortcuts import render
from .forms import TextForm

# try to import a predict(text) function from your existing ml_pipeline
try:
    from TextClassificationWebapp.ml_pipeline import predict
except Exception:
    predict = None

def index(request):
    result = None
    error = None

    if request.method == 'POST':
        form = TextForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            if predict:
                try:
                    # expect predict to return a string or dict-friendly object
                    result = predict(text)
                except Exception as e:
                    error = f'Prediction error: {e}'
            else:
                error = 'ML pipeline not available. Implement predict(text) in TextClassificationWebapp/ml_pipeline.py'
    else:
        form = TextForm()

    return render(request, 'index.html', {'form': form, 'result': result, 'error': error})