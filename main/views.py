from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from .handle import handle_uploaded_file

def index(request):
  form = UploadFileForm()
  return render(request, "main/index.html", {'form': form})

# Imaginary function to handle an uploaded file.
def upload_file(request, fname=''):
  if request.method == 'POST':
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
      fname = handle_uploaded_file(request.FILES['file'])
      return HttpResponseRedirect(f"/img/{fname}")
  return render(request, 'main/img.html', {'fname': fname})