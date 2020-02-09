from django.urls import path

from . import views

urlpatterns = [
    path('img/', views.upload_file, name='img'),
    path('img/<fname>/', views.upload_file, name='img'),
    path('', views.index, name='index'),
]