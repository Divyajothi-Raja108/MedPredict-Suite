from django.contrib import admin
from django.urls import path
from home import views
urlpatterns = [
    path('', views.index, name='home'),
    path('heart/', views.heart1, name='heart'),
    path('heart/submit',views.result),
    path('diabetes/', views.diabetes, name='diabetes'),
    path('diabetes/submit1', views.result2),
    path('liver/', views.liver, name='liver'),
    path('liver/submit2', views.result3)
]
