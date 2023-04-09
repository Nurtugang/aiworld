from django.urls import path

from .views import *


urlpatterns = [
    path('', index, name = 'index'),
    path('prof/', prof, name='prof'),
    path('find/', find, name='find'),
    path('find/<str:cam>', find, name='find'),
    path('car/', car, name='car'),
    path('car/<str:cam>', car, name='car'),
    path('mainpage/', mainpage, name = 'mainpage'),
    path('mainpage/<str:cam>', mainpage, name = 'mainpage'),
    path('tablo/', tablo, name= 'tablo'),
    path('ajax/', ajax, name= 'ajax'),
    path('scan/',scan,name='scan'),
    path('scan/<str:cam>',scan,name='scan'),
 

    path('register/', register, name='register'),
    path('login/', login, name = 'login'),
    path('login/<str:cam>/', login, name = 'login'),


]
