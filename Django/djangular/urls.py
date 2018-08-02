""" https://docs.djangoproject.com/en/1.9/topics/http/urls/
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.views.generic import TemplateView
from django.views.decorators.csrf import ensure_csrf_cookie

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', ensure_csrf_cookie(TemplateView.as_view(template_name="home.html"))),
    url(r'^scrumboard/', include('scrumboard.urls')),
    url(r'^auth_api/', include('auth_api.urls')),
]
