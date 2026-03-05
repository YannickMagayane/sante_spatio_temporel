from django.contrib import admin
from django.urls import path, include

import os
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('analytics.urls')),
    path('home/', RedirectView.as_view(pattern_name='analytics:dashboard', permanent=False)),
    path('favicon.ico', RedirectView.as_view(url='/static/favicon.ico')),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static('/media/', document_root=os.path.join(settings.BASE_DIR, 'media'))
