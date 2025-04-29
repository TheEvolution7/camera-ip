from django.urls import path
from . import views

app_name = "devices"

urlpatterns = [
    path(
        "devices/",
        views.DeviceListView.as_view(),
        name="device_list",
    ),
    path(
        "cameras/",
        views.CameraListView.as_view(),
        name="camera_list",
    ),
    path(
        "cameras/<int:pk>",
        views.CameraDetailView.as_view(),
        name="camera_detail",
    ),
    path(
        "cameras/<int:pk>/update",
        views.CameraUpdateView.as_view(),
        name="camera_update",
    ),
    
]
