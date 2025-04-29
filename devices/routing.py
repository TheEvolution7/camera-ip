from django.urls import path

from . import consumers

websocket_urlpatterns = [
    path('ws/camera-stream/<int:pk>/', consumers.CameraStreamConsumer.as_asgi()),
]