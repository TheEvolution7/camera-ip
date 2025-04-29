from django.db import models
from django.urls import reverse


class Device(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.name}"

    def get_absolute_url(self):
        return self.camera.get_absolute_url()


class Camera(Device):
    device = models.OneToOneField(
        Device,
        on_delete=models.CASCADE,
        parent_link=True,
    )
    rtsp_url = models.CharField(max_length=200)
    privacy_mode = models.BooleanField(
        default=True,
    )
    tag_on = models.BooleanField(
        default=True,
    )

    def __str__(self):
        return f"{self.device}"

    def get_absolute_url(self):
        return reverse("devices:camera_detail", args=(self.pk,))
