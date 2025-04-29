from django.contrib import admin
from . import models


@admin.register(models.Device)
class DeviceAdmin(admin.ModelAdmin):
    pass


@admin.register(models.Camera)
class CameraAdmin(admin.ModelAdmin):
    pass
