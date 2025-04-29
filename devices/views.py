from django.views.generic import ListView
from django.views.generic import DetailView
from django.views.generic import CreateView
from django.views.generic import UpdateView

from devices.forms import CameraForm
from . import models


class DeviceListView(ListView):
    model = models.Device


class CameraListView(ListView):
    model = models.Camera


class CameraDetailView(DetailView):
    model = models.Camera

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["form"] = CameraForm(instance=self.object)
        return context


class CameraUpdateView(UpdateView):
    model = models.Camera
    fields = ("privacy_mode",)

    def post(self, request, *args, **kwargs):
        return super().post(request, *args, **kwargs)

    def form_valid(self, form):
        return super().form_valid(form)
