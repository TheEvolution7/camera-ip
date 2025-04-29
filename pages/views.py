from django.views.generic import TemplateView

from devices.models import Device


class IndexView(TemplateView):
    template_name = "pages/index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        devices = Device.objects.all()
        context["devices"] = devices
        return context
