from django import forms

from devices import widgets
from . import models


class CameraForm(forms.ModelForm):
    take_photo = forms.BooleanField(
        disabled=True,
        widget=widgets.CheckboxToggleButtonWidget(
            attrs={"icon": "bi bi-camera"},
        ),
    )
    manual_recording = forms.BooleanField(
        disabled=True,
        widget=widgets.CheckboxToggleButtonWidget(
            attrs={
                "icon": "bi bi-camera-reels",
            },
        )
    )
    volume = forms.BooleanField(
        disabled=True,
        widget=widgets.CheckboxToggleButtonWidget(
            attrs={
                "icon": "bi bi-shield-lock",
            },
        )
    )
    voice_call = forms.BooleanField(
        disabled=True,
        widget=widgets.CheckboxToggleButtonWidget(
            attrs={
                "icon": "bi bi-volume-up",
            },
        )
    )
    talk = forms.BooleanField(
        disabled=True,
        widget=widgets.CheckboxToggleButtonWidget(
            attrs={
                "icon": "bi bi-telephone",
            },
        )
    )
    pan_tilt = forms.BooleanField(
        disabled=True,
        widget=widgets.CheckboxToggleButtonWidget(
            attrs={
                "icon": "bi bi-mic",
            },
        )
    )
    alarm_off = forms.BooleanField(
        disabled=True,
        widget=widgets.CheckboxToggleButtonWidget(
            attrs={
                "icon": "bi bi-arrows-move",
            },
        )
    )
    privacy_mode = forms.BooleanField(
        widget=widgets.CheckboxToggleButtonWidget(
            attrs={
                "icon": "bi bi-shield-lock",
            },
        )
    )
    tag_on = forms.BooleanField(
        widget=widgets.CheckboxToggleButtonWidget(
            attrs={
                "icon": "bi bi-person-bounding-box",
            },
        )
    )

    class Meta:
        model = models.Camera
        fields = "__all__"
        fields = (
            "take_photo",
            "manual_recording",
            "volume",
            "voice_call",
            "talk",
            "pan_tilt",
            "alarm_off",
            "privacy_mode",
            "tag_on",
        )
