from django.forms.widgets import CheckboxInput


class CheckboxToggleButtonWidget(CheckboxInput):
    def __init__(self, attrs=None):
        attrs.update(
            {
                "class": "btn-check",
                "onchange": "this.form.submit()",
            }
        )
        super().__init__(attrs)

    # template_name = "devices/forms/widgets/checkbox_toggle_button.html"
