from django.db import models


class Page(models.Model):
    title = models.CharField(
        max_length=200,
    )
    slug = models.SlugField(
        unique=True,
        null=True,
        blank=True,
    )
