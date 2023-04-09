from django.contrib import admin
from .models import *


class TrackAdmin(admin.ModelAdmin):
    readonly_fields = ('tracked_time',)


admin.site.register(LastFace)
admin.site.register(Profile)
admin.site.register(FindPerson)
admin.site.register(Track, TrackAdmin)
