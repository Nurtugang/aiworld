# Generated by Django 3.1.3 on 2023-04-07 09:06

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('App123', '0003_findperson'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='findperson',
            name='name',
        ),
    ]
