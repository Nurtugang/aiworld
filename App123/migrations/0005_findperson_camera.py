# Generated by Django 3.2.18 on 2023-04-08 14:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('App123', '0004_remove_findperson_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='findperson',
            name='camera',
            field=models.CharField(blank=True, choices=[('Camera1', 'Camera1'), ('Camera2', 'Camera2')], default='Camera1', max_length=20),
        ),
    ]
