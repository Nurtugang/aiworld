# Generated by Django 3.2.18 on 2023-04-09 04:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('App123', '0005_findperson_camera'),
    ]

    operations = [
        migrations.AlterField(
            model_name='findperson',
            name='camera',
            field=models.CharField(blank=True, choices=[('Camera1', 'Camera1'), ('Camera2', 'Camera2'), ('Camera3', 'Camera3')], default='Camera1', max_length=20),
        ),
    ]