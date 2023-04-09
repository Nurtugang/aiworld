from django.db import models

st = [('pr', 'pr'), ('abs', 'abs'), ('late', 'late')]
class Profile(models.Model):
    first_name = models.CharField(max_length=70)
    last_name = models.CharField(max_length=70)
    email = models.EmailField()
    st = models.CharField(choices=st,max_length=20,null=True,blank=False,default='abs')
    image = models.ImageField()
    updated = models.DateTimeField(auto_now=True)
    shift = models.TimeField()
    def __str__(self):
        return self.first_name +' '+self.last_name

class LastFace(models.Model):
    last_face = models.CharField(max_length=200)
    date = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.last_face
    
class Track(models.Model):
    profile = models.ForeignKey('Profile', on_delete=models.CASCADE)
    tracked_time = models.DateTimeField(auto_now=True)
    tracked_img = models.ImageField()
    def __str__(self):
        return self.profile.first_name + ' ' + str(self.tracked_time)

class FindPerson(models.Model):
    CHOICES = [('Camera1', 'Camera1'), ('Camera2', 'Camera2'), ('Camera3', 'Camera3')]
    photo = models.ImageField(upload_to='photos/')
    camera = models.CharField(choices=CHOICES,max_length=20,blank=True,default='Camera1')


