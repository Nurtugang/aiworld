from datetime import datetime
import os
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from .models import *
from .forms import *
import face_recognition
import cv2
import numpy as np
from .byte_tracker import BYTETracker, STrack 
# from yolox.tracker.byte_tracker import BYTETracker, STrack 
from onemetric.cv.utils.iou import box_iou_batch 
from dataclasses import dataclass 
from supervision.detection.core import Detections, BoxAnnotator 
from ultralytics import YOLO 
from typing import List 
import threading
from roboflow import Roboflow

rf = Roboflow(api_key="elymibnurzE5l4neBJqy")
project = rf.workspace().project("kznumberplate")
model1 = project.version(1).model

@dataclass(frozen=True) 
class BYTETrackerArgs: 
    track_thresh: float = 0.25 
    track_buffer: int = 30 
    match_thresh: float = 0.8 
    aspect_ratio_thresh: float = 3.0 
    min_box_area: float = 1.0 
    mot20: bool = False 

def detections2boxes(detections: Detections) -> np.ndarray: 
    return np.hstack(( 
        detections.xyxy, 
        detections.confidence[:, np.newaxis] 
    )) 
 

def tracks2boxes(tracks: List[STrack]) -> np.ndarray: 
    return np.array([ 
        track.tlbr 
        for track 
        in tracks 
    ], dtype=float) 
 

def match_detections_with_tracks( 
    detections: Detections,  
    tracks: List[STrack] 
) -> Detections: 
    if not np.any(detections.xyxy) or len(tracks) == 0: 
        return np.empty((0,)) 
 
    tracks_boxes = tracks2boxes(tracks=tracks) 
    iou = box_iou_batch(tracks_boxes, detections.xyxy) 
    track2detection = np.argmax(iou, axis=1) 
     
    tracker_ids = [None] * len(detections) 
    for tracker_index, detection_index in enumerate(track2detection): 
        if iou[tracker_index, detection_index] != 0: 
            tracker_ids[detection_index] = tracks[tracker_index].track_id
 
    return tracker_ids 


class FreshestFrame(threading.Thread):
	def __init__(self, capture, name='FreshestFrame'):
		self.capture = capture
		assert self.capture.isOpened()

		# this lets the read() method block until there's a new frame
		self.cond = threading.Condition()

		# this allows us to stop the thread gracefully
		self.running = False

		# keeping the newest frame around
		self.frame = None

		# passing a sequence number allows read() to NOT block
		# if the currently available one is exactly the one you ask for
		self.latestnum = 0

		# this is just for demo purposes		
		self.callback = None
		
		super().__init__(name=name)
		self.start()

	def start(self):
		self.running = True
		super().start()

	def release(self, timeout=None):
		self.running = False
		self.join(timeout=timeout)
		self.capture.release()

	def run(self):
		counter = 0
		while self.running:
			# block for fresh frame
			(rv, img) = self.capture.read()
			assert rv
			counter += 1

			# publish the frame
			with self.cond: # lock the condition for this operation
				self.frame = img if rv else None
				self.latestnum = counter
				self.cond.notify_all()

			if self.callback:
				self.callback(img)

	def read(self, wait=True, seqnumber=None, timeout=None):
		# with no arguments (wait=True), it always blocks for a fresh frame
		# with wait=False it returns the current frame immediately (polling)
		# with a seqnumber, it blocks until that frame is available (or no wait at all)
		# with timeout argument, may return an earlier frame;
		#   may even be (0,None) if nothing received yet

		with self.cond:
			if wait:
				if seqnumber is None:
					seqnumber = self.latestnum+1
				if seqnumber < 1:
					seqnumber = 1
				
				rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
				if not rv:
					return (self.latestnum, self.frame)

			return (self.latestnum, self.frame)


def index(request):
    if request.method == 'POST':
        form = FindPersonForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            print(form.cleaned_data)
            cam = form.cleaned_data['camera']
            return redirect('find', cam=cam)
    else:
        form = FindPersonForm()
    return render(request, 'index.html', {'form': form})


def prof(request):
    context={
        'profiles': Profile.objects.all().order_by('-id')
    }
    return render(request, 'auth_system/prof.html', context)


def car(request,cam):
    from supervision.detection.line_counter import LineZone, LineZoneAnnotator
    from supervision.geometry.core import Point 
    MODEL = "yolov8x.pt" 
    model = YOLO(MODEL) 
    class_id = 2
    model.predict(source="0", show=False, stream=True, classes=class_id)  
    model.fuse()
    CLASS_NAMES_DICT = model.model.names 
    print('mytca', cam)
    if cam == 'Camera1':
        RTSP_URL = 'rtsp://admin:admin123@192.168.28.2:554/cam/realmonitor?channel=1&subtype=1'
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    elif cam == 'Camera2':
        RTSP_URL = 'rtsp://admin:admin12345@192.168.28.220:554/cam/realmonitor?channel=1&subtype=1' 
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    elif cam == 'Camera3':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fresh = FreshestFrame(cap)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    width = int(width)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = int(height)
    LINE_START = Point(0, height-200) 
    LINE_END = Point(width, height-200) 
    byte_tracker = BYTETracker(BYTETrackerArgs())
    line_counter = LineZone(start=LINE_START, end=LINE_END) 
    line_annotator = LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.5) 
    box_annotator = BoxAnnotator(thickness=1, text_scale=0.5, text_thickness=1)
    cnt = 0
    while(True):
        cnt,frame = fresh.read(seqnumber=cnt+1)
        results = model(frame)  
        detections = Detections( 
            xyxy=results[0].boxes.xyxy.cpu().numpy(), 
            confidence=results[0].boxes.conf.cpu().numpy(), 
            class_id=results[0].boxes.cls.cpu().numpy().astype(int) 
        )    
        detections = detections[detections.class_id==class_id]
        print("Количество автомобилей:", detections.__len__())
        tracks = byte_tracker.update( 
            output_results=detections2boxes(detections=detections), 
            img_info=frame.shape, 
            img_size=frame.shape 
        ) 
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks) 
        detections.tracker_id = np.array(tracker_id) 
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool) 
        detections.filter(mask=mask, inplace=True) 
        labels = [ 
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" 
            for _, confidence, class_id, tracker_id 
            in detections 
        ]  
        line_counter.trigger(detections=detections) 
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels) 
        line_annotator.annotate(frame=frame, line_counter=line_counter) 
        for i in detections.xyxy:
            x1 = int(i[0])
            x2 = int(i[2])
            y1 = int(i[1])
            y2 = int(i[3])       
            cropped = frame[y1:y2,x1:x2]
            scale_percent = 2.20 # percent of original size
            newwidth = int(cropped.shape[0] * scale_percent )
            newheight = int(cropped.shape[1] * scale_percent )
            dim = (newheight, newwidth)
            name = ''
            frame2 = cv2.resize(cropped,dim,interpolation=cv2.INTER_AREA) 
            var1 = model1.predict(frame2, confidence=40, overlap=30).json()
            var2 = var1['predictions']
            if len(var2) > 0:
                sango = 0
                for v in var2:
                    newx1 = (v['x']-int(v['width']/2))
                    newx1 = int(newx1)
                    newx2 = (v['x']+int(v['width']/2))
                    newx2 = int(newx2)
                    newy1 = (v['y']-int(v['height']/2))
                    newy1 = int(newy1)
                    newy2 = (v['y']+int(v['height']/2))
                    newy2 = int(newy2)
                    cropped3 = frame2[newy1:newy2,newx1:newx2]
                    cv2.imshow(str(sango), cropped3)
                    sango += 1
        cv2.imshow("frame",frame)
        key = cv2.waitKey(200)
        if key == 27:
            break
    fresh.release()
    cv2.destroyAllWindows()
    return render(request, 'car.html')



def find(request,cam):
    MODEL = "yolov8n.pt" 
    model = YOLO(MODEL) 
    model.predict(source="0", show=False, stream=True, classes=0)  
    model.fuse()
    CLASS_NAMES_DICT = model.model.names
    person_face_encoding = []
    photo = FindPerson.objects.last().photo
    image_of_person = face_recognition.load_image_file(photo)
    person_face_encoding.append(face_recognition.face_encodings(image_of_person)[0])
    if cam == 'Camera1':
        RTSP_URL = 'rtsp://admin:admin123@192.168.28.2:554/cam/realmonitor?channel=1&subtype=1'
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        video_capture = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    elif cam == 'Camera2':
        RTSP_URL = 'rtsp://admin:admin12345@192.168.28.220:554/cam/realmonitor?channel=1&subtype=1' 
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        video_capture = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    elif cam == 'Camera3':
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fresh = FreshestFrame(video_capture)
    width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) 
    width = int(width)
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = int(height)
    byte_tracker = BYTETracker(BYTETrackerArgs()) 
    box_annotator = BoxAnnotator(thickness=1, text_scale=0.5, text_thickness=1)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cnt = flag = o = 0
    ind = -1
    tekseru_id = -1
    while True:
        cnt,frame = fresh.read(seqnumber=cnt+1)
        results = model(frame)  
        detections = Detections( 
            xyxy=results[0].boxes.xyxy.cpu().numpy(), 
            confidence=results[0].boxes.conf.cpu().numpy(), 
            class_id=results[0].boxes.cls.cpu().numpy().astype(int) 
        )
        
        detections = detections[detections.class_id==0]
        if(detections.xyxy.__len__==0):
            flag = 0
        tracks = byte_tracker.update( 
            output_results=detections2boxes(detections=detections), 
            img_info=frame.shape, 
            img_size=frame.shape 
        ) 
        
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)     
        detections.tracker_id = np.array(tracker_id) 
        #filtering out detections without trackers 
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool) 
        detections.filter(mask=mask, inplace=True) 
        # format custom labels 
        labels = [ 
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" 
            for _, confidence, class_id, tracker_id 
            in detections 
        ]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels) 

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        j = 1

        o = 0
        for i in detections.xyxy:
            x1 = int(i[0])
            x2 = int(i[2])
            y1 = int(i[1])
            y2 = int(i[3])       
            cropped = frame[y1:y2,x1:x2]
            scale_percent = 2.20
            newwidth = int(cropped.shape[0] * scale_percent )
            newheight = int(cropped.shape[1] * scale_percent )
            dim = (newheight, newwidth)
            name = ''
            frame2 = cv2.resize(cropped,dim,interpolation=cv2.INTER_AREA) 
            if process_this_frame:
                face_locations = face_recognition.face_locations(frame2)
                face_encodings = face_recognition.face_encodings(
                    frame2, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        person_face_encoding, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(
                        person_face_encoding, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        print('Человек найден')
                        ind = o
                        flag = 1
                        tekseru_id = detections.tracker_id[ind]
                        break
                    else:
                        flag = 0

            process_this_frame = not process_this_frame

            o+=1

        if flag == 1 and detections.__len__() > 0:
            x11 = int(detections.xyxy[ind,0])                    
            x21 = int(detections.xyxy[ind,2])
            y11 = int(detections.xyxy[ind,1])
            y21 = int(detections.xyxy[ind,3]) 
            cropped1 = frame[y11:y21,x11:x21]
            scale_percent1 = 2.20 # percent of original size
            newwidth1 = int(cropped1.shape[0] * scale_percent1 )
            newheight1= int(cropped1.shape[1] * scale_percent1 )
            dim1 = (newheight1, newwidth1)
            frame3 = cv2.resize(cropped1,dim1,interpolation=cv2.INTER_AREA) 
            cv2.imshow('Adam', frame3)
            if(tekseru_id != detections.tracker_id[ind]):
                flag = 0
                ind = -1
                cv2.destroyWindow('Adam')
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    FindPerson.objects.filter(photo=photo).delete()

    return render(request, 'find.html')


def tablo(request):
    scanned = LastFace.objects.all().order_by('date').reverse()
    present = Profile.objects.filter(st='pr').order_by('updated').reverse()
    absent = Profile.objects.filter(st='abs').order_by('shift')
    late = Profile.objects.filter(st='late').order_by('updated').reverse()
    last_face = LastFace.objects.last()
    profile = Profile.objects.get(image__icontains=last_face)

    context = {
        'scanned': scanned,
        'present': present,
        'absent': absent,
        'late': late,
        'profile': profile
    }
    return render(request, 'auth_system/tablo.html', context)

def mainpage(request, cam):
    LastFace.objects.all().delete()
    return render(request, 'auth_system/mainpage.html', {'cam':cam})

def ajax(request):
    last_face = LastFace.objects.last()
    context = {
        'last_face': last_face
    }
    return render(request, 'auth_system/ajax.html', context)

def scan(request, cam):
    MODEL = "yolov8n.pt" 

    model = YOLO(MODEL) 
    model.predict(source="0", show=False, stream=True, classes=0)  
    model.fuse()
    global last_face
    CLASS_NAMES_DICT = model.model.names 
    known_face_encodings = []
    known_face_names = []

    profiles = Profile.objects.all()
    for profile in profiles:
        person = profile.image
        person_face_encoding = np.loadtxt(f'arrays/{profile.first_name}.txt')
        known_face_encodings.append(person_face_encoding)
        known_face_names.append(f'{person}'[:-4])
    print('Известные лица:', known_face_names)

    if cam == 'Camera1':
        RTSP_URL = 'rtsp://admin:admin123@192.168.28.2:554/cam/realmonitor?channel=1&subtype=1'
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        video_capture = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    elif cam == 'Camera2':
        RTSP_URL = 'rtsp://admin:admin12345@192.168.28.220:554/cam/realmonitor?channel=1&subtype=1' 
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        video_capture = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    elif cam == 'Camera3':
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fresh = FreshestFrame(video_capture)
    byte_tracker = BYTETracker(BYTETrackerArgs()) 
    box_annotator = BoxAnnotator(thickness=1, text_scale=0.5, text_thickness=1)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cnt = 0
    while True:
        cnt,frame = fresh.read(seqnumber=cnt+1)
        results = model(frame)  
        detections = Detections( 
            xyxy=results[0].boxes.xyxy.cpu().numpy(), 
            confidence=results[0].boxes.conf.cpu().numpy(), 
            class_id=results[0].boxes.cls.cpu().numpy().astype(int) 
        )
        detections = detections[detections.class_id==0]
        tracks = byte_tracker.update( 
            output_results=detections2boxes(detections=detections), 
            img_info=frame.shape, 
            img_size=frame.shape 
        ) 
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)     
        detections.tracker_id = np.array(tracker_id) 
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool) 
        detections.filter(mask=mask, inplace=True) 
        labels = [ 
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" 
            for _, confidence, class_id, tracker_id 
            in detections 
        ]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels) 
        for i in detections.xyxy:
            x1 = int(i[0])
            x2 = int(i[2])
            y1 = int(i[1])
            y2 = int(i[3])       
            cropped = frame[y1:y2,x1:x2]
            scale_percent = 2.20 
            newwidth = int(cropped.shape[0] * scale_percent )
            newheight = int(cropped.shape[1] * scale_percent )
            dim = (newheight, newwidth)
            name = ''
            frame2 = cv2.resize(cropped,dim,interpolation=cv2.INTER_AREA) 
            if process_this_frame:
                face_locations = face_recognition.face_locations(frame2)
                face_encodings = face_recognition.face_encodings(
                    frame2, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        print('Лучшее совпадение:', name)
                        profile = Profile.objects.get(image__icontains=name)
                        print('Профиль этого лица:', profile)
                        if profile.st == 'pr' or profile.st == 'late':
                            print('Вы уже отмечены как', profile.st)
                        else:
                            print('Сейчас мы вас отметим.')
                            if datetime.now().time() > profile.shift:
                                profile.st = 'late'
                            else:
                                profile.st = 'pr'
                            profile.save()
                        last_face = LastFace(last_face=name)
                        last_face.save()
                        last_face = name
                    face_names.append(name)

            process_this_frame = not process_this_frame

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return HttpResponse('закрываем сканер', last_face)

def login(request, cam):
    try:
        last_face = LastFace.objects.last()
        profile = Profile.objects.get(image__icontains=last_face)  
        return redirect('tablo')
    except:
        last_face = None
        profile = None
        context = {
            'profile': profile,
            'last_face': last_face
        }
        return render(request, 'auth_system/login.html', context)


def register(request):
    form = ProfileForm
    if request.method == 'POST':
        form = ProfileForm(request.POST,request.FILES)
        if form.is_valid():
            print('Форма правильно заполнена')
            form.save()
            image = form.cleaned_data['image']
            first_name = form.cleaned_data['first_name']
            image_of_person = face_recognition.load_image_file(image)
            face_encoding = face_recognition.face_encodings(image_of_person)[0]
            np.savetxt(f"arrays/{first_name}.txt", face_encoding)
            return redirect('mainpage')
        else:
            print('Форма неправильно заполнена:')
            print(form.errors)
    context ={'form':form}
    return render(request,'auth_system/register.html',context)