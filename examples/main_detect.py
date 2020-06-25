import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass
    
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

import time
import numpy as np
import cv2
import pygame
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import h5py
import argparse
import datetime
from yolo import YOLO, detect_video
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from shapely.geometry import box, Polygon
from Advanced_Lane_Lines_Detection import process_line_detection

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import queue
except ImportError:
    import Queue as queue

weights_path = 'model_data/yolo.h5'
anchors_path = 'model_data/yolo_anchors.txt'
classes_path = 'model_data/coco_classes.txt'

#Loading the Yolov3 model used for the detection
yolo = YOLO(
        model_path = weights_path,
        anchors_path = anchors_path,
        classes_path = classes_path,
        model_image_size = (416, 416),
        score = 0.3,
        iou = 0.45,
        )

camera_w1=1280
camera_h1=720

camera_w2=1280
camera_h2=720

#Defining the CarlaSyncMode class with all its function definition
class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

#Function used to verify if an obstacle is in the action range of our vehicle                
def is_obstacle_detected(poly1, poly2):
    # Define Each polygon
    polygon1_shape = Polygon(poly1)
    polygon2_shape = Polygon(poly2)

    # Calculate Intersection between the 2 polygons and return True if it is greater then a threshold
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    threshold = 0.33
    return (polygon_intersection / polygon1_shape.area > threshold)

#Function used to convert the detected rectangle to a more general polygon with the same dimensions
def convert_rect_to_poly(rect):
    y1,x1 = rect[0], rect[1]
    y2,x2 = rect[2], rect[3]
    return np.array([[x1,y1],[x1,y2], [x2,y2], [x2,y1]])


#Function that takes as argument a CarlaImage taken by the rgb_camera sensor and execute the detection through Yolo
#It returns the image and strLight, out_boxes and predicted_classes used in other functions
#strLight = the color of the detected traffic light
#out_boxes = a list containing all the boxes detected
#predicted_classes = a list containing the predicted class for each box
def obstacle_img(image):
    
    #passage useful to transform the CarlaImage into a cv2 image
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = img.reshape(camera_h1, camera_w1, 4) #4 dimensions because the image also has the 'alpha' channel, that we will ignore for now
    img = img[:,:,:3]
    img = img[:, :, ::-1]
    
    #transformation of cv2 image into pil image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(rgb_img)
       
    r_image, strLight, out_boxes, predicted_classes = yolo.detect_image(im_pil)     #detection

    cv2_img = np.array(r_image)

    try:
        cv2_img = process_line_detection(cv2_img)    #fucntion imported from another script to draw the lane detection on the image
    except:
        pass
    
    return cv2_img, strLight, out_boxes, predicted_classes

#Function to tranform a CarlaImage into a cv2 image, used for the spectator to follow the vehicle      
def process_img(image):

    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = img.reshape(image.height, image.width, 4) #4 dimensions because the image also has the 'alpha' channel, that we will ignore for now
    img = img[:,:,:3]

    return img
            

#Function that closes the video streams, and destroy the actors list when the simulation has finished            
def close_session(actor_list, camera_rgb_video_out, camera_view_video_out):
    camera_rgb_video_out.release()
    camera_view_video_out.release()
    print('destroying actors.')
    for actor in actor_list:
        actor.destroy()

    pygame.quit()
    print('done.')

def main():
    actor_list = []
    
    pygame.init()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    #world = client.load_world('Town02')
    
    #world.set_weather(carla.WeatherParameters.ClearNoon) #WetCloudySunset
    
    #carla.WeatherParameters.sun_altitude_angle=-90

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(blueprint_library.find('vehicle.tesla.model3'), start_pose)
        actor_list.append(vehicle)
        #vehicle.set_simulate_physics(False)
        
        #Set spectator location
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=10)))
        carla.Rotation(pitch=-90)

        
        #Define sensors
        
        #First RGB camera to perform object and lane detection

        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", f"{camera_w1}")
        cam_bp.set_attribute("image_size_y", f"{camera_h1}")
        cam_bp.set_attribute("fov", "120")

        camera_rgb = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=2.5, z=1.5)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        
        #Second RGB camera to follow the vehicle
        
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", f"{camera_w2}")
        cam_bp.set_attribute("image_size_y", f"{camera_h2}")
        cam_bp.set_attribute("fov", "120")
        
        camera_view = world.spawn_actor(
           cam_bp,
           carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
           attach_to=vehicle)
        actor_list.append(camera_view)
        
        #Basic agent
        agent = BasicAgent(vehicle, target_speed=18)
        #Set agent destination
        spawn_point = m.get_spawn_points()[0]
        destination = (spawn_point.location.x, spawn_point.location.y, spawn_point.location.z)
        agent.set_destination(destination)
        
        #Video generation using the images taken from the 2 camera sensors
        curr_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        camera_rgb_video_out = cv2.VideoWriter('camera_rgb_'+curr_date+'.avi', fourcc, 30, (640, 480))
        camera_view_video_out = cv2.VideoWriter('camera_view_'+curr_date+'.avi', fourcc, 30, (640, 480))
        
        #Queue for obstacles
        obstacle_brake = False
        obstacle_slow = False

        #List used for the majority rule for the obstacle detection. Has as many elements as are the simulated_fps 
        simulated_fps=30
        obstacle_hist = []
        for i in range(0,simulated_fps):
            obstacle_hist.append(0)
        
        
        #Dynamic polygon used to calculate the obstacle in the vehicle range
        imshape = (camera_h1, camera_w1, 3)
        x1 = 30
        oh = 0
        h = int(imshape[0]/1.75) #1.65
        l = int(imshape[1]/2.25) #2.5
        
        previous_control=None
        steering=0
        
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_view, fps=simulated_fps) as sync_mode:
            while True:
            
                clock.tick()

                # Advance the simulation and wait for the data.
                
                start_time = time.time()
                snapshot, image_rgb, image_view = sync_mode.tick(timeout=1.0) #, img_view
                print('time tick:', time.time() - start_time)
                
                ## Image process
                
                #Reshape the images
                image_rgb, strLight, out_boxes, predicted_classes = obstacle_img(image_rgb)  #Call of the function used for the detection
                
                is_obstacle = False   #boolean used to check if an obstacle in range is detected
                
                steering=min(steering*2, l)
               
                dynamic_poly = np.array([(x1,imshape[0]-oh),(int(l*(1+steering)), h), (int(imshape[1]-l*(1-steering)), h), (imshape[1]-x1,imshape[0]-oh)])
                
                
                #check if an obstacle is in the accepted list of obstacle and update the boolean
                for box in zip(out_boxes, predicted_classes):
                    
                    out_class=box[1]

                    if out_class in ["car","person","truck","bicycle","train", "motorbike","bus"]:
                        obstacle_poly = convert_rect_to_poly(box[0])   #convertion of the bounding box to a polygon
                        
                        if is_obstacle_detected(obstacle_poly, dynamic_poly):    #calculate the intersection between the 2 polygons and return a boolean
                            is_obstacle = True
                        
                #append the resulting boolean to the majority rule list
                obstacle_hist.pop(0)
                obstacle_hist.append(is_obstacle)
                obstacle_brake = (sum(obstacle_hist)>int(simulated_fps/3)) #brake when 10 frames out of 30 contain the obstacle
                obstacle_slow = (sum(obstacle_hist)>int(simulated_fps/5)) #slow down when 6 frames out of 30 contain the obstacle
                
                #Call of the function for the following vehicle images
                image_view = process_img(image_view)

                
                #Control vehicle
                control = agent.run_step(strLight, obstacle_brake, obstacle_slow)
                vehicle.apply_control(control)
                
                steering=control.steer
                
                #Visualize outputs
                #Write information on the image_view

                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                fontScale              = 0.5
                fontColor              = (255,255,255)
                lineType               = 1
                
                text_steer = 'Steer: '+str(control.steer)
                text_throttle = 'Throttle: '+str(control.throttle)
                text_brake = 'Brake: '+str(control.brake)
                text_traffic_light = 'Traffic light: ' + strLight
                text_obstacle = 'Obstacle detection: ' + str(obstacle_brake)
                
                text_obstacle2 = ''.join(str(e)+' ' for e in obstacle_hist)
                
                image_view=np.array(image_view)
                
                
                cv2.putText(image_view, text_obstacle, (3,620), font, fontScale,fontColor,lineType)
                cv2.putText(image_view, text_traffic_light, (3,640), font, fontScale,fontColor,lineType)
                cv2.putText(image_view, text_steer, (3,660), font, fontScale,fontColor,lineType)
                cv2.putText(image_view, text_throttle, (3,680), font, fontScale,fontColor,lineType)
                cv2.putText(image_view, text_brake, (3,700), font, fontScale,fontColor,lineType)
                

                #Pass the images to create the videos
                image_rgb=cv2.resize(image_rgb, (640,480))
                image_view=cv2.resize(image_view, (640,480))
                camera_rgb_video_out.write(image_rgb)
                camera_view_video_out.write(image_view)

                print('Client FPS:', clock.get_fps())
                
                if vehicle.get_location() == destination:
                    print("Target reached, mission accomplished...")
                    break
                    
    finally:
        close_session(actor_list, camera_rgb_video_out, camera_view_video_out)
    
    close_session(actor_list, camera_rgb_video_out, camera_view_video_out)
    
if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')