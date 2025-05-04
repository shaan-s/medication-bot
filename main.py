from deepface import DeepFace
import cv2
import pygame
from pygame_widgets.button import Button
import pygame_widgets

page = "menu"

def set_screen(name):
    global page
    page = name

pygame.init()

cam = cv2.VideoCapture(0)

COL_BG = "#fffdf6"
COL_EMP = "#8fd9e6"
COL_SUB = "#e4554d"
COL_BLK = "#000000"

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(frame_height,frame_width)
screen = pygame.display.set_mode((1000,600))
pygame.display.set_caption("PharmaBuddy Dashboard")
logo_img = pygame.image.load("Logo.png").convert()


run = True
clock = pygame.time.Clock()

now_60 = pygame.font.Font('now-light.otf', 60)
now_48 = pygame.font.Font('now-light.otf', 48)
now_36 = pygame.font.Font('now-light.otf', 36)
now_24 = pygame.font.Font('now-light.otf', 24)
now_18 = pygame.font.Font('now-light.otf', 18)

model = DeepFace.build_model("Facenet512")

# number of ticks to save photos for facial recognition (30 fps)
interval = 90

cnt_interval = 0
result = ""

while run:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            run = False

    ret, frame = cam.read()

    screen.blit(logo_img,(250,50))

    b1 = Button(
    screen, 100, 100, 200, 100, text='Start',
    fontSize=50, margin=20, font=now_24,
    inactiveColour=COL_EMP, radius=5,
    onClick=lambda: set_screen("process"))

    b2 = Button(
    screen, 400, 100, 200, 100, text='New User',
    fontSize=50, margin=20, font=now_24,
    inactiveColour=COL_EMP, radius=5,
    onClick=lambda: set_screen("add"))

    b3 = Button(
    screen, 700, 100, 200, 100, text='Manage',
    fontSize=50, margin=20, font=now_24,
    inactiveColour=COL_EMP, radius=5,
    onClick=lambda: set_screen("manage"))

    back = Button(
    screen, 450, 540, 100, 50, text='Go back',
    fontSize=50, margin=20, font=now_18,
    inactiveColour=COL_EMP, radius=5,
    onClick=lambda: set_screen("menu"))



    if page == "process": 
        if cnt_interval == interval:
            cnt_interval = 0

            df_result = DeepFace.verify(img1_path=cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5), img2_path = "users\\shaan\\ba1.jpg", enforce_detection=False, model_name="Facenet512")
            result = str(df_result['verified'])
    
        else:
            cnt_interval += 1

        frame_surface = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "BGR")

        screen.blit(frame_surface, (180,50))

        text_surface = now_60.render(result, True, (255, 255, 255))
        screen.blit(text_surface, (600,600))

    elif page == "add":
        text_surface = now_24.render("You need 3 photos to train the device to recognize your face! ", True, COL_BLK)
        screen.blit(text_surface, (20,20))
        text_surface = now_24.render("Try to use a variety of angles and lighting ", True, COL_BLK)
        screen.blit(text_surface, (20,50))

        frame_surface = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "BGR")
        scaled = pygame.transform.scale_by(frame_surface,0.75)

        screen.blit(scaled, (100,120))
    

    if page == "menu":
        text_surface = now_60.render("PharmaBuddy Menu ", True, COL_EMP)
        screen.blit(text_surface, (40,20))
        del back
    else:
        del b1
        del b2
        del b3

    pygame_widgets.update(events)

    pygame.display.update()
    screen.fill((COL_BG))

    clock.tick(30) #30 fps

cam.release()
pygame.quit()
    