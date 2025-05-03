from deepface import DeepFace
import cv2
import pygame

pygame.init()

cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

screen = pygame.display.set_mode((frame_width+400,frame_height+400))
pygame.display.set_caption("Medbuddy Dashboard")

run = True
clock = pygame.time.Clock()

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    ret, frame = cam.read()

    # Display the captured frame
    #cv2.imshow('Camera', frame)

    frame_surface = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "BGR")

    screen.blit(frame_surface, (0,0))

    pygame.display.update()
    screen.fill((0,0,0))

    clock.tick(30)

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
pygame.quit()
    